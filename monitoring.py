import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, List, Optional
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, db_path: str = "model_monitoring.db"):
        """Initialize the model monitor.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()
        
    def _init_db(self):
        """Initialize the SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create predictions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            prediction INTEGER,
            confidence FLOAT,
            ground_truth INTEGER,
            features TEXT,
            model_version TEXT
        )
        ''')
        
        # Create metrics table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            metric_name TEXT,
            metric_value FLOAT,
            window_size INTEGER
        )
        ''')
        
        # Create alerts table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            alert_type TEXT,
            message TEXT,
            severity TEXT,
            additional_data TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_prediction(self, 
                      prediction: int,
                      confidence: float,
                      ground_truth: Optional[int] = None,
                      features: Optional[Dict] = None,
                      model_version: str = "1.0.0"):
        """Log a prediction to the database.
        
        Args:
            prediction: Model's prediction
            confidence: Confidence score
            ground_truth: True label (if available)
            features: Dictionary of input features
            model_version: Version of the model
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO predictions (timestamp, prediction, confidence, ground_truth, features, model_version)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            prediction,
            confidence,
            ground_truth,
            json.dumps(features) if features else None,
            model_version
        ))
        
        conn.commit()
        conn.close()
    
    def calculate_metrics(self, window_size: int = 1000) -> Dict[str, float]:
        """Calculate performance metrics for the most recent predictions.
        
        Args:
            window_size: Number of most recent predictions to consider
            
        Returns:
            Dictionary of metric names and values
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get all recent predictions regardless of ground truth
        query_all = f'''
        SELECT prediction, confidence, ground_truth, timestamp
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {window_size}
        '''
        
        df_all = pd.read_sql_query(query_all, conn)
        
        if len(df_all) == 0:
            conn.close()
            return {}
        
        # Calculate metrics that don't require ground truth
        metrics = {
            'total_predictions': len(df_all),
            'avg_confidence': float(df_all['confidence'].mean()),
            'min_confidence': float(df_all['confidence'].min()),
            'max_confidence': float(df_all['confidence'].max()),
            'last_prediction_time': df_all['timestamp'].iloc[0],
            'unique_classes': len(df_all['prediction'].unique())
        }
        
        # Add prediction distribution
        class_counts = df_all['prediction'].value_counts().to_dict()
        for class_id, count in class_counts.items():
            metrics[f'class_{class_id}_count'] = count
            metrics[f'class_{class_id}_pct'] = float(count / len(df_all))
        
        # If ground truth is available, calculate standard ML metrics
        if 'ground_truth' in df_all.columns and df_all['ground_truth'].notna().any():
            # Get only predictions with ground truth
            df_with_truth = df_all.dropna(subset=['ground_truth'])
            
            if len(df_with_truth) > 0:
                # Convert to integers for metrics calculation
                y_true = df_with_truth['ground_truth'].astype(int)
                y_pred = df_with_truth['prediction'].astype(int)
                
                # Calculate standard ML metrics
                metrics.update({
                    'accuracy': float(accuracy_score(y_true, y_pred)),
                    'precision': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'recall': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'f1': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
                    'ground_truth_count': len(df_with_truth)
                })
        
        # Log metrics to database
        cursor = conn.cursor()
        for metric_name, metric_value in metrics.items():
            # Skip non-numeric metrics for database storage
            if isinstance(metric_value, (int, float)):
                cursor.execute('''
                INSERT INTO metrics (timestamp, metric_name, metric_value, window_size)
                VALUES (?, ?, ?, ?)
                ''', (datetime.now(), metric_name, metric_value, window_size))
        
        conn.commit()
        conn.close()
        
        return metrics
    
    def detect_drift(self, 
                    window_size: int = 100,
                    confidence_threshold: float = 0.05,
                    distribution_threshold: float = 0.2) -> Dict[str, bool]:
        """Detect drift in model performance and predictions.
        
        Args:
            window_size: Number of most recent predictions to consider
            confidence_threshold: Threshold for confidence drift detection
            distribution_threshold: Threshold for distribution drift detection
            
        Returns:
            Dictionary of drift indicators
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get recent predictions for current window
        current_query = f'''
        SELECT prediction, confidence, timestamp
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {window_size}
        '''
        
        # Get predictions for previous window
        previous_query = f'''
        SELECT prediction, confidence, timestamp
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {window_size} OFFSET {window_size}
        '''
        
        df_current = pd.read_sql_query(current_query, conn)
        df_previous = pd.read_sql_query(previous_query, conn)
        
        drift_indicators = {
            'drift_detected': False,
            'reasons': []
        }
        
        # Need enough data for comparison
        if len(df_current) < 5 or len(df_previous) < 5:
            conn.close()
            return {'drift_detected': False, 'reason': 'Insufficient data for drift detection'}
        
        # 1. Check for change in average confidence
        current_conf_mean = df_current['confidence'].mean()
        previous_conf_mean = df_previous['confidence'].mean()
        
        conf_change = abs(current_conf_mean - previous_conf_mean)
        if conf_change > confidence_threshold:
            drift_indicators['drift_detected'] = True
            drift_indicators['reasons'].append(
                f'Confidence drift: {conf_change:.4f} (threshold: {confidence_threshold})'
            )
        
        # 2. Check for change in prediction distribution
        current_dist = df_current['prediction'].value_counts(normalize=True).to_dict()
        previous_dist = df_previous['prediction'].value_counts(normalize=True).to_dict()
        
        # Calculate JS divergence between distributions
        all_classes = set(list(current_dist.keys()) + list(previous_dist.keys()))
        
        # Fill missing classes with 0
        for class_id in all_classes:
            if class_id not in current_dist:
                current_dist[class_id] = 0
            if class_id not in previous_dist:
                previous_dist[class_id] = 0
        
        # Sort dictionaries by keys for consistent comparison
        current_values = [current_dist[k] for k in sorted(current_dist.keys())]
        previous_values = [previous_dist[k] for k in sorted(previous_dist.keys())]
        
        # Simple distribution difference metric (mean absolute difference)
        dist_change = sum(abs(c - p) for c, p in zip(current_values, previous_values)) / len(all_classes)
        
        if dist_change > distribution_threshold:
            drift_indicators['drift_detected'] = True
            drift_indicators['reasons'].append(
                f'Distribution drift: {dist_change:.4f} (threshold: {distribution_threshold})'
            )
        
        # 3. Add metrics for monitoring
        drift_indicators['metrics'] = {
            'confidence_change': conf_change,
            'distribution_change': dist_change,
            'current_window_size': len(df_current),
            'previous_window_size': len(df_previous),
            'current_confidence_mean': current_conf_mean,
            'previous_confidence_mean': previous_conf_mean,
        }
        
        conn.close()
        return drift_indicators
    
    def generate_alert(self, 
                      alert_type: str,
                      message: str,
                      severity: str = "WARNING",
                      additional_data: Dict = None):
        """Generate an alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (INFO, WARNING, ERROR)
            additional_data: Additional data to store with the alert
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now()
        
        # Store additional data as JSON if provided
        data_json = None
        if additional_data:
            data_json = json.dumps(additional_data)
        
        # Make sure the alerts table has the additional_data column
        try:
            cursor.execute("SELECT additional_data FROM alerts LIMIT 1")
        except sqlite3.OperationalError:
            # Add additional_data column if it doesn't exist
            cursor.execute("ALTER TABLE alerts ADD COLUMN additional_data TEXT")
        
        cursor.execute('''
        INSERT INTO alerts (timestamp, alert_type, message, severity, additional_data)
        VALUES (?, ?, ?, ?, ?)
        ''', (timestamp, alert_type, message, severity, data_json))
        
        conn.commit()
        conn.close()
        
        # Log alert
        log_level = {
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR
        }.get(severity, logging.WARNING)
        
        logger.log(log_level, f"Alert: {alert_type} - {message} (Severity: {severity})")
    
    def plot_metrics(self, 
                    metric_name: str,
                    window_size: int = 1000,
                    save_path: Optional[str] = None):
        """Plot historical metrics.
        
        Args:
            metric_name: Name of the metric to plot
            window_size: Number of data points to plot
            save_path: Path to save the plot (if None, plot is displayed)
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
        SELECT timestamp, metric_value
        FROM metrics
        WHERE metric_name = ?
        ORDER BY timestamp DESC
        LIMIT {window_size}
        '''
        
        df = pd.read_sql_query(query, conn, params=(metric_name,))
        conn.close()
        
        if len(df) == 0:
            logger.warning(f"No data available for metric: {metric_name}")
            return
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='timestamp', y='metric_value')
        plt.title(f'{metric_name} Over Time')
        plt.xlabel('Time')
        plt.ylabel(metric_name)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        
        plt.close()
    
    def get_recent_alerts(self, limit: int = 10) -> pd.DataFrame:
        """Get recent alerts.
        
        Args:
            limit: Number of recent alerts to retrieve
            
        Returns:
            DataFrame containing recent alerts
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f'''
        SELECT timestamp, alert_type, message, severity
        FROM alerts
        ORDER BY timestamp DESC
        LIMIT {limit}
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df 