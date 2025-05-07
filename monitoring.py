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
            severity TEXT
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
        
        # Get recent predictions with ground truth
        query = f'''
        SELECT prediction, ground_truth
        FROM predictions
        WHERE ground_truth IS NOT NULL
        ORDER BY timestamp DESC
        LIMIT {window_size}
        '''
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) == 0:
            return {}
        
        metrics = {
            'accuracy': accuracy_score(df['ground_truth'], df['prediction']),
            'precision': precision_score(df['ground_truth'], df['prediction'], average='weighted'),
            'recall': recall_score(df['ground_truth'], df['prediction'], average='weighted'),
            'f1': f1_score(df['ground_truth'], df['prediction'], average='weighted')
        }
        
        # Log metrics to database
        cursor = conn.cursor()
        for metric_name, metric_value in metrics.items():
            cursor.execute('''
            INSERT INTO metrics (timestamp, metric_name, metric_value, window_size)
            VALUES (?, ?, ?, ?)
            ''', (datetime.now(), metric_name, metric_value, window_size))
        
        conn.commit()
        conn.close()
        
        return metrics
    
    def detect_drift(self, 
                    window_size: int = 1000,
                    threshold: float = 0.05) -> Dict[str, bool]:
        """Detect drift in model performance and predictions.
        
        Args:
            window_size: Number of most recent predictions to consider
            threshold: Threshold for drift detection
            
        Returns:
            Dictionary of drift indicators
        """
        conn = sqlite3.connect(self.db_path)
        
        # Get recent predictions
        query = f'''
        SELECT prediction, confidence, ground_truth
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT {window_size}
        '''
        
        df = pd.read_sql_query(query, conn)
        
        if len(df) < window_size:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        drift_indicators = {
            'drift_detected': False,
            'reasons': []
        }
        
        # 1. Check for significant change in prediction distribution
        if 'ground_truth' in df.columns and df['ground_truth'].notna().any():
            recent_accuracy = accuracy_score(
                df['ground_truth'].dropna(),
                df['prediction'].loc[df['ground_truth'].notna()]
            )
            
            # Get historical accuracy
            cursor = conn.cursor()
            cursor.execute('''
            SELECT metric_value
            FROM metrics
            WHERE metric_name = 'accuracy'
            ORDER BY timestamp DESC
            LIMIT 1
            ''')
            result = cursor.fetchone()
            
            if result:
                historical_accuracy = result[0]
                if abs(recent_accuracy - historical_accuracy) > threshold:
                    drift_indicators['drift_detected'] = True
                    drift_indicators['reasons'].append(
                        f'Accuracy drift: {abs(recent_accuracy - historical_accuracy):.3f}'
                    )
        
        # 2. Check for significant change in confidence distribution
        if 'confidence' in df.columns:
            recent_conf_mean = df['confidence'].mean()
            recent_conf_std = df['confidence'].std()
            
            # Get historical confidence stats
            cursor.execute('''
            SELECT AVG(confidence), STDDEV(confidence)
            FROM predictions
            WHERE timestamp < (
                SELECT MIN(timestamp)
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT {window_size}
            )
            ''')
            result = cursor.fetchone()
            
            if result and result[0] is not None:
                hist_conf_mean, hist_conf_std = result
                if abs(recent_conf_mean - hist_conf_mean) > threshold:
                    drift_indicators['drift_detected'] = True
                    drift_indicators['reasons'].append(
                        f'Confidence drift: {abs(recent_conf_mean - hist_conf_mean):.3f}'
                    )
        
        conn.close()
        return drift_indicators
    
    def generate_alert(self, 
                      alert_type: str,
                      message: str,
                      severity: str = "WARNING"):
        """Generate an alert.
        
        Args:
            alert_type: Type of alert
            message: Alert message
            severity: Alert severity (INFO, WARNING, ERROR)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO alerts (timestamp, alert_type, message, severity)
        VALUES (?, ?, ?, ?)
        ''', (datetime.now(), alert_type, message, severity))
        
        conn.commit()
        conn.close()
        
        # Log alert
        logger.warning(f"Alert: {alert_type} - {message} (Severity: {severity})")
    
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