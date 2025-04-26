import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ImageCSVDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_id = self.df.iloc[idx]['img_id']
        label = self.df.iloc[idx]['label']
        img_path = os.path.join(self.img_dir, img_id + '.jpeg')
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label 