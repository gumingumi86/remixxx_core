#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import sagemaker
from sagemaker.pytorch import PyTorch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioDataset(Dataset):
    def __init__(self, csv_file, transform=None, target_size=(128, 1024)):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # メルスペクトログラムの読み込み
        original_mel = np.load(self.data.iloc[idx]['original_mel'])
        remix_mel = np.load(self.data.iloc[idx]['remix_mel'])
        
        # サイズの正規化
        original_mel = self._normalize_size(original_mel)
        remix_mel = self._normalize_size(remix_mel)
        
        return torch.FloatTensor(original_mel), torch.FloatTensor(remix_mel)
    
    def _normalize_size(self, mel):
        # 現在のサイズを取得
        current_height, current_width = mel.shape
        
        # 高さが異なる場合はリサイズ
        if current_height != self.target_size[0]:
            mel = librosa.util.fix_length(mel, size=self.target_size[0], axis=0)
        
        # 幅が異なる場合はリサイズ
        if current_width != self.target_size[1]:
            mel = librosa.util.fix_length(mel, size=self.target_size[1], axis=1)
        
        return mel

class RemixGenerator(nn.Module):
    def __init__(self):
        super(RemixGenerator, self).__init__()
        
        # エンコーダー
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
        )
        
        # デコーダー
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

def train_model(model, train_loader, val_loader, num_epochs=100, output_dir='models'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    train_losses = []
    val_losses = []
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        # 訓練
        model.train()
        train_loss = 0
        for original, remix in train_loader:
            original = original.unsqueeze(1).to(device)
            remix = remix.unsqueeze(1).to(device)
            
            optimizer.zero_grad()
            output = model(original)
            loss = criterion(output, remix)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # 検証
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for original, remix in val_loader:
                original = original.unsqueeze(1).to(device)
                remix = remix.unsqueeze(1).to(device)
                
                output = model(original)
                loss = criterion(output, remix)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}')
        
        # モデルの保存
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(output_dir, f'model_epoch_{epoch+1}.pth'))
    
    # 最終モデルの保存
    torch.save(model.state_dict(), os.path.join(output_dir, 'model_final.pth'))
    
    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_curve.png'))
    plt.close()
    
    return model

class ModelTrainer:
    def __init__(self, role: str, instance_type: str = 'ml.p3.2xlarge'):
        self.role = role
        self.instance_type = instance_type
        self.sagemaker_session = sagemaker.Session()
        
    def train_model(self,
                   input_data: str,
                   output_path: str,
                   hyperparameters: dict = None):
        """
        SageMaker Training Jobを実行してモデルを学習します。
        
        Args:
            input_data: 学習データのS3パス
            output_path: モデルの出力先S3パス
            hyperparameters: ハイパーパラメータの辞書
        """
        if hyperparameters is None:
            hyperparameters = {
                'epochs': 10,
                'batch-size': 32,
                'learning-rate': 0.001
            }
            
        pytorch_estimator = PyTorch(
            entry_point='train.py',
            role=self.role,
            instance_type=self.instance_type,
            instance_count=1,
            framework_version='1.8.1',
            py_version='py3',
            hyperparameters=hyperparameters,
            output_path=output_path,
            sagemaker_session=self.sagemaker_session
        )
        
        pytorch_estimator.fit({'training': input_data})
        
        return pytorch_estimator

def main():
    # 環境変数から設定を読み込み
    role = os.environ.get('SAGEMAKER_ROLE')
    input_data = os.environ.get('TRAINING_DATA')
    output_path = os.environ.get('MODEL_OUTPUT_PATH')
    
    if not all([role, input_data, output_path]):
        raise ValueError("Required environment variables are missing")
    
    # 学習の実行
    trainer = ModelTrainer(role=role)
    estimator = trainer.train_model(
        input_data=input_data,
        output_path=output_path
    )
    
    logger.info("Training job completed successfully")
    
    # モデルのデプロイ（オプション）
    if os.environ.get('DEPLOY_MODEL', 'false').lower() == 'true':
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.m5.xlarge'
        )
        logger.info("Model deployed successfully")

if __name__ == "__main__":
    main() 