#!/usr/bin/env python3
import os
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

class AudioDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # メルスペクトログラムの読み込み
        original_mel = np.load(self.data.iloc[idx]['original_mel'])
        remix_mel = np.load(self.data.iloc[idx]['remix_mel'])
        
        return torch.FloatTensor(original_mel), torch.FloatTensor(remix_mel)

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

def train_model(model, train_loader, val_loader, num_epochs=100):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    train_losses = []
    val_losses = []
    
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
    
    # 学習曲線のプロット
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.close()
    
    return model

def generate_remix(model, original_audio_path, output_path):
    # 音声の読み込みと前処理
    audio, sr = librosa.load(original_audio_path, sr=22050)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / mel.std()
    
    # モデルによる変換
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        input_tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(device)
        output = model(input_tensor)
        output = output.squeeze().cpu().numpy()
    
    # メルスペクトログラムから音声に変換
    output = output * mel.std() + mel.mean()
    output = librosa.db_to_power(output)
    audio_remix = librosa.feature.inverse.mel_to_audio(output, sr=sr)
    
    # 音声の保存
    librosa.output.write_wav(output_path, audio_remix, sr)

def main():
    # データセットの準備
    dataset = AudioDataset('processed/processed_tracks.csv')
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)
    
    # モデルの学習
    model = RemixGenerator()
    trained_model = train_model(model, train_loader, val_loader)
    
    # モデルの保存
    torch.save(trained_model.state_dict(), 'remix_generator.pth')
    
    # テスト用のリミックス生成
    test_audio = input("リミックスを生成する音声ファイルのパスを入力してください: ")
    output_path = "generated_remix.wav"
    generate_remix(trained_model, test_audio, output_path)
    print(f"リミックスが生成されました: {output_path}")

if __name__ == "__main__":
    main() 