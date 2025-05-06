#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import argparse

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

def generate_remix(model_path, input_audio_path, output_path, target_sr=22050, n_mels=128):
    """
    学習済みモデルを使用してリミックスを生成する
    """
    # モデルの読み込み
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RemixGenerator()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # 音声の読み込みと前処理
    audio, sr = librosa.load(input_audio_path, sr=target_sr)
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)
    mel = (mel - mel.mean()) / mel.std()
    
    # モデルによる変換
    with torch.no_grad():
        input_tensor = torch.FloatTensor(mel).unsqueeze(0).unsqueeze(0).to(device)
        output = model(input_tensor)
        output = output.squeeze().cpu().numpy()
    
    # メルスペクトログラムから音声に変換
    output = output * mel.std() + mel.mean()
    output = librosa.db_to_power(output)
    audio_remix = librosa.feature.inverse.mel_to_audio(output, sr=sr)
    
    # 音声の保存
    sf.write(output_path, audio_remix, sr)
    print(f"リミックスが生成されました: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='学習済みモデルを使用してリミックスを生成する')
    parser.add_argument('model_path', help='学習済みモデルのパス')
    parser.add_argument('input_audio', help='入力音声ファイルのパス')
    parser.add_argument('--output-dir', default='generated', help='生成されたリミックスの保存先ディレクトリ')
    parser.add_argument('--target-sr', type=int, default=22050, help='目標サンプリングレート')
    parser.add_argument('--n-mels', type=int, default=128, help='メルスペクトログラムの次元数')
    args = parser.parse_args()
    
    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 出力ファイル名の生成
    input_filename = os.path.splitext(os.path.basename(args.input_audio))[0]
    output_path = os.path.join(args.output_dir, f"{input_filename}_remix.wav")
    
    # リミックスの生成
    generate_remix(args.model_path, args.input_audio, output_path, args.target_sr, args.n_mels)

if __name__ == "__main__":
    main() 