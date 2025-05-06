#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import argparse
import sagemaker
from sagemaker.transformer import Transformer
import boto3
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

class ModelGenerator:
    def __init__(self, role: str, model_name: str = None):
        self.role = role
        self.model_name = model_name
        self.sagemaker_session = sagemaker.Session()
        
    def batch_transform(self,
                       input_data: str,
                       output_data: str,
                       instance_type: str = 'ml.m5.xlarge',
                       instance_count: int = 1) -> str:
        """
        SageMaker Batch Transformを実行してバッチ推論を行います。
        
        Args:
            input_data: 入力データのS3パス
            output_data: 出力データのS3パス
            instance_type: インスタンスタイプ
            instance_count: インスタンス数
            
        Returns:
            バッチ変換ジョブの名前
        """
        transformer = Transformer(
            model_name=self.model_name,
            instance_type=instance_type,
            instance_count=instance_count,
            output_path=output_data,
            sagemaker_session=self.sagemaker_session
        )
        
        transformer.transform(
            data=input_data,
            data_type='S3Prefix',
            content_type='application/json',
            split_type='Line'
        )
        
        return transformer.latest_transform_job.job_name
        
    def deploy_endpoint(self,
                       instance_type: str = 'ml.m5.xlarge',
                       initial_instance_count: int = 1) -> str:
        """
        SageMaker Endpointをデプロイします。
        
        Args:
            instance_type: インスタンスタイプ
            initial_instance_count: 初期インスタンス数
            
        Returns:
            エンドポイント名
        """
        predictor = self.model.deploy(
            initial_instance_count=initial_instance_count,
            instance_type=instance_type
        )
        
        return predictor.endpoint_name

def main():
    # 環境変数から設定を読み込み
    role = os.environ.get('SAGEMAKER_ROLE')
    model_name = os.environ.get('MODEL_NAME')
    input_data = os.environ.get('INPUT_DATA')
    output_data = os.environ.get('OUTPUT_DATA')
    
    if not all([role, model_name, input_data, output_data]):
        raise ValueError("Required environment variables are missing")
    
    # 生成器の初期化
    generator = ModelGenerator(role=role, model_name=model_name)
    
    # バッチ変換の実行
    if os.environ.get('BATCH_TRANSFORM', 'false').lower() == 'true':
        job_name = generator.batch_transform(
            input_data=input_data,
            output_data=output_data
        )
        logger.info(f"Batch transform job started: {job_name}")
    
    # エンドポイントのデプロイ
    if os.environ.get('DEPLOY_ENDPOINT', 'false').lower() == 'true':
        endpoint_name = generator.deploy_endpoint()
        logger.info(f"Endpoint deployed: {endpoint_name}")

if __name__ == "__main__":
    main() 