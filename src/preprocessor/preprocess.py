#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import logging
from typing import List, Dict
import boto3
from tqdm import tqdm
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, s3_bucket: str, s3_prefix: str):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
        
    def download_from_s3(self, s3_key: str, local_path: str):
        """S3からファイルをダウンロード"""
        try:
            self.s3_client.download_file(self.s3_bucket, s3_key, local_path)
            logger.info(f"ダウンロード成功: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"ダウンロードエラー: {s3_key} - {str(e)}")
            return False
    
    def upload_to_s3(self, local_path: str, s3_key: str):
        """ファイルをS3にアップロード"""
        try:
            self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
            logger.info(f"アップロード成功: {s3_key}")
            return True
        except Exception as e:
            logger.error(f"アップロードエラー: {s3_key} - {str(e)}")
            return False
    
    def process_audio(self, audio_path: str, target_sr: int = 22050, n_mels: int = 128) -> np.ndarray:
        """音声ファイルを前処理"""
        try:
            # 音声の読み込み
            audio, sr = librosa.load(audio_path, sr=target_sr)
            
            # メルスペクトログラムの計算
            mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
            mel = librosa.power_to_db(mel, ref=np.max)
            
            return mel
        except Exception as e:
            logger.error(f"音声処理エラー: {audio_path} - {str(e)}")
            return None
    
    def process_tracks(self, input_dir: str, output_dir: str, target_sr: int = 22050, n_mels: int = 128) -> List[Dict]:
        """トラックペアを処理"""
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        mel_dir = os.path.join(output_dir, 'mels')
        os.makedirs(mel_dir, exist_ok=True)
        
        # track_pairs.csvの読み込み
        csv_path = os.path.join(input_dir, 'track_pairs.csv')
        if not os.path.exists(csv_path):
            logger.error("track_pairs.csvが見つかりません")
            return []
        
        df = pd.read_csv(csv_path)
        processed_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df)):
            try:
                # リミックス音源の処理
                remix_path = os.path.join(input_dir, f"remix_{row['remix_url'].split('/')[-1]}.mp3")
                remix_mel_path = os.path.join(mel_dir, f"remix_{row['id']}.npy")
                
                if os.path.exists(remix_path) and not os.path.exists(remix_mel_path):
                    remix_mel = self.process_audio(remix_path, target_sr, n_mels)
                    if remix_mel is not None:
                        np.save(remix_mel_path, remix_mel)
                
                # オリジナル音源の処理
                original_path = os.path.join(input_dir, f"original_{row['original_name']}.mp3")
                original_mel_path = os.path.join(mel_dir, f"original_{row['id']}.npy")
                
                if os.path.exists(original_path) and not os.path.exists(original_mel_path):
                    original_mel = self.process_audio(original_path, target_sr, n_mels)
                    if original_mel is not None:
                        np.save(original_mel_path, original_mel)
                
                processed_data.append({
                    'id': row['id'],
                    'original_mel': original_mel_path if os.path.exists(original_mel_path) else None,
                    'remix_mel': remix_mel_path if os.path.exists(remix_mel_path) else None,
                    'status': 'success'
                })
                
            except Exception as e:
                logger.error(f"処理エラー: {row['remix_url']} - {str(e)}")
                processed_data.append({
                    'id': row['id'],
                    'status': 'error',
                    'error': str(e)
                })
        
        # 処理結果をCSVに保存
        processed_df = pd.DataFrame(processed_data)
        processed_csv = os.path.join(output_dir, 'processed_tracks.csv')
        processed_df.to_csv(processed_csv, index=False)
        logger.info(f"処理結果を保存しました: {processed_csv}")
        
        return processed_data

class SageMakerPreprocessor:
    def __init__(self, role: str, instance_type: str = 'ml.m5.xlarge'):
        self.role = role
        self.instance_type = instance_type
        self.sagemaker_session = sagemaker.Session()
        
    def run_preprocessing(self, 
                         input_data: str,
                         output_data: str,
                         code_path: str = 'preprocess.py'):
        """SageMaker Processing Jobを実行"""
        sklearn_processor = SKLearnProcessor(
            framework_version='0.23-1',
            role=self.role,
            instance_type=self.instance_type,
            instance_count=1,
            base_job_name='audio-preprocessing',
            sagemaker_session=self.sagemaker_session
        )
        
        sklearn_processor.run(
            code=code_path,
            inputs=[
                ProcessingInput(
                    source=input_data,
                    destination='/opt/ml/processing/input'
                )
            ],
            outputs=[
                ProcessingOutput(
                    source='/opt/ml/processing/output',
                    destination=output_data
                )
            ],
            arguments=[
                '--input-data', '/opt/ml/processing/input',
                '--output-data', '/opt/ml/processing/output',
                '--target-sr', '22050',
                '--n-mels', '128'
            ]
        )

def main():
    # 環境変数から設定を読み込み
    s3_bucket = os.environ.get('S3_BUCKET')
    s3_prefix = os.environ.get('S3_PREFIX', 'raw_data')
    role = os.environ.get('SAGEMAKER_ROLE')
    use_sagemaker = os.environ.get('USE_SAGEMAKER', 'false').lower() == 'true'
    target_sr = int(os.environ.get('TARGET_SR', '22050'))
    n_mels = int(os.environ.get('N_MELS', '128'))
    
    if not s3_bucket:
        raise ValueError("S3_BUCKET environment variable is required")
    
    if use_sagemaker and not role:
        raise ValueError("SAGEMAKER_ROLE environment variable is required when using SageMaker")
    
    # ローカルディレクトリの設定
    input_dir = 'input'
    output_dir = 'output'
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # S3からtrack_pairs.csvをダウンロード
    preprocessor = AudioPreprocessor(s3_bucket, s3_prefix)
    csv_key = 'track_pairs.csv'  # S3バケット直下から取得
    csv_path = os.path.join(input_dir, 'track_pairs.csv')
    
    if not preprocessor.download_from_s3(csv_key, csv_path):
        raise ValueError("Failed to download track_pairs.csv from S3")
    
    if use_sagemaker:
        # SageMaker Processing Jobを実行
        sm_preprocessor = SageMakerPreprocessor(role)
        sm_preprocessor.run_preprocessing(
            input_data=f"s3://{s3_bucket}/{s3_prefix}",
            output_data=f"s3://{s3_bucket}/{s3_prefix}/processed"
        )
    else:
        # ローカルで前処理を実行
        processed_data = preprocessor.process_tracks(
            input_dir, 
            output_dir,
            target_sr=target_sr,
            n_mels=n_mels
        )
        
        # 処理結果をS3にアップロード
        if processed_data:
            # メルスペクトログラムをS3にアップロード
            mel_dir = os.path.join(output_dir, 'mels')
            for file in os.listdir(mel_dir):
                if file.endswith('.npy'):
                    local_path = os.path.join(mel_dir, file)
                    s3_key = f"{s3_prefix}/processed/mels/{file}"
                    preprocessor.upload_to_s3(local_path, s3_key)
            
            # processed_tracks.csvをS3バケット直下にアップロード
            processed_csv = os.path.join(output_dir, 'processed_tracks.csv')
            if os.path.exists(processed_csv):
                preprocessor.upload_to_s3(processed_csv, 'processed_tracks.csv')
    
    logger.info("前処理が完了しました")

if __name__ == "__main__":
    main() 