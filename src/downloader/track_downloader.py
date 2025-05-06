#!/usr/bin/env python3
import os
import subprocess
import boto3
import pandas as pd
import logging
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrackDownloader:
    def __init__(self, s3_bucket: str, s3_prefix: str):
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        self.s3_client = boto3.client('s3')
        
    def download_tracks(self, csv_file: str, output_dir: str) -> List[Dict]:
        """CSVファイルに基づいてトラックをダウンロード"""
        os.makedirs(output_dir, exist_ok=True)
        df = pd.read_csv(csv_file)
        
        # 必要なカラムの存在を確認
        required_columns = ['remix_url']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"CSVファイルに必要なカラムがありません: {missing_columns}")
        
        downloaded_data = []
        for _, row in df.iterrows():
            try:
                # リミックス音源のダウンロード
                logger.info(f"リミックス音源をダウンロード中: {row['remix_url']}")
                subprocess.run([
                    "scdl",
                    "-l", row['remix_url'],
                    "--path", output_dir,
                    "--max-size", "10m",
                    "-c",
                    "--overwrite"
                ], check=True)
                
                downloaded_data.append({
                    'remix_url': row['remix_url'],
                    'status': 'success'
                })
                
            except subprocess.CalledProcessError as e:
                logger.error(f"ダウンロードエラー: {e}")
                downloaded_data.append({
                    'remix_url': row['remix_url'],
                    'status': 'error',
                    'error': str(e)
                })
        
        return downloaded_data
    
    def upload_to_s3(self, local_dir: str):
        """ダウンロードしたファイルをS3にアップロード"""
        for file in os.listdir(local_dir):
            if file.endswith(('.mp3', '.wav', '.csv')):
                local_path = os.path.join(local_dir, file)
                s3_key = f"{self.s3_prefix}/{file}"
                
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                    logger.info(f"アップロード成功: {file}")
                except Exception as e:
                    logger.error(f"アップロードエラー: {file} - {str(e)}")

def main():
    # 環境変数から設定を読み込み
    s3_bucket = os.environ.get('S3_BUCKET')
    s3_prefix = os.environ.get('S3_PREFIX', 'raw_data')
    csv_file = os.environ.get('INPUT_CSV', 'track_pairs.csv')
    output_dir = os.environ.get('OUTPUT_DIR', 'downloads')
    
    if not s3_bucket:
        raise ValueError("S3_BUCKET environment variable is required")
    
    # ダウンローダーの初期化
    downloader = TrackDownloader(s3_bucket, s3_prefix)
    
    # トラックのダウンロード
    results = downloader.download_tracks(csv_file, output_dir)
    
    # S3へのアップロード
    downloader.upload_to_s3(output_dir)
    
    # 結果のログ出力
    logger.info(f"ダウンロード完了: {len(results)}ファイル")
    for result in results:
        logger.info(f"Status: {result['status']}")

if __name__ == "__main__":
    main()