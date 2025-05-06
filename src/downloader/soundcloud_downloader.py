#!/usr/bin/env python3
import os
import requests
from bs4 import BeautifulSoup
import subprocess
import pandas as pd
from datetime import datetime

def download_tracks(track_pairs_file, output_dir):
    """
    track_pairs.csvから曲のペアを読み込み、ダウンロードする
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 曲のペアを読み込む
    df = pd.read_csv(track_pairs_file)
    
    # ダウンロード結果を保存するリスト
    results = []
    
    for _, row in df.iterrows():
        original_url = row['original_url']
        remix_url = row['remix_url']
        
        # オリジナル曲のダウンロード
        original_path = os.path.join(output_dir, f"original_{row['id']}.wav")
        if not os.path.exists(original_path):
            try:
                subprocess.run(['scdl', '-l', original_url, '-f', original_path], check=True)
                print(f"ダウンロード成功: {original_path}")
            except subprocess.CalledProcessError as e:
                print(f"ダウンロード失敗: {original_url}")
                print(f"エラー: {str(e)}")
                continue
        
        # リミックス曲のダウンロード
        remix_path = os.path.join(output_dir, f"remix_{row['id']}.wav")
        if not os.path.exists(remix_path):
            try:
                subprocess.run(['scdl', '-l', remix_url, '-f', remix_path], check=True)
                print(f"ダウンロード成功: {remix_path}")
            except subprocess.CalledProcessError as e:
                print(f"ダウンロード失敗: {remix_url}")
                print(f"エラー: {str(e)}")
                continue
        
        # 結果を保存
        results.append({
            'id': row['id'],
            'original_path': original_path,
            'remix_path': remix_path
        })
    
    # 結果をCSVファイルに保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'downloaded_tracks.csv'), index=False)
    print(f"ダウンロード結果を保存しました: {os.path.join(output_dir, 'downloaded_tracks.csv')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='SoundCloudから曲をダウンロードする')
    parser.add_argument('track_pairs_file', help='曲のペアが記載されたCSVファイル')
    parser.add_argument('--output-dir', default='downloads', help='ダウンロードした曲の保存先ディレクトリ')
    args = parser.parse_args()
    
    download_tracks(args.track_pairs_file, args.output_dir) 