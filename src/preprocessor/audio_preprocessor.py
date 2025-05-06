#!/usr/bin/env python3
import os
import numpy as np
import librosa
import pandas as pd
from tqdm import tqdm

def preprocess_audio(input_csv, output_dir, target_sr=22050, n_mels=128):
    """
    音声ファイルを前処理し、メルスペクトログラムを生成する
    """
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    mel_dir = os.path.join(output_dir, 'mels')
    os.makedirs(mel_dir, exist_ok=True)
    
    # 入力CSVファイルを読み込む
    df = pd.read_csv(input_csv)
    
    # 処理結果を保存するリスト
    results = []
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        # オリジナル曲の処理
        original_path = row['original_path']
        original_mel_path = os.path.join(mel_dir, f"original_{row['id']}.npy")
        
        if not os.path.exists(original_mel_path):
            try:
                # 音声の読み込みと前処理
                audio, sr = librosa.load(original_path, sr=target_sr)
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
                mel = librosa.power_to_db(mel, ref=np.max)
                
                # メルスペクトログラムの保存
                np.save(original_mel_path, mel)
            except Exception as e:
                print(f"処理失敗: {original_path}")
                print(f"エラー: {str(e)}")
                continue
        
        # リミックス曲の処理
        remix_path = row['remix_path']
        remix_mel_path = os.path.join(mel_dir, f"remix_{row['id']}.npy")
        
        if not os.path.exists(remix_mel_path):
            try:
                # 音声の読み込みと前処理
                audio, sr = librosa.load(remix_path, sr=target_sr)
                mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
                mel = librosa.power_to_db(mel, ref=np.max)
                
                # メルスペクトログラムの保存
                np.save(remix_mel_path, mel)
            except Exception as e:
                print(f"処理失敗: {remix_path}")
                print(f"エラー: {str(e)}")
                continue
        
        # 結果を保存
        results.append({
            'id': row['id'],
            'original_mel': original_mel_path,
            'remix_mel': remix_mel_path
        })
    
    # 結果をCSVファイルに保存
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(output_dir, 'processed_tracks.csv'), index=False)
    print(f"処理結果を保存しました: {os.path.join(output_dir, 'processed_tracks.csv')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='音声ファイルを前処理する')
    parser.add_argument('input_csv', help='ダウンロードした曲の情報が記載されたCSVファイル')
    parser.add_argument('--output-dir', default='processed', help='処理結果の保存先ディレクトリ')
    parser.add_argument('--target-sr', type=int, default=22050, help='目標サンプリングレート')
    parser.add_argument('--n-mels', type=int, default=128, help='メルスペクトログラムの次元数')
    args = parser.parse_args()
    
    preprocess_audio(args.input_csv, args.output_dir, args.target_sr, args.n_mels) 