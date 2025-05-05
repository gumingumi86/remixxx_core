#!/usr/bin/env python3
import os
import re
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import shutil

class AudioPreprocessor:
    def __init__(self, input_dir, output_dir):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.sr = 22050  # サンプリングレート
        self.duration = 30  # 音声の長さ（秒）
        self.n_mels = 128  # メルスペクトログラムのビン数
        self.n_fft = 2048  # FFTのサイズ
        self.hop_length = 512  # ホップ長
        
        # 出力ディレクトリの作成
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'original').mkdir(exist_ok=True)
        (self.output_dir / 'remix').mkdir(exist_ok=True)
        (self.output_dir / 'melspectrograms').mkdir(exist_ok=True)
    
    def clean_filename(self, filename):
        """ファイル名を正規化"""
        # 拡張子を除去
        name = Path(filename).stem
        
        # 特殊文字を除去
        name = re.sub(r'[^\w\s-]', '', name)
        
        # 連続する空白を1つに
        name = re.sub(r'\s+', ' ', name)
        
        # 前後の空白を除去
        name = name.strip()
        
        return name
    
    def normalize_audio(self, audio):
        """音声の正規化"""
        # 音量の正規化
        audio = librosa.util.normalize(audio)
        
        # DCオフセットの除去
        audio = audio - np.mean(audio)
        
        return audio
    
    def remove_noise(self, audio, sr):
        """ノイズ除去"""
        # 高周波ノイズの除去
        audio = librosa.effects.preemphasis(audio)
        
        # クリッピングの防止
        audio = np.clip(audio, -1.0, 1.0)
        
        return audio
    
    def process_audio_file(self, input_path, output_path, is_remix=False):
        """音声ファイルの処理"""
        try:
            # 音声の読み込み
            audio, sr = librosa.load(input_path, sr=self.sr, duration=self.duration)
            
            # 前処理
            audio = self.normalize_audio(audio)
            audio = self.remove_noise(audio, sr)
            
            # 音声の保存
            sf.write(output_path, audio, sr)
            
            # メルスペクトログラムの生成
            mel = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.n_mels,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # 対数スケールに変換
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # 正規化
            mel_norm = (mel_db - mel_db.mean()) / mel_db.std()
            
            # メルスペクトログラムの保存
            mel_path = self.output_dir / 'melspectrograms' / f"{Path(output_path).stem}_mel.npy"
            np.save(mel_path, mel_norm)
            
            return True
            
        except Exception as e:
            print(f"エラー: {input_path} の処理中にエラーが発生しました: {str(e)}")
            return False
    
    def process_dataset(self, csv_file):
        """データセット全体の処理"""
        import pandas as pd
        
        # CSVファイルの読み込み
        df = pd.read_csv(csv_file)
        
        # 処理結果を記録するリスト
        processed_files = []
        
        for _, row in df.iterrows():
            # オリジナル音源の処理
            original_input = self.input_dir / row['original_file']
            original_output = self.output_dir / 'original' / f"{self.clean_filename(row['original_file'])}.wav"
            
            # リミックス音源の処理
            remix_input = self.input_dir / row['remix_file']
            remix_output = self.output_dir / 'remix' / f"{self.clean_filename(row['remix_file'])}.wav"
            
            # ファイルの処理
            original_success = self.process_audio_file(original_input, original_output)
            remix_success = self.process_audio_file(remix_input, remix_output, is_remix=True)
            
            if original_success and remix_success:
                processed_files.append({
                    'original_file': str(original_output),
                    'remix_file': str(remix_output),
                    'original_mel': str(self.output_dir / 'melspectrograms' / f"{original_output.stem}_mel.npy"),
                    'remix_mel': str(self.output_dir / 'melspectrograms' / f"{remix_output.stem}_mel.npy")
                })
        
        # 処理済みファイルの情報をCSVに保存
        processed_df = pd.DataFrame(processed_files)
        processed_df.to_csv(self.output_dir / 'processed_tracks.csv', index=False)
        
        return processed_df

def main():
    # 前処理の実行
    preprocessor = AudioPreprocessor(
        input_dir='/workspace',
        output_dir='/workspace/processed'
    )
    
    # データセットの処理
    processed_df = preprocessor.process_dataset('track_pairs.csv')
    
    print(f"処理が完了しました。{len(processed_df)}個のファイルが処理されました。")
    print(f"処理済みファイルの情報は 'processed_tracks.csv' に保存されました。")

if __name__ == "__main__":
    main() 