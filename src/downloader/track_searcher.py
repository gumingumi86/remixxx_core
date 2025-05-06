#!/usr/bin/env python3
import os
import time
import subprocess
import urllib.parse
import re
import csv
import boto3
from datetime import datetime
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 除外する単語リスト
EXCLUDE_WORDS = [
    'remix', 'bootleg', 'edit', 'mashup', 'mix', 'cover',
    'lofi', 'lo-fi', 'instrumental', 'karaoke', 'acoustic',
    'live', 'version', 'extended', 'radio', 'edit'
]

class TrackManager:
    def __init__(self, s3_bucket=None, s3_prefix='raw_data'):
        self.driver = None
        self.s3_bucket = s3_bucket
        self.s3_prefix = s3_prefix
        
        # ECRの実行時以外の場合はローカルの認証情報を使用
        if s3_bucket:
            if not os.environ.get('AWS_CONTAINER_CREDENTIALS_RELATIVE_URI'):
                session = boto3.Session(profile_name='default')
                self.s3_client = session.client('s3')
            else:
                self.s3_client = boto3.client('s3')
        
    def setup_driver(self):
        """Chromeドライバーのセットアップ"""
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        
    def clean_filename(self, filename):
        """ファイル名から除外ワードを削除し、オリジナル曲名を抽出"""
        # 拡張子を除去
        name = os.path.splitext(filename)[0]
        
        # 括弧内の文字を削除
        name = re.sub(r'\([^)]*\)', '', name)
        name = re.sub(r'\[[^\]]*\]', '', name)
        
        # 除外ワードを削除
        for word in EXCLUDE_WORDS:
            name = re.sub(rf'\b{word}\b', '', name, flags=re.IGNORECASE)
        
        # 余分な空白を削除
        name = ' '.join(name.split())
        
        return name.strip()
    
    def download_soundcloud_track(self, url: str, output_dir: str) -> str:
        """SoundCloudからトラックをダウンロード"""
        try:
            logger.info(f"ダウンロード中: {url}")
            subprocess.run([
                "scdl",
                "-l", url,
                "--path", output_dir,
                "--max-size", "10m",
                "-c",
                "--overwrite"
            ], check=True)
            
            # ダウンロードされたファイルを確認
            files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
                logger.info(f"ダウンロード成功: {latest_file}")
                return latest_file
            else:
                logger.warning(f"ファイルが見つかりません: {url}")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"ダウンロードエラー: {url} - {e}")
            return None
    
    def download_youtube_track(self, url: str, output_dir: str) -> str:
        """YouTubeからトラックをダウンロード"""
        try:
            logger.info(f"ダウンロード中: {url}")
            subprocess.run([
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "--audio-quality", "0",
                "-o", f"{output_dir}/%(title)s.%(ext)s",
                url
            ], check=True)
            
            # ダウンロードされたファイルを確認
            files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]
            if files:
                latest_file = max(files, key=lambda x: os.path.getctime(os.path.join(output_dir, x)))
                logger.info(f"ダウンロード成功: {latest_file}")
                return latest_file
            else:
                logger.warning(f"ファイルが見つかりません: {url}")
                return None
                
        except subprocess.CalledProcessError as e:
            logger.error(f"ダウンロードエラー: {url} - {e}")
            return None
    
    def upload_to_s3(self, local_dir: str):
        """ダウンロードしたファイルをS3にアップロード"""
        if not self.s3_bucket:
            logger.warning("S3バケットが設定されていないため、アップロードをスキップします")
            return
            
        for file in os.listdir(local_dir):
            if file.endswith(('.mp3', '.wav', '.csv')):
                local_path = os.path.join(local_dir, file)
                s3_key = f"{self.s3_prefix}/{file}"
                
                try:
                    self.s3_client.upload_file(local_path, self.s3_bucket, s3_key)
                    logger.info(f"アップロード成功: {file}")
                except Exception as e:
                    logger.error(f"アップロードエラー: {file} - {str(e)}")
    
    def search_soundcloud_tracks(self, search_query, max_tracks=10):
        """SoundCloudでリミックス音源を検索"""
        if not self.driver:
            self.setup_driver()
            
        try:
            logger.info("SoundCloudの検索ページにアクセス中...")
            encoded_query = urllib.parse.quote(search_query)
            search_url = f"https://soundcloud.com/search/sounds?q={encoded_query}"
            self.driver.get(search_url)
            
            # 明示的な待機を設定
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_all_elements_located((
                By.CSS_SELECTOR,
                "a.sc-link-primary.soundTitle__title"
            )))
            
            logger.info("検索結果を解析中...")
            track_links = []
            seen_urls = set()
            
            while len(track_links) < max_tracks:
                try:
                    elements = self.driver.find_elements(
                        By.CSS_SELECTOR,
                        "a.sc-link-primary.soundTitle__title"
                    )
                    for element in elements:
                        url = element.get_attribute('href')
                        if url and url not in seen_urls:
                            track_links.append(url)
                            seen_urls.add(url)
                        if len(track_links) >= max_tracks:
                            break
                    if len(track_links) >= max_tracks:
                        break
                except StaleElementReferenceException:
                    time.sleep(1)
                    continue
                
                # スクロールしてより多くの結果を読み込む
                self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(2)
            
            return track_links[:max_tracks]
            
        except Exception as e:
            logger.error(f"SoundCloud検索エラー: {e}")
            return []
    
    def search_youtube_video(self, track_name):
        """YouTubeでオリジナル曲を検索"""
        if not self.driver:
            self.setup_driver()
            
        try:
            logger.info(f"YouTubeで検索中: {track_name}")
            encoded_query = urllib.parse.quote(f"{track_name} audio")
            search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
            self.driver.get(search_url)
            
            # 明示的な待機を設定
            wait = WebDriverWait(self.driver, 10)
            wait.until(EC.presence_of_all_elements_located((
                By.CSS_SELECTOR,
                "a#video-title"
            )))
            
            # 最初の検索結果を取得
            elements = self.driver.find_elements(By.CSS_SELECTOR, "a#video-title")
            
            if elements:
                return elements[0].get_attribute('href')
            return None
            
        except Exception as e:
            logger.error(f"YouTube検索エラー: {e}")
            return None
    
    def save_track_pairs(self, remix_file, remix_url, original_name, youtube_url, original_file):
        """リミックスとオリジナルの対応関係をCSVに保存"""
        csv_file = 'track_pairs.csv'
        file_exists = os.path.exists(csv_file)
        
        with open(csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['timestamp', 'remix_file', 'remix_url', 'original_name', 'youtube_url', 'original_file'])
            
            writer.writerow([
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                remix_file,
                remix_url,
                original_name,
                youtube_url,
                original_file
            ])
    
    def process_tracks(self, search_query, max_tracks=5, download_dir='downloads'):
        """トラックの検索、ダウンロード、S3アップロードを一括で実行"""
        try:
            os.makedirs(download_dir, exist_ok=True)
            
            # SoundCloudでリミックス音源を検索
            logger.info("検索結果からURLを取得中...")
            urls = self.search_soundcloud_tracks(search_query, max_tracks)
            
            if not urls:
                logger.error("トラックが見つかりませんでした。")
                return
            
            logger.info(f"{len(urls)}個のトラックが見つかりました。")
            
            # リミックス音源のダウンロード
            downloaded_files = []
            for i, url in enumerate(urls, 1):
                logger.info(f"\n[{i}/{len(urls)}] 処理中...")
                remix_file = self.download_soundcloud_track(url, download_dir)
                if remix_file:
                    downloaded_files.append((remix_file, url))
                    time.sleep(1)
            
            # オリジナル曲の検索とダウンロード
            logger.info("オリジナル曲の検索を開始します...")
            for i, (filename, url) in enumerate(downloaded_files, 1):
                original_name = self.clean_filename(filename)
                if original_name:
                    logger.info(f"\n[{i}/{len(downloaded_files)}] ファイル名: {filename}")
                    logger.info(f"抽出した曲名: {original_name}")
                    
                    youtube_url = self.search_youtube_video(original_name)
                    if youtube_url:
                        original_file = self.download_youtube_track(youtube_url, download_dir)
                        if original_file:
                            self.save_track_pairs(
                                os.path.join(download_dir, filename),
                                url,
                                original_name,
                                youtube_url,
                                os.path.join(download_dir, original_file)
                            )
                            logger.info("対応関係をCSVに保存しました。")
            
            # S3へのアップロード
            if self.s3_bucket:
                self.upload_to_s3(download_dir)
            
            logger.info("\nすべての処理が完了しました。")
            logger.info(f"ファイルは以下のディレクトリに保存されています：")
            logger.info(f"保存先: {download_dir}")
            
        finally:
            self.close()
    
    def close(self):
        """ドライバーを閉じる"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def main():
    # 環境変数から設定を取得
    search_query = os.environ.get('SEARCH_QUERY')
    max_tracks = int(os.environ.get('MAX_TRACKS', '5'))
    download_dir = os.environ.get('DOWNLOAD_DIR', 'downloads')
    s3_bucket = os.environ.get('S3_BUCKET')
    s3_prefix = os.environ.get('S3_PREFIX', 'raw_data')
    
    if not search_query:
        raise ValueError("SEARCH_QUERY environment variable is required")
    
    # TrackManagerの初期化と実行
    manager = TrackManager(s3_bucket, s3_prefix)
    manager.process_tracks(search_query, max_tracks, download_dir)

if __name__ == "__main__":
    main() 