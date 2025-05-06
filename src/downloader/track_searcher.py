#!/usr/bin/env python3
import os
import time
import subprocess
import urllib.parse
import re
import csv
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

class TrackSearcher:
    def __init__(self):
        self.driver = None
        
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
    
    def close(self):
        """ドライバーを閉じる"""
        if self.driver:
            self.driver.quit()
            self.driver = None

def get_environment_variables():
    """環境変数から設定を取得"""
    search_query = os.environ.get('SEARCH_QUERY')
    max_tracks = int(os.environ.get('MAX_TRACKS', '10'))
    download_dir = os.environ.get('DOWNLOAD_DIR', os.path.join('/workspace', 'downloads'))
    auto_download = os.environ.get('AUTO_DOWNLOAD', 'false').lower() == 'true'
    
    if not search_query:
        raise ValueError("SEARCH_QUERY environment variable is required")
    
    return {
        'search_query': search_query,
        'max_tracks': max_tracks,
        'download_dir': download_dir,
        'auto_download': auto_download
    }

def main():
    searcher = TrackSearcher()
    try:
        # 環境変数から設定を取得
        try:
            config = get_environment_variables()
            search_query = config['search_query']
            max_tracks = config['max_tracks']
            download_dir = config['download_dir']
            auto_download = config['auto_download']
        except ValueError as e:
            logger.error(f"環境変数の設定エラー: {e}")
            # 対話モードにフォールバック
            search_query = input("検索キーワードを入力してください: ")
            max_tracks = int(input("ダウンロードする最大曲数を入力してください（デフォルト: 10）: ") or "10")
            download_dir = os.path.join('/workspace', 'downloads')
            auto_download = False
        
        logger.info("検索結果からURLを取得中...")
        urls = searcher.search_soundcloud_tracks(search_query, max_tracks)
        
        if not urls:
            logger.error("トラックが見つかりませんでした。")
            return
        
        logger.info(f"{len(urls)}個のトラックが見つかりました。")
        
        # 自動ダウンロードが有効でない場合は確認
        if not auto_download:
            proceed = input("ダウンロードを開始しますか？ (y/n): ")
            if proceed.lower() != 'y':
                logger.info("ダウンロードをキャンセルしました。")
                return
        
        os.makedirs(download_dir, exist_ok=True)
        
        # リミックス音源のダウンロード
        downloaded_files = []
        for i, url in enumerate(urls, 1):
            try:
                logger.info(f"\n[{i}/{len(urls)}] ダウンロード中: {url}")
                subprocess.run(["scdl", "-l", url, "--path", download_dir, "--max-size", "10m"], check=True)
                time.sleep(1)
                
                files = [f for f in os.listdir(download_dir) if f.endswith('.mp3')]
                if files:
                    downloaded_files.append(max(files, key=lambda x: os.path.getctime(os.path.join(download_dir, x))))
            except subprocess.CalledProcessError as e:
                logger.error(f"ダウンロードエラー: {url} - {e}")
        
        logger.info("リミックス音源のダウンロードが完了しました。")
        
        # オリジナル曲の検索とダウンロード
        logger.info("オリジナル曲の検索を開始します...")
        for i, (filename, url) in enumerate(zip(downloaded_files, urls), 1):
            original_name = searcher.clean_filename(filename)
            if original_name:
                logger.info(f"\n[{i}/{len(downloaded_files)}] ファイル名: {filename}")
                logger.info(f"抽出した曲名: {original_name}")
                
                youtube_url = searcher.search_youtube_video(original_name)
                if youtube_url:
                    logger.info(f"オリジナル曲をダウンロード中: {youtube_url}")
                    try:
                        subprocess.run([
                            "yt-dlp",
                            "-x",
                            "--audio-format", "mp3",
                            "--audio-quality", "0",
                            "-o", f"{download_dir}/%(title)s.%(ext)s",
                            youtube_url
                        ], check=True)
                        
                        original_files = [f for f in os.listdir(download_dir) if f.endswith('.mp3')]
                        if original_files:
                            original_file = max(original_files, key=lambda x: os.path.getctime(os.path.join(download_dir, x)))
                            searcher.save_track_pairs(
                                os.path.join(download_dir, filename),
                                url,
                                original_name,
                                youtube_url,
                                os.path.join(download_dir, original_file)
                            )
                            logger.info("対応関係をCSVに保存しました。")
                    except subprocess.CalledProcessError as e:
                        logger.error(f"ダウンロードエラー: {e}")
                else:
                    logger.warning("オリジナル曲が見つかりませんでした。")
        
        logger.info("\nすべてのダウンロードが完了しました。")
        logger.info(f"ダウンロードされたファイルは以下のディレクトリに保存されています：")
        logger.info(f"保存先: {download_dir}")
            
    finally:
        searcher.close()

if __name__ == "__main__":
    main() 