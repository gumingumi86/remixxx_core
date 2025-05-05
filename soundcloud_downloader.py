#!/usr/bin/env python3
import os
import time
import subprocess
import urllib.parse
import re
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
from webdriver_manager.chrome import ChromeDriverManager

# 除外する単語リスト
EXCLUDE_WORDS = [
    'remix', 'bootleg', 'edit', 'mashup', 'mix', 'cover',
    'lofi', 'lo-fi', 'instrumental', 'karaoke', 'acoustic',
    'live', 'version', 'extended', 'radio', 'edit'
]

def setup_driver():
    """Chromeドライバーのセットアップ"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def clean_filename(filename):
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

def search_original_track(track_name):
    """オリジナル曲を検索"""
    driver = setup_driver()
    try:
        print(f"\nオリジナル曲を検索中: {track_name}")
        encoded_query = urllib.parse.quote(track_name)
        search_url = f"https://soundcloud.com/search/sounds?q={encoded_query}"
        driver.get(search_url)
        
        # 明示的な待機を設定
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_all_elements_located((
            By.CSS_SELECTOR,
            "a.sc-link-primary.soundTitle__title"
        )))
        
        # 最初の検索結果を取得
        elements = driver.find_elements(
            By.CSS_SELECTOR,
            "a.sc-link-primary.soundTitle__title"
        )
        
        if elements:
            return elements[0].get_attribute('href')
        return None
    except Exception as e:
        print(f"検索エラー: {e}")
        return None
    finally:
        driver.quit()

def get_track_urls(search_query, max_tracks=10):
    """検索結果からトラックのURLを取得"""
    driver = setup_driver()
    try:
        print("検索ページにアクセス中...")
        encoded_query = urllib.parse.quote(search_query)
        search_url = f"https://soundcloud.com/search/sounds?q={encoded_query}"
        driver.get(search_url)
        
        # 明示的な待機を設定
        wait = WebDriverWait(driver, 10)
        wait.until(EC.presence_of_all_elements_located((
            By.CSS_SELECTOR,
            "a.sc-link-primary.soundTitle__title"
        )))
        
        print("検索結果を解析中...")
        # トラックリンクを取得（stale対策付き）
        track_links = []
        retry_count = 3
        seen_urls = set()
        
        for _ in range(max_tracks * retry_count):
            try:
                elements = driver.find_elements(
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
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
        
        return track_links[:max_tracks]
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return []
    finally:
        driver.quit()

def download_tracks(urls):
    """scdlを使用してトラックをダウンロード"""
    downloaded_files = []
    for i, url in enumerate(urls, 1):
        try:
            print(f"\n[{i}/{len(urls)}] ダウンロード中: {url}")
            subprocess.run(["scdl", "-l", url, "--add-description"], check=True)
            time.sleep(1)  # サーバーへの負荷を軽減
            
            # 最新のダウンロードファイルを取得
            files = [f for f in os.listdir('/workspace') if f.endswith('.mp3')]
            if files:
                downloaded_files.append(max(files, key=lambda x: os.path.getctime(os.path.join('/workspace', x))))
        except subprocess.CalledProcessError as e:
            print(f"ダウンロードエラー: {url} - {e}")
    
    return downloaded_files

def main():
    search_query = input("検索キーワードを入力してください: ")
    max_tracks = int(input("ダウンロードする最大曲数を入力してください（デフォルト: 10）: ") or "10")
    
    print("\n検索結果からURLを取得中...")
    urls = get_track_urls(search_query, max_tracks)
    
    if not urls:
        print("トラックが見つかりませんでした。")
        return
    
    print(f"\n{len(urls)}個のトラックが見つかりました。")
    proceed = input("ダウンロードを開始しますか？ (y/n): ")
    
    if proceed.lower() == 'y':
        downloaded_files = download_tracks(urls)
        print("\nリミックス音源のダウンロードが完了しました。")
        
        # オリジナル曲の検索とダウンロード
        print("\nオリジナル曲の検索を開始します...")
        for filename in downloaded_files:
            original_name = clean_filename(filename)
            if original_name:
                print(f"\nファイル名: {filename}")
                print(f"抽出した曲名: {original_name}")
                
                original_url = search_original_track(original_name)
                if original_url:
                    print(f"オリジナル曲をダウンロード中: {original_url}")
                    try:
                        subprocess.run(["scdl", "-l", original_url, "--add-description"], check=True)
                        time.sleep(1)
                    except subprocess.CalledProcessError as e:
                        print(f"ダウンロードエラー: {e}")
                else:
                    print("オリジナル曲が見つかりませんでした。")
        
        print("\nすべてのダウンロードが完了しました。")
    else:
        print("ダウンロードをキャンセルしました。")

if __name__ == "__main__":
    main()