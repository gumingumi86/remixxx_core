#!/usr/bin/env python3
import os
import time
import subprocess
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

def setup_driver():
    """Chromeドライバーのセットアップ"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=chrome_options)
    return driver

def get_track_urls(search_url):
    """検索結果からトラックのURLを取得"""
    driver = setup_driver()
    try:
        driver.get(search_url)
        # 検索結果が読み込まれるまで待機
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "a[href*='/tracks/']"))
        )
        
        # トラックのURLを取得
        track_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/tracks/']")
        urls = [link.get_attribute('href') for link in track_links]
        
        # 重複を除去
        return list(set(urls))
    finally:
        driver.quit()

def download_tracks(urls):
    """scdlを使用してトラックをダウンロード"""
    for url in urls:
        try:
            print(f"Downloading: {url}")
            subprocess.run(["scdl", "-l", url, "--add-description"], check=True)
            time.sleep(1)  # サーバーへの負荷を軽減
        except subprocess.CalledProcessError as e:
            print(f"Error downloading {url}: {e}")

def main():
    search_query = input("検索キーワードを入力してください: ")
    search_url = f"https://soundcloud.com/search?q={search_query.replace(' ', '%20')}"
    
    print("検索結果からURLを取得中...")
    urls = get_track_urls(search_url)
    
    if not urls:
        print("トラックが見つかりませんでした。")
        return
    
    print(f"{len(urls)}個のトラックが見つかりました。")
    download_tracks(urls)
    print("ダウンロードが完了しました。")

if __name__ == "__main__":
    main() 