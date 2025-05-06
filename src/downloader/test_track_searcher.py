#!/usr/bin/env python3
import os
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil
from track_searcher import TrackManager, EXCLUDE_WORDS

class TestTrackManager(unittest.TestCase):
    def setUp(self):
        """テストの前準備"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = TrackManager()
        
    def tearDown(self):
        """テストの後処理"""
        shutil.rmtree(self.temp_dir)
        if self.manager.driver:
            self.manager.close()
    
    def test_clean_filename(self):
        """ファイル名のクリーニング機能のテスト"""
        test_cases = [
            ("test (remix).mp3", "test"),
            ("original [lofi edit].mp3", "original"),
            ("song (bootleg) [extended mix].mp3", "song"),
            ("normal song.mp3", "normal song"),
            ("multiple (remix) [edit] (bootleg).mp3", "multiple"),
        ]
        
        for input_name, expected in test_cases:
            with self.subTest(input_name=input_name):
                result = self.manager.clean_filename(input_name)
                self.assertEqual(result, expected)
    
    @patch('subprocess.run')
    def test_download_soundcloud_track(self, mock_run):
        """SoundCloudダウンロード機能のテスト"""
        # モックの設定
        mock_run.return_value = MagicMock(returncode=0)
        
        # テスト用のダミーファイルを作成
        test_file = os.path.join(self.temp_dir, "test.mp3")
        with open(test_file, "w") as f:
            f.write("dummy content")
        
        # テスト実行
        result = self.manager.download_soundcloud_track("https://soundcloud.com/test", self.temp_dir)
        
        # 検証
        self.assertIsNotNone(result)
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_download_youtube_track(self, mock_run):
        """YouTubeダウンロード機能のテスト"""
        # モックの設定
        mock_run.return_value = MagicMock(returncode=0)
        
        # テスト用のダミーファイルを作成
        test_file = os.path.join(self.temp_dir, "test.mp3")
        with open(test_file, "w") as f:
            f.write("dummy content")
        
        # テスト実行
        result = self.manager.download_youtube_track("https://youtube.com/test", self.temp_dir)
        
        # 検証
        self.assertIsNotNone(result)
        mock_run.assert_called_once()
    
    @patch('boto3.Session')
    @patch('boto3.client')
    def test_upload_to_s3(self, mock_boto3, mock_session):
        """S3アップロード機能のテスト"""
        # モックの設定
        mock_s3 = MagicMock()
        mock_boto3.return_value = mock_s3
        mock_session.return_value.client.return_value = mock_s3
        
        # テスト用のダミーファイルを作成
        test_file = os.path.join(self.temp_dir, "test.mp3")
        with open(test_file, "w") as f:
            f.write("dummy content")
        
        # テスト実行
        manager = TrackManager(s3_bucket="test-bucket", s3_prefix="test-prefix")
        manager.s3_client = mock_s3
        manager.upload_to_s3(self.temp_dir)
        
        # 検証
        mock_s3.upload_file.assert_called_once()
    
    @patch('selenium.webdriver.Chrome')
    def test_search_soundcloud_tracks(self, mock_driver):
        """SoundCloud検索機能のテスト"""
        # モックの設定
        mock_element = MagicMock()
        mock_element.get_attribute.return_value = "https://soundcloud.com/test"
        mock_driver.return_value.find_elements.return_value = [mock_element]
        
        # テスト実行
        with patch.object(self.manager, 'setup_driver') as mock_setup:
            self.manager.driver = mock_driver.return_value
            results = self.manager.search_soundcloud_tracks("test query", 1)
        
        # 検証
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0], "https://soundcloud.com/test")
    
    @patch('selenium.webdriver.Chrome')
    def test_search_youtube_video(self, mock_driver):
        """YouTube検索機能のテスト"""
        # モックの設定
        mock_element = MagicMock()
        mock_element.get_attribute.return_value = "https://youtube.com/test"
        mock_driver.return_value.find_elements.return_value = [mock_element]
        
        # テスト実行
        with patch.object(self.manager, 'setup_driver') as mock_setup:
            self.manager.driver = mock_driver.return_value
            result = self.manager.search_youtube_video("test query")
        
        # 検証
        self.assertEqual(result, "https://youtube.com/test")
    
    def test_save_track_pairs(self):
        """トラックペアの保存機能のテスト"""
        # テストデータ
        remix_file = "remix.mp3"
        remix_url = "https://soundcloud.com/remix"
        original_name = "Original Song"
        youtube_url = "https://youtube.com/original"
        original_file = "original.mp3"
        
        # テスト実行
        csv_path = os.path.join(self.temp_dir, "track_pairs.csv")
        self.manager.save_track_pairs(
            remix_file,
            remix_url,
            original_name,
            youtube_url,
            original_file
        )
        
        # 検証
        self.assertTrue(os.path.exists("track_pairs.csv"))
        with open("track_pairs.csv", "r", encoding="utf-8") as f:
            content = f.read()
            self.assertIn(remix_url, content)
            self.assertIn(youtube_url, content)
            self.assertIn(original_name, content)

if __name__ == '__main__':
    unittest.main() 