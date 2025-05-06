# 音声リミックス生成システム

このプロジェクトは、バイブコーディングにより作成されました。
SoundCloudから音声をダウンロードし、機械学習モデルを使用してリミックスを生成するシステムです。

## システム構成

システムは以下の4つのコンポーネントで構成されています：

1. **ダウンローダー** (`src/downloader`)
   - SoundCloudから音声をダウンロード
   - 入力：`track_pairs.csv`（SoundCloudのURLペアを含むCSVファイル）
   - 出力：`downloads/downloaded_tracks.csv`と音声ファイル

2. **プリプロセッサー** (`src/preprocessor`)
   - ダウンロードした音声の前処理
   - 入力：`downloads/downloaded_tracks.csv`
   - 出力：`processed/processed_tracks.csv`と処理済み音声ファイル

3. **トレーナー** (`src/trainer`)
   - リミックス生成モデルの学習
   - 入力：`processed/processed_tracks.csv`
   - 出力：`models/model_final.pth`（学習済みモデル）

4. **ジェネレーター** (`src/generator`)
   - 学習済みモデルを使用したリミックス生成
   - 入力：`models/model_final.pth`と`input.wav`
   - 出力：`generated/input_remix.wav`

## 必要条件

- Docker
- Docker Compose
- NVIDIA GPU（推奨）

## 使用方法

### 1. 環境の準備

```bash
# プロジェクトのクローン
git clone <repository-url>
cd <project-directory>

# 必要なディレクトリの作成
mkdir -p downloads processed models generated
```

### 2. 音声のダウンロード

```bash
# track_pairs.csvの準備
echo "id,original_url,remix_url" > track_pairs.csv
echo "1,https://soundcloud.com/example/original1,https://soundcloud.com/example/remix1" >> track_pairs.csv

# ダウンロードの実行
docker-compose run downloader
```

### 3. 音声の前処理

```bash
docker-compose run preprocessor
```

### 4. モデルの学習

```bash
docker-compose run trainer
```

### 5. リミックスの生成

```bash
# 入力音声ファイルを配置
cp your_input.wav input.wav

# リミックスの生成
docker-compose run generator
```

## 注意事項

- 各コンポーネントは独立して実行可能です
- 作業ディレクトリは `/workspace` にマウントされます
- コンテナ内での変更は、ホストマシンと同期されます
- GPUを使用する場合は、NVIDIA Container Toolkitがインストールされている必要があります

## トラブルシューティング

### 依存関係の問題

依存関係に問題がある場合は、各コンポーネントの`requirements.txt`を確認し、必要に応じて更新してください。

### メモリ不足

GPUのメモリが不足する場合は、以下の対策を試してください：
- バッチサイズを小さくする
- モデルのサイズを小さくする
- CPUモードで実行する

## ライセンス

このプロジェクトはMITライセンスの下で公開されています。 