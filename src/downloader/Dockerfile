FROM ubuntu:22.04

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    vim \
    python3 \
    python3-pip \
    ffmpeg \
    wget \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Chromeのインストール
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# 依存関係ファイルのコピーとインストール
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# シェルの設定
SHELL ["/bin/bash", "-c"]

# 共通の環境変数
ENV S3_BUCKET="remixxx-resources"
ENV S3_PREFIX="raw_data"
ENV AWS_PROFILE="default"

# 検索用の環境変数
ENV SEARCH_QUERY="anime lofi remix"
ENV MAX_TRACKS="5"
ENV OUTPUT_CSV="track_pairs.csv"

# ダウンロード用の環境変数
ENV DOWNLOAD_DIR=downloads