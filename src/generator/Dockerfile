FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-pip \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリの設定
WORKDIR /workspace

# 依存関係ファイルのコピーとインストール
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# シェルの設定
SHELL ["/bin/bash", "-c"] 