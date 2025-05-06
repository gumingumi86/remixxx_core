FROM ubuntu:22.04

# 必要なパッケージのインストール
RUN apt-get update && apt-get install -y \
    build-essential \
    python3 \
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

# SageMaker関連の環境変数
ENV SAGEMAKER_ROLE="arn:aws:iam::160045700050:role/service-role/AmazonSageMaker-ExecutionRole-20250417T193238"
ENV INPUT_DATA="s3://remixxx-resources/raw_data/"
ENV OUTPUT_DATA="s3://remixxx-resources/processed_data/"
ENV INSTANCE_TYPE="ml.m5.xlarge"

# 音声処理パラメータ
ENV TARGET_SR="22050"
ENV N_MELS="128"