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

# SageMaker関連の環境変数
ENV SAGEMAKER_ROLE="arn:aws:iam::160045700050:role/service-role/AmazonSageMaker-ExecutionRole-20250417T193238"
ENV MODEL_NAME="remix-generator-model"
ENV INPUT_DATA="s3://remixxx-resources/input/"
ENV OUTPUT_DATA="s3://remixxx-resources/output/"

# オプションの環境変数
ENV BATCH_TRANSFORM="true"
ENV DEPLOY_ENDPOINT="false"

# インスタンス設定（オプション）
ENV INSTANCE_TYPE="ml.m5.xlarge"
ENV INSTANCE_COUNT="1"