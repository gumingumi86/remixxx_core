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
ENV TRAINING_DATA="s3://remixxx-resources/processed_data/"
ENV MODEL_OUTPUT_PATH="s3://remixxx-resources/models/"
ENV DEPLOY_MODEL="false"
ENV TRAINING_INSTANCE_TYPE="ml.p3.2xlarge"

# ハイパーパラメータ（オプション）
ENV EPOCHS="10"
ENV BATCH_SIZE="32"
ENV LEARNING_RATE="0.001"