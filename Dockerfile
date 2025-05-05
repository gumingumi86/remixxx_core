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
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Chromeのインストール
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add - \
    && echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google.list \
    && apt-get update \
    && apt-get install -y google-chrome-stable \
    && rm -rf /var/lib/apt/lists/*

# scdlのインストール
RUN pip3 install scdl

# Pythonパッケージのインストール
RUN pip3 install requests beautifulsoup4 selenium webdriver_manager yt-dlp \
    numpy pandas librosa soundfile torch torchaudio \
    scikit-learn matplotlib seaborn

# 作業ディレクトリの設定
WORKDIR /workspace

# シェルの設定
SHELL ["/bin/bash", "-c"] 