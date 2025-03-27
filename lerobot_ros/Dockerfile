# ベースはNVIDIAのCUDA公式イメージ（Ubuntu 22.04 + CUDA 12.4）
FROM nvidia/cuda:12.4.0-base-ubuntu22.04

# 必要なツールのインストール
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    gnupg \
    lsb-release \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 3.10のインストール
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-venv python3.10-dev python3-pip && \
    rm -rf /var/lib/apt/lists/*

# デフォルトの python, pip を 3.10 にリンク
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1


# ROS One GPGキーとリポジトリ追加
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSL https://ros.packages.techfak.net/gpg.key -o /etc/apt/keyrings/ros-one-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros1.list && \
    echo "# deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net $(lsb_release -cs) main-dbg" >> /etc/apt/sources.list.d/ros1.list

# ROS One デスクトップ版と rosdep のインストール
RUN apt-get update && \
    apt-get install -y python3-rosdep && \
    rosdep init && \
    echo "yaml https://ros.packages.techfak.net/ros-one.yaml ubuntu" > /etc/ros/rosdep/sources.list.d/1-ros-one.list && \
    rosdep update && \
    apt-get install -y ros-one-desktop && \
    rm -rf /var/lib/apt/lists/*

# 作業ディレクトリ
WORKDIR /workspace

# CUDAの動作確認用（オプション）
RUN python -c "import sys; print(f'Python version: {sys.version}')"
