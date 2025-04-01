# FROM nvidia/cuda:12.4.0-base-ubuntu22.04
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 基本パッケージとPython 3.10
RUN apt-get update && apt-get install -y \
    software-properties-common \
    wget \
    curl \
    gnupg \
    lsb-release \
    build-essential \
    sudo \
    git \
    tmux \
    emacs \
    mlocate \
    python3.10 \
    python3.10-venv \
    python3.10-dev \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Pythonのデフォルト設定
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1


# RUN apt-get update && apt-get install -y gnupg2 curl && \
#     curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub | apt-key add - && \
#     echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64 /" > /etc/apt/sources.list.d/cuda.list
# RUN apt-get update && apt-get install -y \
#     libcudnn9=9.1.0.* \
#     libcudnn9-dev=9.1.0.* \
#     && rm -rf /var/lib/apt/lists/*

# # NVIDIA CUDA リポジトリはキー登録済みなので、これだけでOK
# RUN echo "deb [signed-by=/usr/share/keyrings/cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" \
#     > /etc/apt/sources.list.d/cuda.list

# # cuDNN インストール
# RUN apt-get update && \
#     apt-get install -y libcudnn9 libcudnn9-dev && \
#     rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# ROS One GPGキーとリポジトリ追加
RUN mkdir -p /etc/apt/keyrings && \
    curl -sSL https://ros.packages.techfak.net/gpg.key -o /etc/apt/keyrings/ros-one-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros1.list && \
    echo "# deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/ros-one-keyring.gpg] https://ros.packages.techfak.net $(lsb_release -cs) main-dbg" >> /etc/apt/sources.list.d/ros1.list

# ROS Oneとrosdep、catkin-toolsなど必要パッケージのインストール
RUN apt-get update && apt-get install -y \
    ros-one-desktop \
    python3-rosdep \
    python3-catkin-tools \
    && rosdep init || true

# rosdep 定義
RUN mkdir -p /etc/ros/rosdep/sources.list.d && \
    echo "yaml https://ros.packages.techfak.net/ros-one.yaml ubuntu" > /etc/ros/rosdep/sources.list.d/1-ros-one.list && \
    rosdep update

# catkin ワークスペース作成 & リポジトリクローン
RUN mkdir -p /root/catkin_ws/src && \
    cd /root/catkin_ws/src && \
    git clone -b add-diffusion-policy https://github.com/Kanazawanaoaki/eus_imitation.git

# 依存ライブラリ（pythonパッケージ）
RUN cp /root/catkin_ws/src/eus_imitation/lerobot_ros/requirements.txt /tmp/requirements.txt
# RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt
RUN pip install --upgrade pip && pip install --no-deps -r /tmp/requirements.txt && pip install sympy --ignore-installed
RUN pip install "numpy<2" \
    datasets

# # 依存ライブラリ（aptパッケージ）
# RUN apt-get update && apt-get install -y \
#     ros-one-cv-bridge \
#     ros-one-rospy \
#     ros-one-rosbag \
#     ros-one-message-filters \
#     ros-one-sensor-msgs \
#     python3-numpy \
#     python3-opencv \
#     && rm -rf /var/lib/apt/lists/*

# 依存解決と catkin build 実行
RUN cd /root/catkin_ws && \
    rosdep install --from-paths src --ignore-src -r -y && \
    catkin config --extend /opt/ros/one && \
    catkin build

# ROS 環境を常に読み込む設定
RUN echo "source /opt/ros/one/setup.bash" >> /root/.bashrc && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

## lerobotのinstall
RUN apt-get update && apt-get install -y \
    libavformat-dev \
    libavcodec-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /root/workspace && \
    cd /root/workspace && \
    git clone https://github.com/huggingface/lerobot.git && \
    cd lerobot && git checkout 538455a965ce25a6f7b81d8b5c8cee46b90a7df9  && pip install --no-binary=av -e . --ignore-installed

RUN cd /root/workspace && \
    git clone https://github.com/ojh6404/imitator.git && \
    cd imitator && pip install -e .

RUN pip install ipdb

# 作業ディレクトリ
WORKDIR /root/catkin_ws
