FROM ros:noetic-ros-base-focal
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && apt-get install -y \
    python3-pip \
    libx11-6 \
    ros-noetic-cv-bridge && \
    echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc && \
    rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy>=1.19.5,<1.27.0" PyYAML opencv-contrib-python scipy pymap3d
