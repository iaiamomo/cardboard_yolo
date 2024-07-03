FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

RUN apt-get update &&\
    apt-get install -y \
        python3 \
        python3-pip \
        git \
        cmake \
        lsof \
        sudo \
        less \
        nano \
        vim \
        wget \
        ffmpeg \
        libsm6 \
        libxext6 \
        libjpeg-dev \
        zlib1g-dev \
        libpython3-dev \
        libopenblas-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev &&\
    apt-get clean &&\
    rm -rf /var/cache

# This adds the 'default' user to sudoers with full privileges:
RUN HOME=/home/default &&\
    mkdir -p ${HOME} &&\
    GROUP_ID=1000 &&\
    USER_ID=1000 &&\
    groupadd -r default -f -g "$GROUP_ID" &&\
    useradd -u "$USER_ID" -r -g default -d "$HOME" -s /sbin/nologin \
        -c " Default Application User" default &&\
    chown -R "$USER_ID:$GROUP_ID" ${HOME} &&\
    usermod -a -G sudo default &&\
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CCACHE_DIR=/build/docker_ccache
ENV LD_LIBRARY_PATH=/usr/local/lib

WORKDIR /home/default

# install requirements
RUN pip install python-dotenv utils jupyter clearml

# clone yolov5 and install requirements
RUN git clone https://github.com/ultralytics/yolov5.git &&\
    cd yolov5 &&\
    pip install -qr requirements.txt &&\
    git config --global --add safe.directory /home/default/yolov5

WORKDIR /home/default