# Setup cuda, cupy-cuda9.2, chainer and workspace on ROS melodic
# Original:
# https://github.com/sktometometo/dockerfiles/blob/master/melodic_cuda9.2/Dockerfile

FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04

ENV DEBIAN_FRONTEND=noninteractive

Maintainer Naoya Yamaguchi <708yamaguchi@gmail.com>

# Ubuntu 18.04 Image with
#  - ROS melodic
#  - cuda 9.2
#  - cudnn7
#  - chainer 6.4.0

# -------------------------------------------------------
# ROS Installation
# -------------------------------------------------------
# install packages
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2  \
    && rm -rf /var/lib/apt/lists/*
# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu bionic main" > /etc/apt/sources.list.d/ros1-latest.list
# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools \
    python-catkin-tools \
    python-wstools \
    && rm -rf /var/lib/apt/lists/*
# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8
# bootstrap rosdep
RUN rosdep init && rosdep update
# install ros packages
ENV ROS_DISTRO melodic
RUN apt-get update && apt-get install -y \
    ros-melodic-desktop-full=1.4.1-0* \
    && rm -rf /var/lib/apt/lists/*


# -------------------------------------------------------
# Basic Configuration
# -------------------------------------------------------

RUN apt-get update \
    && apt-get dist-upgrade -q -y \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update -q && \
    apt-get install -q -y \
        acpi \
        apt \
        apt-cacher-ng \
        apt-utils \
        aptitude \
        bc \
        ccache \
        cmake \
        cmake-curses-gui \
        cron \
        curl \
        emacs \
        git \
        hddtemp \
        htop \
        ibus-mozc \
        indicator-multiload \
        ipython \
        less \
        libboost-all-dev \
        libeigen3-dev \
        libflann-dev \
        libfreetype6-dev \
        libgnome2-bin \
        libnotify-bin \
        libqhull-dev \
        libvtk6-dev \
        lm-sensors \
        lsb-release \
        man-db \
        mesa-utils \
        mlocate \
        nkf \
        notify-osd \
        ntp \
        patch \
        pkg-config \
        python \
        python-dev \
        python-pip \
        python-setuptools \
        python-vtk6 \
        rlwrap \
        software-properties-common \
        ssh \
        sudo \
        tcl-vtk6 \
        wget \
        xdg-user-dirs-gtk \
        xsel \
        vim \
        emacs \
        tmux \
        git \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*


# -------------------------------------------------------
# Chainer Installation
# -------------------------------------------------------

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
        python3-dev \
        python3-pip \
        python3-wheel \
        python3-setuptools \
        git \
        cmake \
        libblas3 \
        libblas-dev \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip install -U \
        "setuptools<45" \
        "pip==9.0.3"

RUN pip install -U \
        "chainer==6.4.0" \
        "cupy-cuda92==6.4.0" \
        "fcn==6.4.17" \
        "ipython<6" \
        "matplotlib<3" \
        numpy \
        percol \
        flake8 \
        grip \
        hacking


# -------------------------------------------------------
#
# -------------------------------------------------------
RUN echo '' >> ~/.bashrc
RUN echo '# CUDA' >> ~/.bashrc
RUN echo 'export PATH=/usr/local/cuda/bin:${PATH}' >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}' >> ~/.bashrc
RUN echo '' >> ~/.bashrc
RUN echo '# ROS' >> ~/.bashrc
RUN echo 'source /opt/ros/melodic/setup.bash' >> ~/.bashrc
RUN echo 'source $HOME/catkin_ws/devel/setup.bash' >> ~/.bashrc

# Create ROS workspace
RUN mkdir $HOME/catkin_ws/src -p && \
    cd $HOME/catkin_ws && \
    catkin init && \
    cd $HOME/catkin_ws/src && \
    git clone https://github.com/jsk-ros-pkg/jsk_recognition.git && \
    git clone https://github.com/708yamaguchi/decopin_hand.git && \
    apt-get update && apt-get upgrade && \
    rosdep install --from-paths . --ignore-src -y -r && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/* && \
    pip install imgaug --no-deps # cannot be installed by rosdep
RUN ["/bin/bash", "-c", "source /opt/ros/melodic/setup.bash && cd $HOME/catkin_ws && catkin build decopin_hand"]

CMD /bin/bash -c "cd $HOME && bash --login -c 'exec bash'"
