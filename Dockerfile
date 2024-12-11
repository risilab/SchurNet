# Start from Ubuntu 20.04
FROM nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Update the apt package list and upgrade packages
RUN apt-get update --fix-missing && apt-get upgrade -y && apt-get autoremove

# Install required packages
RUN echo "deb http://archive.ubuntu.com/ubuntu focal-updates main restricted universe multiverse" | tee -a /etc/apt/sources.list
RUN apt update -y

RUN apt-get install -y libgssapi-krb5-2 libssh-4 libcurl4 curl

# Download and install Miniforge
RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname -s)-$(uname -m).sh"
RUN bash Miniforge3-Linux-x86_64.sh -b 

ENV PATH="/root/miniforge3/bin:${PATH}"

# Install other important packages
RUN apt-get install -y g++ freeglut3-dev build-essential libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev

# Update and install software-properties-common
RUN apt-get update && apt-get install -y software-properties-common

# Add the PPA repository for graphics drivers and update
RUN add-apt-repository ppa:graphics-drivers/ppa --yes && apt-get update

# Install git
RUN apt-get install -y git

# Create Downloads directory and set as working directory
RUN mkdir /Downloads
WORKDIR /Downloads

# Copy requirements.txt into Downloads directory
COPY requirements.txt /Downloads

# Set CUDA_HOME environment variable
ENV CUDA_HOME=/usr/local/cuda-11.8

# Install ninja-build
RUN apt install -y ninja-build

# Install Python packages
RUN pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install torch_geometric


# Download and install cnine
RUN git clone https://github.com/richardxd/cnine.git /Downloads/cnine
WORKDIR /Downloads/cnine/python
RUN sed -i 's/compile_with_cuda=False/compile_with_cuda = True/' setup.py
RUN pip install -e .

# Download and install ptens
RUN git clone -b dev4 https://github.com/richardxd/ptens.git /Downloads/ptens
WORKDIR /Downloads/ptens/python

# Configure ptens
RUN sed -i 's/compile_with_cuda = False/compile_with_cuda = True/' setup.py

# Install ptens
RUN pip install -e .

RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Clean up APT when done
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Expose port 8001
EXPOSE 8001
