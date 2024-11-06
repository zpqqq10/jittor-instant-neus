# from jnerf
# FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y upgrade \
#     && apt-get -y install --no-install-recommends libgl1 tcl-dev tk-dev python3 python-is-python3 python3-pip python3-tk python3-dev git

# COPY ./requirements.txt ./

# RUN python3 -m pip install -U pip \
# 	&& pip3 --no-cache-dir install -r ./requirements.txt \
# 	&& rm ./requirements.txt

# for the final round
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    python3-wheel \
    python3-setuptools && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip3 install cupy-cuda12x

COPY . /workspace
WORKDIR /workspace

RUN pip install opencv-python-headless
RUN python -m pip install --user -e .

SHELL ["/bin/bash", "-c"]

CMD ["/bin/bash"] 