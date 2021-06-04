FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# Timezone setting
RUN apt-get update && apt-get install -y --no-install-recommends tzdata

# Install something
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common
RUN apt-add-repository -y ppa:fish-shell/release-3
RUN apt-get update && apt-get install -y --no-install-recommends fish

RUN apt-get update && apt-get install -y --no-install-recommends nano git sudo curl

# OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends libopencv-dev

# Install Python library
COPY requirements.txt /
RUN pip install -r /requirements.txt

ARG UID
ARG USER
ARG PASSWORD
RUN groupadd -g 1000 ${USER}_group
RUN useradd -m --uid=${UID} --gid=${USER}_group --groups=sudo ${USER}
RUN echo ${USER}:${PASSWORD} | chpasswd
RUN echo 'root:root' | chpasswd
USER ${USER}

ENV PATH $PATH:/home/${USER}/.local/bin
