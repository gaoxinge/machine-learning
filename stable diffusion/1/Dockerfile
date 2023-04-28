FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

ENV IGNORE_CMD_ARGS_ERRORS=1 

RUN apt update
RUN apt install -y python3-dev
RUN apt install -y python3-setuptools
RUN apt install -y python3-pip
RUN apt install -y python3-venv
RUN apt install -y git
RUN apt install -y vim
RUN apt install -y wget
RUN DEBIAN_FRONTEND=noninteractive TZ=Asia/Shanghai apt-get install -y ffmpeg libsm6 libxext6

WORKDIR /home/root
COPY webui.sh .
COPY launch.py .

RUN bash webui.sh --prepare

CMD ["sleep", "infinity"]
