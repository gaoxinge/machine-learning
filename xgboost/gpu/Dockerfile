FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-setuptools
RUN easy_install3 -U pip

RUN pip3 install sklearn
RUN pip3 install xgboost

COPY main.py main.py

CMD ["sleep", "infinity"]