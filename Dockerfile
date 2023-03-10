FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

ADD requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

WORKDIR /root

ADD tsp /root/tsp
ADD launch /root/launch
ADD setup.py /root/setup.py

RUN pip install -e .
