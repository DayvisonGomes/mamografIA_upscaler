FROM nvcr.io/nvidia/pytorch:23.04-py3

COPY requirements.txt .

RUN pip3 install -r requirements.txt