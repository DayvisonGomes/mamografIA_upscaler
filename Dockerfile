FROM nvcr.io/nvidia/pytorch:23.04-py3

COPY requirements.txt .
RUN pip install diffusers transformers accelerate scipy safetensors typing

RUN pip install -r requirements.txt