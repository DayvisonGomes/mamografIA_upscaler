# A simple script to build the Docker image.

#TAG=mamografia_upscaler
TAG=tcc
USER=Dayvison

#docker build -t "${USER}:${TAG}" -f Dockerfile .
docker build -t "${TAG}" -f Dockerfile .