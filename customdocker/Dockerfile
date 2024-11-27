# parameters
ARG REPO_NAME="PIDPlus"
ARG DESCRIPTION="PID project duckietown 2024"
ARG MAINTAINER="Simon-Olivier Duguay (simonkirbi@hotmail.com)"
# pick an icon from: https://fontawesome.com/v4.7.0/icons/
ARG ICON="cube"

# ==================================================>
# ==> Do not change the code below this line
ARG ARCH=amd64
ARG DISTRO=daffy
ARG DOCKER_REGISTRY=docker.io
#ARG BASE_IMAGE=dt-ros-commons
ARG BASE_IMAGE=dt-machine-learning-base-pytorch
ARG BASE_TAG=${DISTRO}-${ARCH}
ARG LAUNCHER=default

# define base image
FROM ${DOCKER_REGISTRY}/duckietown/${BASE_IMAGE}:${BASE_TAG} as base

# install python3 dependencies
ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}

RUN pip3 install numpy pillow torchvision