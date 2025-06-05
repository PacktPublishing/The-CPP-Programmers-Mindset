# Dockerfile for installing Ubuntu 24.04
FROM ubuntu:24.04

LABEL maintainer="Alexei4Kondratiev@users.noreply.github.com"

ENV DEBIAN_FRONTEND="noninteractive"
ENV DEBCONF_NOWARNINGS="yes"

COPY ["./install-gtest.sh", "/root"]
COPY ["./prerequisites-ubuntu.sh", "/root"]

WORKDIR /root

RUN ./prerequisites-ubuntu.sh

RUN ldconfig
