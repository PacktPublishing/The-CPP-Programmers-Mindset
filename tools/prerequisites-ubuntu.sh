#!/bin/bash -

# Install project prerequisites

## Update your existing list of packages and install build tools
TZ=Etc/UTC apt-get update && apt-get install -y \
  tzdata \
  build-essential \
  gdb \
  cmake \
  ninja-build \
  ccache \
  clang-format \
  clang-tidy \
  python3 \
  python3-pip \
  libboost-all-dev \
  libeigen3-dev \
  tree &&
  rm -rf /var/lib/apt/lists/*

# Using CCache to speed up compilation
export PATH=/usr/lib/ccache:$PATH

# Install libraries
./install-gtest.sh

# Configure Dynamic Linker Run Time Bindings
ldconfig
