#!/bin/bash -

# Install Google Test Framework

# Update your existing list of packages and install build tools
apt-get update && apt-get install -y \
  build-essential \
  cmake \
  libgtest-dev &&
  rm -rf /var/lib/apt/lists/*

cd /usr/src/googletest || exit

cmake --fresh -B ./build
cmake --build ./build
cmake --install ./build
cmake --build ./build --target clean

# Configure Dynamic Linker Run Time Bindings
ldconfig
