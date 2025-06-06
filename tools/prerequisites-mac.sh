#!/bin/bash -

# Install project prerequisites

# Update your existing list of packages and install build tools
brew update && brew install \
  gcc \
  cmake \
  ninja \
  ccache \
  clang-format \
  python3 \
  boost \
  eigen \
  googletest \
  cpplint \
  cmake-lint \
  google-benchmark \
  llvm \
  libomp \
  openblas \
  spdlog \
  rapidjson \
  tree &&
  brew autoremove && brew cleanup
