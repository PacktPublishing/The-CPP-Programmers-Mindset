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
  git \
  cpplint \
  cmake-lint \
  tree &&
  brew autoremove && brew cleanup
