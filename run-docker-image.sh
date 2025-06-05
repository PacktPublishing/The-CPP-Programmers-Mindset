#!/bin/bash -

# Run the computational-thinking-with-cpp docker image

echo
echo "Running docker image: mount $(pwd) directory as a root user volume"
docker run --rm -it --volume "$(pwd):/root" computational-thinking-with-cpp:latest
