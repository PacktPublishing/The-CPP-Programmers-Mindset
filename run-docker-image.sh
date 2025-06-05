#!/bin/bash -

# Run the Computational-Thinking-with-CPP docker image

echo
echo "Running docker image: mount $(pwd) directory as a root user volume"
docker run --rm -it --volume "$(pwd):/root" Computational-Thinking-with-CPP:latest
