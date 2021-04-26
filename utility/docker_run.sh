#!/bin/bash
# This script run your submission inside a docker image, this is identical in termrs of 
# how your code will be executed on AIcrowd platform

if [ -e environ_secret.sh ]
then
    echo "Note: Gathering environment variables from environ_secret.sh"
    source utility/environ_secret.sh
else
    echo "Note: Gathering environment variables from environ.sh"
    source utility/environ.sh
fi

# Skip building docker image on run, by default each run means new docker image build
if [[ " $@ " =~ " --no-build " ]]; then
    echo "Skipping docker image build"
else
    echo "Building docker image, for skipping docker image build use \"--no-build\""
    ./utility/docker_build.sh
fi

# Expected Env variables : in environ.sh
sudo docker run \
--net=host \
--user 0 \
-e AICROWD_IS_GRADING=True \
-e AICROWD_DEBUG_MODE=True \
-it ${IMAGE_NAME}:${IMAGE_TAG} \
/bin/bash