#!/bin/bash

if [ -e environ_secret.sh ]
then
    source utility/environ_secret.sh
else
    source utility/environ.sh
fi

if ! [ -x "$(command -v aicrowd-repo2docker)" ]; then
  echo 'Error: aicrowd-repo2docker is not installed.' >&2
  echo 'Please install it using requirements.txt or pip install -U aicrowd-repo2docker' >&2
  exit 1
fi

# Expected Env variables : in environ.sh

REPO2DOCKER="$(which aicrowd-repo2docker)"

sudo ${REPO2DOCKER} --no-run \
  --user-id 1001 \
  --user-name aicrowd \
  --image-name ${IMAGE_NAME}:${IMAGE_TAG} \
  --debug .
