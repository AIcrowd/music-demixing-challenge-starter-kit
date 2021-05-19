#!/bin/bash
tagname=$1

git tag -am "submission-"$tagname submission-$tagname
git push aicrowd submission-$tagname
