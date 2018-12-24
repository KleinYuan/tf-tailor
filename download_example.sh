#!/usr/bin/env bash

echo "Downloading pre-trained models ..."
wget http://jaina.cs.ucdavis.edu/datasets/adv/imagenet/alexnet_frozen.pb -P $(pwd)/example/
