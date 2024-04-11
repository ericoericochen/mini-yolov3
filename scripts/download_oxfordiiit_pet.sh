#!/bin/bash

cd ..
mkdir ./data
cd ./data

wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz

tar -xzvf images.tar.gz
tar -xzvf annotations.tar.gz
