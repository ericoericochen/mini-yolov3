#!/bin/bash

cd ..
mkdir ./data
cd ./data

mkdir oxford_iiit_pet

cd ./oxford_iiit_pet

wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/annotations.tar.gz

tar -xzvf images.tar.gz
tar -xzvf annotations.tar.gz
