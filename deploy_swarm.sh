#!/bin/bash

# NOT WORKING YET
# Allows user to supply a list of tar files or image names to load/run
# before initializing a Docker swarm of images. Lots of functionality is
# missing at this point while I learn about how Docker Swarm works. This
# is a very basic skeleton of what the finished script should look like.
# May be better to implement in python.
# Important note: Docker swarm uses services rather than images. A service
# is basically one instance of an image that can be swapped out to help
# with failures and load-balancing.

# check if docker swarm is already initialized
docker info | grep Swarm: | grep inactive && docker swarm init

# iterate through arguments given
for image_file in "$@"
do
  # load image from the .tar file
  docker load -i $image_file
  
  # get the image name and tag from the loaded images
  image_name=$(docker images --format "{{.Repository}}:{{.Tag}}" | head -n 1)
  
  # create a service from the image using image name as service name
  service_name=$(echo $image_name | tr ':' '_')
  docker service create --name $service_name $image_name
done

