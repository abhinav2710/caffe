#!/usr/bin/env sh
# Compute the mean image from the imagenet training lmdb
# N.B. this is available in data/ilsvrc12

EXAMPLE=/media/data/abhinav/ann_project_data
DATA=/media/data/abhinav/ann_project_data
TOOLS=build/tools

$TOOLS/compute_image_mean $EXAMPLE/ilsvrc12_train_edge_resized_lmdb \
  $DATA/imagenet_edge_resized_mean.binaryproto

echo "Done."
