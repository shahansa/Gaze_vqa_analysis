#!/bin/bash

# setup data paths

mkdir data || echo 'data directory exists'
if ! curl -X GET https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip -o data/v2_questions_val.zip; then
  echo 'Error! unable to fetch v2_questions_val'
  exit
fi

if ! unzip data/v2_questions_val.zip -d data/; then
  echo ' Error! unzip'
  exit
fi

if ! curl -X GET http://images.cocodataset.org/zips/val2014.zip -o data/val2014.zip; then
  echo 'Error! unable to fetch v2_val_images'
  exit
fi

