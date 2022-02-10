#!/bin/bash

if [ $# -eq 0 ]
then
  echo "usege: $0 exp_xxx"
else
  kaggle datasets init -p "weights/$1"
  project_name=$(basename `pwd`)
  sed -i -e "s/INSERT_TITLE_HERE/${project_name}-${1/_/-}/g" "weights/$1/dataset-metadata.json"
  sed -i -e "s/INSERT_SLUG_HERE/${project_name}-${1/_/-}/g" "weights/$1/dataset-metadata.json"

  echo "##################################"
  echo "### show dataset-metadata.json ###"
  echo "##################################"
  cat "weights/$1/dataset-metadata.json" | jq .

  cp "config/$1.yaml" "weights/$1/"

  echo "##################################"
  echo "###        upload files        ###"
  echo "##################################"
  ls "weights/$1/"

  kaggle datasets create -p "weights/$1/"
  rm -rf "weights/$1/$1.yaml"
fi
