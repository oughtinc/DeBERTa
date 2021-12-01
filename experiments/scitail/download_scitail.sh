#!/bin/bash

if [[ $# -eq 0 ]] ; then
    echo "Please supply a directory"
    exit 1
fi
DIR=$1
if [[ ! -d "$DIR/scitail" ]]; then
  mkdir -p "$DIR/scitail"
  curl -s -o $DIR/scitail.zip https://ai2-public-datasets.s3.amazonaws.com/scitail/SciTailV1.1.zip 
  unzip -qq $DIR/scitail.zip -d $DIR
  mv $DIR/SciTailV1.1/tsv_format/scitail_1.0_train.tsv $DIR/scitail/train.tsv
  mv $DIR/SciTailV1.1/tsv_format/scitail_1.0_dev.tsv $DIR/scitail/dev.tsv
  mv $DIR/SciTailV1.1/tsv_format/scitail_1.0_test.tsv $DIR/scitail/test.tsv
  rm -rf $DIR/SciTailV1.1
  rm -rf $DIR/scitail.zip
fi
