#!/bin/bash
source .env

# copy models to node
cp -r $MODEL_SRC /tmp/hub
# copy cached datasets to node
cp -r $DATASET_SRC /tmp/datasets

source $VIRTUAL_ENV

echo "Finetuning..."
python finetuning/run_clm.py $@
