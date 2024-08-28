#!/bin/bash
source /home/mremeli/twitter-information-flow/.env

echo "Copying files from /fast/mremeli to /tmp..."
# copy models to node
mkdir $MODEL_DST
cp -r $MODEL_SRC/models--meta-llama--Llama-2-7b-hf $MODEL_DST

source $VIRTUAL_ENV

echo "Running prompting..."
echo "$@"
python prompt_one.py $@

