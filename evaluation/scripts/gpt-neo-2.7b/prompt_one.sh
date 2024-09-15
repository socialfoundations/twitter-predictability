#!/bin/bash
source .env

# copy models to node
mkdir $MODEL_DST
cp -r $MODEL_SRC/models--EleutherAI--gpt-neo-2.7B $MODEL_DST

source $VIRTUAL_ENV

echo "Running prompting..."
echo "$@"
python evaluation/prompt_one.py $@

