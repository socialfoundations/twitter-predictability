#!/bin/bash
source .env

# copy models to node
mkdir  $MODEL_DST
cp -r $MODEL_SRC/models--meta-llama--Meta-Llama-3-70B $MODEL_DST

#source $VIRTUAL_ENV
source $OPTIMUM_ENV

echo "Running prompting..."
echo "$@"
python evaluation/prompt_one.py $@

