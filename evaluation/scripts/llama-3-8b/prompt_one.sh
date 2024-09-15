#!/bin/bash
source .env

# copy models to node
mkdir  $MODEL_DST
mkdir $MODEL_DST/llama-3-8b_10M
cp $FINETUNED_LLAMA3_PATH/* $MODEL_DST/llama-3-8b_10M
cp -r $MODEL_SRC/models--meta-llama--Meta-Llama-3-8B $MODEL_DST

#source $VIRTUAL_ENV
source $OPTIMUM_ENV

echo "Running prompting..."
echo "$@"
python evaluation/prompt_one.py $@

