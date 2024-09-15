source .env

# copy models to node
mkdir -p $MODEL_DST
cp -r $MODEL_SRC/models--google--gemma-2-2b $MODEL_DST

source $VIRTUAL_ENV

python evaluation/prompt_all_subjects.py $@ 
