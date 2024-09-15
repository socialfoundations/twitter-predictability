source .env

# copy models to node
mkdir -p $MODEL_DST
cp -r $MODEL_SRC/models--mistralai--Mistral-7B-v0.3 $MODEL_DST

source $VIRTUAL_ENV

python evaluation/prompt_all_subjects.py $@ 
