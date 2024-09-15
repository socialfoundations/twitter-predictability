source .env

# copy models to node
mkdir $MODEL_DST
cp -r $MODEL_SRC/models--meta-llama--Llama-2-70b-hf $MODEL_DST

source $VIRTUAL_ENV

python evaluation/prompt_all_subjects.py $@ 
