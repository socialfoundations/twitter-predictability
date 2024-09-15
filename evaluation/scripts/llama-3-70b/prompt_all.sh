source .env

# copy models to node
mkdir -p $MODEL_DST
cp -r $MODEL_SRC/models--meta-llama--Meta-Llama-3-70B $MODEL_DST

#source $VIRTUAL_ENV
source $OPTIMUM_ENV

python evaluation/prompt_all_subjects.py $@ 
