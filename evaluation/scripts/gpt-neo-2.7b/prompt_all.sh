source .env

# copy models to node
mkdir $MODEL_DST
cp -r $MODEL_SRC/models--EleutherAI--gpt-neo-2.7B $MODEL_DST

source $VIRTUAL_ENV

python evaluation/prompt_all_subjects.py $@ 
