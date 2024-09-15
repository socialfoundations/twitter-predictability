source .env

# copy models to node
mkdir $MODEL_DST 
cp -r $MODEL_SRC/models--gpt2-xl $MODEL_DST
cp -r $FINETUNED_GPT2_PATH $MODEL_DST

source $VIRTUAL_ENV 

echo "Running prompting..."
echo "$@"
python evaluation/prompt_all_subjects.py $@ 
