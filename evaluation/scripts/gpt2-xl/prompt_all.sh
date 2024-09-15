source /home/mremeli/twitter-information-flow/.env

echo "Copying files from /fast/mremeli to /tmp..."
# copy models to node
mkdir $MODEL_DST 
cp -r $MODEL_SRC/models--gpt2-xl $MODEL_DST
cp -r /fast/mremeli/twitter-information-flow/finetune/out_subject_eval/gpt2-xl_10M_constant_lr_eos_sep $MODEL_DST

source $VIRTUAL_ENV 

echo "Running prompting..."
echo "$@"
python prompt_all_subjects.py $@ 
