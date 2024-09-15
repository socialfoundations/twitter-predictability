source /home/mremeli/twitter-information-flow/.env

echo "Copying files from /fast/mremeli to /tmp..."
# copy models to node
mkdir -p $MODEL_DST
cp -r $MODEL_SRC/models--google--gemma-2-2b $MODEL_DST

source $VIRTUAL_ENV

python prompt_all_subjects.py $@ 
