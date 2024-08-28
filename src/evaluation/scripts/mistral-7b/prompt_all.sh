source /home/mremeli/twitter-information-flow/.env

echo "Copying files from /fast/mremeli to /tmp..."
# copy models to node
mkdir -p $MODEL_DST
cp -r $MODEL_SRC/models--mistralai--Mistral-7B-v0.3 $MODEL_DST

source $VIRTUAL_ENV

python prompt_all_subjects.py $@ 
