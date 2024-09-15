source /home/mremeli/twitter-information-flow/.env

echo "Copying files from /fast/mremeli to /tmp..."
# copy models to node
mkdir $MODEL_DST
cp -r $MODEL_SRC/models--Qwen--Qwen2-72B $MODEL_DST

source $VIRTUAL_ENV

python prompt_all_subjects.py $@ 
