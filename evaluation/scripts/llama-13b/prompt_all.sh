source /home/mremeli/twitter-information-flow/.env

echo "Copying files from /fast/mremeli to /tmp..."
# copy models to node
mkdir $MODEL_DST
cp -r $MODEL_SRC/models--meta-llama--Llama-2-13b-hf $MODEL_DST

source $VIRTUAL_ENV

python prompt_all_subjects.py $@ 
