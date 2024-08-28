source /home/mremeli/twitter-information-flow/.env

echo "Copying files from /fast/mremeli to /tmp..."
# copy models to node
mkdir  $MODEL_DST
mkdir $MODEL_DST/llama-3-8b_10M
cp /home/mremeli/github_repo_clones/axolotl/outputs/llama-3-8b_10M/* $MODEL_DST/llama-3-8b_10M
cp -r $MODEL_SRC/models--meta-llama--Meta-Llama-3-8B $MODEL_DST

# source $VIRTUAL_ENV
source $OPTIMUM_ENV

echo "Running prompting..."
echo "$@"
python prompt_all_subjects.py $@ 
