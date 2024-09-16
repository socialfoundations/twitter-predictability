#!/bin/bash
source .env

source $VIRTUAL_ENV

# run all scenarios - different combinations of contexts for finetuning
echo "Finetuning..."
python finetune/user_finetune.py --finetune_on user $@
python finetune/user_finetune.py --finetune_on peer $@
python finetune/user_finetune.py --finetune_on random $@
python finetune/user_finetune.py --finetune_on user peer $@
python finetune/user_finetune.py --finetune_on user random $@
python finetune/user_finetune.py --finetune_on peer random $@
python finetune/user_finetune.py --finetune_on user peer random $@
