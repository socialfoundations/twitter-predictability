#!/bin/bash
source .env

echo "Submitting jobs.."
subjects_path=$SUBJECT_DATA_PATH
dirs=(${subjects_path}/*)
for entry in "${dirs[@]}"
do
	subject=${entry##*/}
	res_path=$FINETUNED_MODEL_PATH/$subject
	if [ -d "$res_path" ]; then
		echo "Already processed"
	else
		echo "$subject"
		condor_submit_bid 25 finetune/user_finetune.sub User=$subject
	fi
done

#array=( 987267104 991137303279079424 )
#for i in "${array[@]}"
#do
#	echo "$i"
#	condor_submit_bid 25 user_finetune.sub User=$i
#done

#condor_submit_bid 25 user_finetune.sub User=276697103
