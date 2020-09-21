#!/bin/bash

# cd code
exp_dir=$1
test_=$2
pred_dir=$3
num_process=$4
beam_size=$5
mkdir -p $exp_dir/$pred_dir
i=0
for weights in $exp_dir/model_state_epoch_*.th
do
	base=$(basename $weights)
	num=$(echo $base | tr -dc '0-9')
	echo $num
	i=$(($i+1))
	# $HOME/.local/bin/allennlp
	allennlp predict $exp_dir $test_ --include-package imojie --predictor noie_seq2seq --output-file $exp_dir/$pred_dir/output_$num.jsonl --weights-file $weights --cuda-device 0 --batch-size 128 --overrides '{"model": {"beam_size": '$beam_size'}}' &
	if [ "$(($i%$num_process))" -eq "0" ]; then
		wait
	fi
done

wait

