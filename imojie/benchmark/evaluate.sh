exp_dir=$1
eval=$2
perform=${3:-gen_pro_carb_compile}

eval_dir=$exp_dir/$eval'_'$beam_size
if [[ $perform == *"gen"* ]]; then
    echo "============== GENERATING OUTPUTS =============="
    bash benchmark/gen_outputs.sh $exp_dir data/$eval'_sentences.txt' $eval'_'$beam_size $num_process $beam_size
fi

if [[ $perform == *"pro"* ]]; then
    echo "============== PROCESSING OUTPUTS =============="
    python benchmark/process_outputs.py --pred_dir $eval_dir --test data/$eval'_sentences.txt' --type $type
fi

if [[ $perform == *"carb"* ]]; then
    echo "============== EVALUATING OVER CARB =============="
    bash benchmark/carb.sh $eval_dir data/$eval/extractions.tsv
fi

if [[ $perform == *"compile"* ]]; then
    echo "============== COMPILING RESULTS =============="
    python benchmark/compile_results.py --inp_dir $eval_dir | tee -a $eval_dir/best_results.txt
fi


if [[ $perform == *"rerank"* ]]; then
    if [ "$type" == "append" ]
    then
        export eval_dir=$(realpath $eval_dir)

        if [[ $eval_dir == *wiki4* ]]; 
        then
            prob_dir=../models/wiki4/bert_encoder/conf00
        elif [[ $eval_dir == *wiki5* ]];
        then
            prob_dir=../models/wiki5/bert_encoder/conf00
        elif [[ $eval_dir == *wikic* ]];
        then
            prob_dir=../models/wikic/bert_encoder/conf00
        else
            prob_dir=../models/wiki4/bert_encoder/conf00
        fi
        echo "prob dir: "$prob_dir
        prob_dir=$(realpath $prob_dir)
        # best_epoch=$(head -n 5 $prob_dir/$eval'_5'/best_results.txt | grep "Best\ Sum\ Scores.*" | sed 's/Best\ Sum\ Scores.*, \([0-9]*\)\/[0-9]*.*/\1/')

        echo 'Rerank Directory: '$prob_dir
        # echo 'Best Epoch: '$best_epoch

        cd ..
        mkdir -p $eval_dir/rerank
        # python code/noie/probability.py --model_dir $prob_dir --epoch_num $best_epoch --inp_dir $eval_dir --out_dir $eval_dir/rerank --type single
        python code/noie/probability.py --model_dir $prob_dir --inp_dir $eval_dir --out_dir $eval_dir/rerank --type single
        
        echo "generated new probs"
        cd benchmark
        bash carb.sh $eval_dir/rerank ../data/test/carb/$eval.tsv
        python compile_results.py --inp_dir $eval_dir/rerank | tee -a $eval_dir/rerank/best_results.txt
            cp $eval_dir/rerank/best_results.txt $eval_dir/best_results.txt.rerank
    fi
fi

