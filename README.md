# Iterative Labelling

## Installation
```
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m nltk.downloader punkt 
```

All results have been obtained on V100 GPU with CUDA 10.0

## Download data & models
```
wget www.cse.iitd.ac.in/~kskeshav/oie6_models.tar.gz
tar -xvf oie6_models.tar.gz
wget www.cse.iitd.ac.in/~kskeshav/oie6_data.tar.gz
tar -xvf oie6_data.tar.gz
wget www.cse.iitd.ac.in/~kskeshav/rescore_model.tar.gz
tar -xvf rescore_model.tar.gz
mv rescore_model models/
```

## Running Model

New command:
```
python run.py --mode splitpredict --inp sentences.txt --out predictions.txt --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=19_eval_acc=0.548.ckpt --conj_model models/conj_model/epoch=16_eval_acc=0.890.ckpt --rescore_model models/rescore_model --num_extractions 5 
```

Expected models: \
models/conj_model: Perform coordination analysis \
models/oie_model: Perform OpenIE extraction \
models/rescore_model: Do the final rescoring \
<!--
Old Command:
```
python run.py --save models/results --mode splitpredict --predict_fp sentences.txt --task oie --predict_format allennlp --predict_out_fp split_test --oie_model models/oie_model/epoch=19_eval_acc=0.548.ckpt --conj_model models/conj_model/epoch=16_eval_acc=0.890.ckpt --gpus 1 --predict_out_fp predictions
```
-->

--inp sentences.txt - File with one sentence in each line 

--out predictions.txt - File containing the generated extractions

gpus - 0 for no GPU, 1 for single GPU

Additional flags -
```
--type labels // outputs word-level aligned labels to the file path `out`+'.labels'
--type sentences // outputs decomposed sentences to the file path `out`+'.sentences'
```

## Training Model

### Warmup Model
Training:
```
python run.py --save models/post_submission/warmup_model --mode train --model_str bert-base-cased --task oie --epochs 30 --gpus 1 --batch_size 32 --optimizer adam --lr 2e-05 --iterative_layers 2
```

Testing:
```
python run.py --save models/post_submission/warmup_model --mode test --model_str bert-base-cased --task oie --gpus 1
```
F1: 52.5, AUC: 32.3

<!-- python run.py --save models/oie/* --mode train --model_str bert-base-cased --task oie --epochs 20 --gpus 1 --iterative_layers 2 --add_depth --add_span --add_pos --add_verb 
python run.py --save models/oie/base --mode test --model_str bert-base-cased --task oie --gpus 1 --checkpoint models/submission/oie/warmup_model/epoch=19_eval_acc=0.522.ckpt -->

### Constrained Model
Training
```
python run.py --save models/post_submission/const_model --mode resume --model_str bert-base-cased --task oie --epochs 21 --gpus 1 --batch_size 24 --optimizer adam --lr 2e-05 --iterative_layers 2 --checkpoint models/post_submission/warmup_model/epoch=18_eval_acc=0.544.ckpt --constraints posm_hvc_hvr_hve --save_k 3 --accumulate_grad_batches 2 --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 1 --cweights 10_10_10_10 --val_check_interval 0.1
```

Testing
```
python run.py --save models/post_submission/const_model --mode test --model_str bert-base-cased --task oie --gpus 1
```
F1: 54.0, AUC: 35.4

Predicting
```
python run.py --save models/post_submission/const_model --mode predict --model_str bert-base-cased --task oie --gpus 1 --predict_fp sentences.txt
```

<!-- 
python run.py --save models/oie/* --mode test --model_str bert-base-cased --task oie --epochs 22 --gpus 1 --iterative_layers 2 --checkpoint models/oie/may_19/const_12/epoch=20_eval_acc=0.536.ckpt --constraints posm_hvc_hvr_hve --save_k 1 --batch_size 16 --accumulate_grad_batches 2 --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 0.30 --cweights 1_1_1_1 --val_check_interval 0.1
python run.py --save models/oie/base --mode test --model_str bert-base-cased --task oie --gpus 1 --checkpoint models/submission/oie/const_model/epoch=20_eval_acc=0.536.ckpt
-->

### Final Model
<!-- Running
```
python run.py --save models/post_submission/final_model --mode splitpredict --predict_fp carb/data/carb_sentences.txt --task oie --predict_format allennlp --predict_out_fp split_test --oie_model models/post_submission/const_model/epoch=19_eval_acc=0.548.ckpt --conj_model models/submission/conj/large/epoch=16_eval_acc=0.890.ckpt --gpus 1
```
F1: 50.4, 32.3

Confidence Re-scoring
```
# imojie folder
python imojie/aggregate/score.py --model_dir models/be --inp_fp ~/conjunctions/models/post_submission/final_model/split_test.allennlp --out_fp ~/conjunctions/models/post_submission/final_model/split_test.allennlp.conf --topk 5
python carb/carb.py --allennlp models/post_submission/final_model/split_test.allennlp.conf --gold carb/data/gold/test.tsv --out /dev/null
```
F1: 52.8, AUC: 34.3 -->

Running
```
python run.py --mode splitpredict --inp carb/data/carb_sentences.txt --out models/results/final --task oie --gpus 1 --oie_model models/oie_model/epoch=19_eval_acc=0.548.ckpt --conj_model models/conj_model/epoch=16_eval_acc=0.890.ckpt --rescoring --rescore_model models/rescore_model --num_extractions 5 
python utils/oie_to_allennlp.py --inp models/results/final --out models/results/final.carb
python carb/carb.py --allennlp models/results/final.carb --gold carb/data/gold/test.tsv --out /dev/null
```
F1: 52.4, AUC: 34.4

<!-- python run.py --save models/oie/base/ --mode splitpredict --predict_fp ~/imojie/data/test/carb_sentences.txt --task oie --predict_format allennlp --predict_out_fp split_test --oie_model models/submission/oie/const_model/epoch=20_eval_acc=0.536.ckpt --conj_model models/submission/conj/large/epoch=16_eval_acc=0.890.ckpt --gpus 1
python run.py --save models/post_submission/const_model --mode test --model_str bert-base-cased --task oie --gpus 1 -->

### Running Coordination Analysis
```
python run.py --save models/conj/model --mode train_test --model_str bert-large-cased --task conj --batch_size 24 --accumulate 2 --epochs 20 --gpus 1
```

