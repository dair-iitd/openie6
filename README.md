# OpenIE6 System 

This repository contains the code for the paper:\
OpenIE6: Iterative Grid Labeling and Coordination Analysis for Open Information Extraction\
Keshav Kolluru*, Vaibhav Adlakha*, Samarth Aggarwal, Mausam and Soumen Chakrabarti\
EMNLP 2020

\* denotes equal contribution

## Installation
```
conda create -n openie6 python=3.6
conda activate openie6
pip install -r requirements.txt
python -m nltk.downloader stopwords
python -m nltk.downloader punkt 
```

All results have been obtained on V100 GPU with CUDA 10.0
NOTE: HuggingFace transformers==2.6.0 is necessary. The latest version has a breaking change in the way tokenizer is used in the code. It will not raise an error but will give wrong results!

## Download Resources
Download Data (50 MB)
```
zenodo_get 4054476
tar -xvf openie6_data.tar.gz
```

Download Models (6.6 GB)
```
zenodo_get 4055395
tar -xvf openie6_models.tar.gz
```
<!-- wget www.cse.iitd.ac.in/~kskeshav/oie6_models.tar.gz
tar -xvf oie6_models.tar.gz
wget www.cse.iitd.ac.in/~kskeshav/oie6_data.tar.gz
tar -xvf oie6_data.tar.gz
wget www.cse.iitd.ac.in/~kskeshav/rescore_model.tar.gz
tar -xvf rescore_model.tar.gz
mv rescore_model models/ -->

## Running Model

New command:
```
python run.py --mode splitpredict --inp sentences.txt --out predictions.txt --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 
```

Expected models: \
models/conj_model: Performs coordination analysis \
models/oie_model: Performs OpenIE extraction \
models/rescore_model: Performs the final rescoring 

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
python run.py --save models/warmup_oie_model --mode train_test --model_str bert-base-cased --task oie --epochs 30 --gpus 1 --batch_size 24 --optimizer adamW --lr 2e-05 --iterative_layers 2
```

Testing:
```
python run.py --save models/warmup_oie_model --mode test --batch_size 24 --model_str bert-base-cased --task oie --gpus 1
```
Carb F1: 52.4, Carb AUC: 33.8


Predicting
```
python run.py --save models/warmup_oie_model --mode predict --model_str bert-base-cased --task oie --gpus 1 --inp sentences.txt --out predictions.txt
```

Time (Approx): 142 extractions/second

### Constrained Model
Training
```
python run.py --save models/oie_model --mode resume --model_str bert-base-cased --task oie --epochs 16 --gpus 1 --batch_size 16 --optimizer adam --lr 5e-06 --iterative_layers 2 --checkpoint models/warmup_oie_model/epoch=13_eval_acc=0.544.ckpt --constraints posm_hvc_hvr_hve --save_k 3 --accumulate_grad_batches 2 --gradient_clip_val 1 --multi_opt --lr 2e-5 --wreg 1 --cweights 3_3_3_3 --val_check_interval 0.1
```

Testing
```
python run.py --save models/oie_model --mode test --batch_size 16 --model_str bert-base-cased --task oie --gpus 1 
```
Carb F1: 54.0, Carb AUC: 35.7

Predicting
```
python run.py --save models/oie_model --mode predict --model_str bert-base-cased --task oie --gpus 1 --inp sentences.txt --out predictions.txt
```

Time (Approx): 142 extractions/second

NOTE: Due to a bug in the code, [link](https://github.com/dair-iitd/openie6/issues/10), we end up using a loss function based only on the constrained loss term and not the original Cross Entropy (CE) loss. It still seems to work well as the warmup model is already trained with the CE loss and the constrained training is initialized from the warmup model.

### Running Coordination Analysis
```
python run.py --save models/conj_model --mode train_test --model_str bert-large-cased --task conj --epochs 40 --gpus 1 --batch_size 32 --optimizer adamW --lr 2e-05 --iterative_layers 2
```

F1: 87.8

### Final Model

Running
```
python run.py --mode splitpredict --inp carb/data/carb_sentences.txt --out models/results/final --rescoring --task oie --gpus 1 --oie_model models/oie_model/epoch=14_eval_acc=0.551_v0.ckpt --conj_model models/conj_model/epoch=28_eval_acc=0.854.ckpt --rescore_model models/rescore_model --num_extractions 5 
python utils/oie_to_allennlp.py --inp models/results/final --out models/results/final.carb
python carb/carb.py --allennlp models/results/final.carb --gold carb/data/gold/test.tsv --out /dev/null
```
Carb F1: 52.7, Carb AUC: 33.7
Time (Approx): 31 extractions/second

Evaluate using other metrics (Carb(s,s), Wire57 and OIE-16)
```
bash carb/evaluate_all.sh models/results/final.carb carb/data/gold/test.tsv
```

Carb(s,s): F1: 46.4, AUC: 26.8
Carb(s,m) ==> Carb: F1: 52.7, AUC: 33.7
OIE16: F1: 65.6, AUC: 48.4
Wire57: F1: 40.0

## CITE
If you use this code in your research, please cite:

```
@inproceedings{kolluru&al20,
    title = "{O}pen{IE}6: {I}terative {G}rid {L}abeling and {C}oordination {A}nalysis for {O}pen {I}nformation {E}xtraction",\
    author = "Kolluru, Keshav  and
      Adlakha, Vaibhav and
      Aggarwal, Samarth and
      Mausam, and
      Chakrabarti, Soumen",
    booktitle = "The 58th Annual Meeting of the Association for Computational Linguistics (ACL)",
    month = July,
    year = "2020",
    address = {Seattle, U.S.A}
}
```


## LICENSE

Note that the license is the full GPL, which allows many free uses, but not its use in proprietary software which is distributed to others. For distributors of proprietary software, you can contact us for commercial licensing.

## CONTACT

In case of any issues, please send a mail to ```keshav.kolluru (at) gmail (dot) com```

