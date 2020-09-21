# CaRB - A Crowdsourced Benchmark for Open IE

CaRB : ***C***rowdsourced ***a***utomatic open ***R***elation extraction ***B***enchmark


## Introduction

CaRB is a dataset cum evaluation framework for benchmarking Open Information Extraction systems.

The details of this benchmark are elaborated in our [EMNLP 2019 Paper](https://www.aclweb.org/anthology/D19-1651/).

### Citing
If you use this software, please cite:
```
@inproceedings{bhardwaj-etal-2019-carb,
    title = "{C}a{RB}: A Crowdsourced Benchmark for Open {IE}",
    author = "Bhardwaj, Sangnie  and
      Aggarwal, Samarth  and
      Mausam, Mausam",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)",
    month = nov,
    year = "2019",
    address = "Hong Kong, China",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D19-1651",
    doi = "10.18653/v1/D19-1651",
    pages = "6263--6268",
    abstract = "Open Information Extraction (Open IE) systems have been traditionally evaluated via manual annotation. Recently, an automated evaluator with a benchmark dataset (OIE2016) was released {--} it scores Open IE systems automatically by matching system predictions with predictions in the benchmark dataset. Unfortunately, our analysis reveals that its data is rather noisy, and the tuple matching in the evaluator has issues, making the results of automated comparisons less trustworthy. We contribute CaRB, an improved dataset and framework for testing Open IE systems. To the best of our knowledge, CaRB is the first crowdsourced Open IE dataset and it also makes substantive changes in the matching code and metrics. NLP experts annotate CaRB{'}s dataset to be more accurate than OIE2016. Moreover, we find that on one pair of Open IE systems, CaRB framework provides contradictory results to OIE2016. Human assessment verifies that CaRB{'}s ranking of the two systems is the accurate ranking. We release the CaRB framework along with its crowdsourced dataset.",
}
```

### Contact
Leave us a note at 
```samarthaggarwal2510 (at) gmail (dot) com```

## Requirements

* Python 3
* See required python packages [here](requirements.txt).



## Evaluating an Open IE Extractor

Currently, we support the following Open IE output formats:

* [ClausIE](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/software/clausie/)
* [OLLIE](http://knowitall.github.io/ollie/)
* [OpenIE-4](https://github.com/allenai/openie-standalone)
* [OpenIE-5](https://github.com/allenai/openie-standalone)
* [PropS](http://u.cs.biu.ac.il/~stanovg/props.html)
* [ReVerb](http://reverb.cs.washington.edu/)
* [Stanford Open IE](http://nlp.stanford.edu/software/openie.html)
* Tab Separated - Read simple tab format file, where each line consists of:
                                sent, prob, pred,arg1, arg2, ...

To evaluate your OpenIE system:

1. Run your extractor over the [dev sentences](data/dev.txt) or [test sentences](data/test.txt) and store the output into "*your_output*.txt"

2. Depending on your output format, you can get a precision-recall curve by running [carb.py](carb.py):
``` 
Usage:
   python carb.py --gold=GOLD_OIE --out=OUTPUT_FILE (--stanford=STANFORD_OIE | --ollie=OLLIE_OIE |--reverb=REVERB_OIE | --clausie=CLAUSIE_OIE | --openiefour=OPENIEFOUR_OIE | --props=PROPS_OIE)

Options:
  --gold=GOLD_OIE              The gold reference Open IE file (by default, it should be under ./oie_corpus/all.oie).
  --out=OUTPUT_FILE            The output file, into which the precision recall curve will be written.
  --clausie=CLAUSIE_OIE        Read ClausIE format from file CLAUSIE_OIE.
  --ollie=OLLIE_OIE            Read OLLIE format from file OLLIE_OIE.
  --openiefour=OPENIEFOUR_OIE  Read Open IE 4 format from file OPENIEFOUR_OIE.
  --openiefive=OPENIEFIVE_OIE  Read Open IE 5 format from file OPENIEFIVE_OIE.
  --props=PROPS_OIE            Read PropS format from file PROPS_OIE
  --reverb=REVERB_OIE          Read ReVerb format from file REVERB_OIE
  --stanford=STANFORD_OIE      Read Stanford format from file STANFORD_OIE
  --tabbed=TABBED_OIE		   Read tabbed format from file TABBED_OIE
```

## Evaluating Existing Systems

In the course of this work we tested the above mentioned Open IE parsers against our benchmark.
We provide the output files (i.e., Open IE extractions) of each of these
systems in [system_outputs/test](system_outputs/test).
You can give each of these files to [carb.py](carb.py), to get the corresponding precision recall curve.

For example, to evaluate Stanford Open IE output, run:
```
python carb.py --gold=data/gold/test.tsv --out=dump/OpenIE-4.dat --openiefour=system_outputs/test/openie4_output.txt
```

## Plotting

You can plot together multiple outputs of [carb.py](carb.py), by using [pr_plot.py](pr_plot.py):

```
Usage:
   pr_plot --in=DIR_NAME --out=OUTPUT_FILENAME 

Options:
  --in=DIR_NAME            Folder in which to search for *.dat files, all of which should be in a P/R column format (outputs from benchmark.py).
  --out=OUTPUT_FILENAME    Output filename, filetype will determine the format. Possible formats: pdf, pgf, png
```

### References

1. Creating a large benchmark for Open Information Extraction - Stanovsky and Dagan, 2016
2. Analysing Errors of Open Information Extraction Systems - Schneider et al., 2017
3. Wire57 : A Fine-Grained Benchmark for Open Information Extraction - Lechelle et al., 2018


