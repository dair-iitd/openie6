#!/bin/bash

# for i in {0..633};
# do
# 	echo $i
# 	python carb.py --allennlp ../trained_models/wiki5/bert_append/run2/test/pro_output_7_23.0.txt --gold ../data/test/carb/test/$i.tsv --out /dev/null
# 	python carb.py --allennlp ../trained_models/wiki5/bert_append/run2/test/pro_output_7_23.4.txt --gold ../data/test/carb/test/$i.tsv --out /dev/null
# done

echo "With new benchmark and new data"
# python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --ollie ../models/traditional/test/ollie_output.txt
# python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --props ../models/traditional/test/props_output.txt
# python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --openiefour ../models/traditional/test/openie4_output.txt
# python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --openiefive ../models/traditional/test/openie5_output.txt
# python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --clausie ../models/traditional/test/clausie_output.txt

python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --ollie ../../carb-repo/system_outputs/test/ollie_output.txt
python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --props ../../carb-repo/system_outputs/test/props_output.txt
python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --openiefour ../../carb-repo/system_outputs/test/openie4_output.txt
python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --openiefive ../../carb-repo/system_outputs/test/openie5_output.txt
python carb.py --gold ../data/test/carb/test.tsv --out /dev/null --clausie ../../carb-repo/system_outputs/test/clausie_output.txt
