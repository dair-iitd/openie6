#!/bin/bash

for i in {1..576}
do
	echo $i
	>&2 echo $i
	python carb.py --gold ../data/test/carb/test/$i.tsv --out ../dump/1.txt --props ../trained_models/traditional/test/props_output.txt
	python carb.py --gold ../data/test/carb/test/$i.tsv --out ../dump/1.txt --clausie ../trained_models/traditional/test/clausie_output.txt
done
