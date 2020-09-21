extractions_fp=$1
gold_fp=$2
echo "carb(s,s)"
python carb/carb.py --allennlp $extractions_fp --gold $gold_fp --out /dev/null --single_match
echo "carb(s,m)"
python carb/carb.py --allennlp $extractions_fp --gold $gold_fp --out /dev/null
echo "oie16"
python carb/oie16.py --allennlp $extractions_fp --gold $gold_fp --out /dev/null 
echo "wire57"
python carb/wire57_evaluation.py --system $extractions_fp --gold carb/data/test_gold_allennlp_format.txt 
