exp_dir=$1
gold=$2

echo "" > $exp_dir/results.txt
for file in $(ls -v $exp_dir/pro_output_*.txt);
do
    echo $file >> $exp_dir/results.txt
	python benchmark/carb.py --gold=$gold --allennlp=$file --out $file.dat | tee -a $exp_dir/results.txt
done
