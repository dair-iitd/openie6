# exp_dir="../../models/msr_ie/bahdanu_copy_1/"
# gold="../data/test/lower/test.oie.orig"

exp_dir=$1
gold=$2

# for file in $exp_dir/pred_1/pro_output_*.txt;
# do
	# python benchmark.py --gold=$gold --msr=$file --out $file.dat > /dev/null
# done

for file in $exp_dir/pred_5/pro_output_*.txt;
do
	python oie16.py --gold=$gold --msr=$file --out $file.dat > /dev/null
done
