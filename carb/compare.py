# outputs examples where the old eval and new eval rank the two systems differently
# Use gap to bring out only those examples where the gap is significant

import sys
import argparse
import ipdb

def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--oldeval")
	parser.add_argument("--neweval")

	return parser


# AUC: 0.465	 Optimal (precision, recall, F1): [0.603 0.58  0.591]
def parse_eval_line(line):
	auc = float(line[ 5: line.index('\t')])
	temp = line[ line.index("[")+1: line.index("]")]
	temp = temp.split(' ')
	temp = [x for x in temp if x!=""]
	p,r,f1 = float(temp[0]), float(temp[1]), float(temp[2])

	return (auc, p, r, f1)

def get_numbers(filename):
	f=open(filename, 'r')
	lines=f.readlines()
	f.close()

	scores=dict()
	sys1, sys2 = None, None
	for line in lines:
		if "AUC" not in line:
			if line!="\n":
				eg = int(line.strip())
	
			if sys1 is not None:
				scores[eg] = (sys1, sys2)
			sys1, sys2 = None, None

		elif sys1 is None:
			sys1 = parse_eval_line(line)
		
		elif sys2 is None and sys1 is not None:
			sys2 = parse_eval_line(line)

	return scores


if __name__ == '__main__':
	parser = parse_args()
	args = parser.parse_args()

	# print(args)

	# if args.model=='gold':
	# 	gold_reader(args.file)
	
	old_scores = get_numbers(args.oldeval)
	new_scores = get_numbers(args.neweval)

	gap = 0.1
	sys1_better_examples = []
	sys2_better_examples = []

	for i in old_scores.keys():
		if i not in new_scores.keys():
			continue

		# when old eval says sys1 is better and new eval says sys2 is better
		if old_scores[i][0][3] + gap < old_scores[i][1][3] and new_scores[i][0][3] > new_scores[i][1][3] + gap:
			sys1_better_examples.append(i)

		# when new eval says sys1 is better and old eval says sys2 is better
		if new_scores[i][0][3] + gap < new_scores[i][1][3] and old_scores[i][0][3] > old_scores[i][1][3] + gap:
			sys2_better_examples.append(i)

	print("new eval says :")
	print("sys1 is better at")
	print(sys1_better_examples)
	print("sys2 is better at")
	print(sys2_better_examples)

