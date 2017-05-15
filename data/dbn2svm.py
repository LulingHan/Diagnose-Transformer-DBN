import argparse, itertools
import numpy as np

parser = argparse.ArgumentParser(description='Transform the data format of DBN to SVM')

parser.add_argument('input', help = 'file path of input data')
parser.add_argument('label', help = 'file path of label')
parser.add_argument('output', help = 'file path of output')

args = parser.parse_args()
file_input = open(args.input)
file_label = np.loadtxt(args.label)
file_output = open(args.output, 'w')

for x, y in itertools.izip(file_input, file_label):
    label = np.argmax(y) + 1
    file_output.write("%d " % label)
    cnt = 0
    for feature in x.split():
        cnt += 1
        file_output.write("%d:%s " % (cnt, feature))
    file_output.write('\n')

file_output.close()