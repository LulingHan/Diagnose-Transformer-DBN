from random import shuffle
import argparse, itertools

parser = argparse.ArgumentParser(description = "shuffle the input data and labels")
parser.add_argument('input', help = 'file path of input data')
parser.add_argument('label', help = 'file path of label')
parser.add_argument('outputx', help = 'file path of output x')
parser.add_argument('outputy', help = 'file path of output y')

args = parser.parse_args()
file_input = open(args.input)
file_label = open(args.label)
outx = open(args.outputx, 'w')
outy = open(args.outputy, 'w')

data = file_input.readlines()
label = file_label.readlines()
index = range(len(label))
shuffle(index)

for i in index:
    outx.write(data[i])
    outy.write(label[i])

outx.close()
outy.close()
