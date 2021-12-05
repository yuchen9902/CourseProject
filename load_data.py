from os import listdir
import numpy as np
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')

def loadFolder(directory):
    output = []
    for file in listdir(directory):
        full_path = directory+file
        text = []
        with open (full_path, 'rb') as file:
            for line in file:
                text+=tokenizer.tokenize(line.decode(errors='ignore'))
        output.append(text)
    return output

def loadData(train_dir, test_dir):
    train_pos = loadFolder(train_dir + '/pos/')
    train_neg = loadFolder(test_dir + '/neg/')
    test_pos = loadFolder(test_dir + '/pos/')
    test_neg = loadFolder(test_dir + '/neg/')

    train_data = train_pos+train_neg
    test_data = test_pos+test_neg

    train_label = np.array([1]*len(train_pos)+[0]*len(train_neg))
    test_label = np.array([1]*len(test_pos)+[0]*len(test_neg))

    return test_data,test_label,train_data,train_label
