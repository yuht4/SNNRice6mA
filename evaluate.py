from keras.models import load_model
from group_norm import GroupNormalization
import re
import os, sys, copy, getopt, re, argparse
import random
import pandas as pd 
import numpy as np


def dataProcessing(path):

    data = pd.read_csv(path);
    alphabet = np.array(['A', 'G', 'T', 'C'])
    X = [];
    for line in data['data']:

        line = list(line.strip('\n'));
        
        seq = np.array(line, dtype = '|U1').reshape(-1,1);
        seq_data = []

        for i in range(len(seq)):
            if seq[i] == 'A':
                seq_data.append([1,0,0,0])
            if seq[i] == 'T':
                seq_data.append([0,1,0,0])
            if seq[i] == 'C':
                seq_data.append([0,0,1,0])
            if seq[i] == 'G':
                seq_data.append([0,0,0,1])

        X.append(np.array(seq_data));
        
    X = np.array(X);
    y = np.array(data['label'], dtype = np.int32);
 
    return X, y; #(n, 41, 4), (n,)

def main():

    parser = argparse.ArgumentParser(description="predicting 6mA sites in rice genome")
    parser.add_argument("--h5File", type=str, help="the model h5 file", required=True)
    parser.add_argument("--csv", type=str, help="input csv", required=True)
    args = parser.parse_args()

    Path = os.path.abspath(args.h5File)
    csv = os.path.abspath(args.csv)

    if not os.path.exists(Path):
        print("The model not exist! Error\n")
        sys.exit()
    if not os.path.exists(csv):
        print("The model not exist! Error\n")
        sys.exit()

    # print(len(sequence))
    # print(sequence)

    X, y = dataProcessing(csv)
    keras_Model = load_model(Path)

    score = keras_Model.evaluate(X, y)

    print('accuray is ' + str(score[1]))

    
if __name__ == "__main__":
    main()
