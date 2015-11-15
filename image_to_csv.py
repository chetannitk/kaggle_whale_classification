import numpy as np
import cv2
import itertools
import csv
import os

whale_map = {}
with open('train.csv','rb') as csvfile:
    csvreader=csv.reader(csvfile, delimiter=',')
    #print type(csvreader)
    f=0
    for row in csvreader:
        if f==0:
            f=1
            continue
        whale_map[row[0]] = row[1].split('_')[1]

## Create the comma seperated row for train data
f = open("train_whale.csv","w") 
f1 = open("test_whale.csv","w") 
for i in os.listdir('/media/Entertainment/imgs/'):
    #print i
    if i.endswith(".jpg"): 
        ##print i
        img = cv2.imread('/media/Entertainment/imgs/'+i, 0)
        #if img==None:
        #    continue
        resized = cv2.resize(img, (80,80), interpolation = cv2.INTER_AREA)
        merged = list(itertools.chain(*resized))
        if i not in whale_map:
            merged.append(i)
            f1.write(",".join([str(m) for m in merged])+"\n")
        else:
            merged.append(whale_map[i])
            f.write(",".join([str(m) for m in merged])+"\n")
f.close()
f1.close()
