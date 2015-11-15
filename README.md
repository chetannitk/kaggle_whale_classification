Steps Followed in Whale Classification problem
====================================================
1) Download the 8GB dataset of train/test whale images.
2) Create CSV data for training and test.
   --> train_whale.csv.
   --> test_whale.csv.
3) The CSV dataset contain pixels intensity with grayscale and resize to 80X80 resolution.
   --> Total number of classes = 447.
4) train_map.py contain training dataset maping of whale image name to whale ID.
5) Created train.pkl, test.pkl pickle files used in whale_detection.py to avoid redundant
   processing of conversion of csv data to numpy array.
6) Used the DBN with 500 hidden node and 50 epoch.
7) Classification result are stored in result.txt.




