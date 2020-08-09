__original__author__ = 'chetannaik'
__modified__by__ = 'dmmartin888'

#https://github.com/chetannaik/bee_classifier_using_cnn
#Modified to just assign files to the right directory.
#Additional file manipulation done in the model.

from PIL import Image
from tqdm import tqdm

import numpy as np
import pandas as pd
import os
import shutil
import time

# config
size = (48, 48)
num_channels = 3
num_apis_images = 827
num_bombus_images = 3142
train_data_dir = "dataset/images/train/"
train_labels_file = "dataset/train_labels.csv"


# copy files to appropriate directories based on the class
def copy_files():
	print("-- Copying files to data/train directory")
	# Parameters
	labels = pd.read_csv(train_labels_file)
	labels.index = labels.id
	labels = labels['genus']
	labels = labels.T.to_dict()

	if not os.path.exists("data/train/bombus/"):
		os.makedirs("data/train/bombus/")

	if not os.path.exists("data/train/apis/"):
		os.makedirs("data/train/apis/")

	for infile in tqdm(os.listdir(train_data_dir)):
		print("just infile", infile)
		print("infile[:4]", infile[:4], '\n')
		if infile == ".DS_Store":
			continue
		if labels[int(infile[:-4])] == 1.0:
			# Bombus
			file_src = train_data_dir + infile
			file_dest = 'data/train/bombus/' + infile
			shutil.copy2(file_src, file_dest)
		elif labels[int(infile[:-4])] == 0.0:
			# Apis
			file_src = train_data_dir + infile
			file_dest = 'data/train/apis/' + infile
			shutil.copy2(file_src, file_dest)


def main():
	t_start = time.time()
	copy_files()
	print("-- Done!")
	print("Time taken to preprocess data: {}".format(time.time() - t_start))


if __name__ == '__main__':
	main()
