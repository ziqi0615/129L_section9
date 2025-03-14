#!/usr/bin/env python

from os.path import isdir, join as path_join, isfile
from struct import unpack
from array import array as array_unpack

from numpy import array, zeros
from numpy.random import choice, seed as srand
from matplotlib import pyplot as plt

from mlp import MLP
from cnn import CNN
from my_lib import labels_to_one_hot, relu, cross_entropy, softmax, sigmoid, tqdm

mnist_data_dir = 'mnist_data'
test_images_path = path_join(mnist_data_dir, 't10k-images.idx3-ubyte')
test_labels_path = path_join(mnist_data_dir, 't10k-labels.idx1-ubyte')
train_images_path = path_join(mnist_data_dir, 'train-images.idx3-ubyte')
train_labels_path = path_join(mnist_data_dir, 'train-labels.idx1-ubyte')

if not isdir(mnist_data_dir):
	from os import mkdir, remove, listdir
	from shutil import rmtree
	from urllib.request import urlretrieve
	from subprocess import run

	print('Downloading MNIST data...')
	mkdir(mnist_data_dir)
	download_path = 'mnist-dataset.zip'
	urlretrieve('https://www.kaggle.com/api/v1/datasets/download/hojjatk/mnist-dataset', download_path)
	run(['unzip', download_path, '-d', mnist_data_dir])
	remove(download_path)
	for filename in listdir(mnist_data_dir):
		path = path_join(mnist_data_dir, filename)
		if isdir(path):
			rmtree(path)
	print('Download complete.')

def read_binary(path, magic, head_size):
	with open(path, 'rb') as file:
		magic_got, *head = unpack('>'+'I'*(head_size+1), file.read(4*(head_size+1)))
		if magic_got != magic:
			raise ValueError(f'Magic number mismatch, expected {magic}, got {magic_got}')
		data = array_unpack('B', file.read())
	return *head, data
def read_images_labels(images_filepath, labels_filepath):
	size, labels = read_binary(labels_filepath, 2049, 1)
	size, rows, cols, image_data = read_binary(images_filepath, 2051, 3)
	return array(image_data).reshape(size, rows, cols), array(labels)
train_images, train_labels = read_images_labels(train_images_path, train_labels_path)
test_images, test_labels = read_images_labels(test_images_path, test_labels_path)

srand(1108)
train_index = choice(len(train_images), 1024, replace=False)
train_images = train_images[train_index]
train_labels = labels_to_one_hot(train_labels[train_index], 10)

def display_confusion(model):
	confusion = zeros((10, 10), dtype=int)
	for image, label in tqdm(zip(test_images, test_labels), total=len(test_images)):
		confusion[label, model.predict(image).argmax()] += 1
	fig, ax = plt.subplots()
	im = ax.imshow(confusion)
	ax.set_xticks(range(10))
	ax.set_yticks(range(10))
	for i in range(10):
		for j in range(10):
			ax.text(j, i, confusion[i, j], ha='center', va='center', color='w')
	ax.set_xlabel('Predicted')
	ax.set_ylabel('Actual')
	plt.show()

if __name__ == '__main__':
	cnn = CNN(
		input_shape=train_images[0].shape,
		filter_size=6,
		num_filters=3,
		pool_size=2,
		hidden_layer_size=60,
		output_size=10
	)
	cnn_weights_path = 'cnn_weights.npz'
	if isfile(cnn_weights_path):
		cnn.load(cnn_weights_path)
	else:
		losses = cnn.train(train_images, train_labels, [0.2]*256)
		cnn.save(cnn_weights_path)
		plt.plot(losses)
		plt.show()

	display_confusion(cnn)

	mlp = MLP(
		layer_sizes=[train_images[0].size, 700, 500, 10],
		activations=[relu, sigmoid, softmax],
		loss=cross_entropy
	)
	mlp_weights_path = 'mlp_weights.npz'
	if isfile(mlp_weights_path):
		mlp.load(mlp_weights_path)
	else:
		losses = mlp.train(train_images, train_labels, [0.2]*256, 64)
		mlp.save(mlp_weights_path)
		plt.plot(losses)
		plt.show()

	display_confusion(mlp)