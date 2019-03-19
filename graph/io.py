# coding: utf-8

# Author: Moju Wu
#
# IO for relation.json

import json
import random
from core import keras_models


def load_relation_from_files(json_files, val_portion=0.0, test_portion=0.0):
	"""
	Load relation and argument1 and argument2 from multiple json files.

	:param json_files: list of input json files
	:param val_portion: a portion of the data to reserve for validation
	:return: a tuple of the data and validation data
	"""
	data = []
	cleaned_data = []
	for json_file in json_files:
		with open(json_file) as f:
			data += json.load(f)
	print("Loaded data size:", len(data))

	for g in data:
		# if len(g["Tokens"]) > 200:
		#     data.remove(g)
		if len(g["Tokens"]) <= keras_models.model_params['max_sent_len']:
			cleaned_data.append(g)
	print("Original data size:", len(data))
	print("Cleaned data size:", len(cleaned_data))

	for g in cleaned_data:
		assert len(g["Tokens"]) <= keras_models.model_params['max_sent_len']

	val_size = int(len(cleaned_data) * val_portion)
	test_size = int(len(cleaned_data) * test_portion)
	train_size = len(cleaned_data) - val_size - test_size
	random.shuffle(cleaned_data)

	train_data = cleaned_data[:train_size]
	val_data = cleaned_data[train_size:(train_size+val_size)]
	test_data = cleaned_data[(train_size+val_size):]

	print("Training dev and test set sizes:", (len(train_data), len(val_data), len(test_data)))

	return train_data, val_data, test_data


def load_relation_from_file(json_file, val_portion=0.0, test_portion=0.0):
	"""
	Load semantic graphs from a json file and if specified reserve a portion of the data for validation.

	:param json_file: the input json file
	:param val_portion: a portion of the data to reserve for validation
	:return: a tuple of the data and validation data
	"""
	return load_relation_from_files([json_file], val_portion, test_portion)


def load_relation_graphs_from_files(json_files, val_portion=0.0, load_vertices=True):
	"""
	Load semantic graphs from multiple json files and if specified reserve a portion of the data for validation.

	:param json_files: list of input json files
	:param val_portion: a portion of the data to reserve for validation
	:return: a tuple of the data and validation data
	"""
	data = []
	for json_file in json_files:
		with open(json_file) as f:
			if load_vertices:
				data = data + json.load(f)
			else:
				data = data + json.load(f, object_hook=dict_to_graph_with_no_vertices)
	print("Loaded data size:", len(data))

	val_data = []
	if val_portion > 0.0:
		val_size = int(len(data) * val_portion)
		rest_size = len(data) - val_size
		val_data = data[rest_size:]
		data = data[:rest_size]
		print("Training and dev set sizes:", (len(data), len(val_data)))
	return data, val_data


def load_relation_graphs_from_file(json_file, val_portion=0.0, load_vertices=True):
	"""
	Load semantic graphs from a json file and if specified reserve a portion of the data for validation.

	:param json_file: the input json file
	:param val_portion: a portion of the data to reserve for validation
	:return: a tuple of the data and validation data
	"""
	return load_relation_graphs_from_files([json_file], val_portion, load_vertices)


def read_relations_from_file(json_file):
	data = []
	with open(json_file) as f:
		data += json.load(f)
	return data


def load_relation_from_existing_sets(relationMention_ph):

	train_data = read_relations_from_file(relationMention_ph+'train.relationMention.json')
	val_data = read_relations_from_file(relationMention_ph+'val.relationMention.json')
	test_data = read_relations_from_file(relationMention_ph+'test.relationMention.json')

	print("Train set size:", len(train_data))
	print("Val set size:", len(val_data))
	print("Test set size:", len(test_data))
	return train_data, val_data, test_data


def dict_to_graph_with_no_vertices(d):
	if 'vertexSet' in d:
		del d['vertexSet']
	return d
