"""
Date: 2019/2/13
Version: 0
Last update: 2019/3/13
Author: Moju Wu
"""

import json

from sklearn.metrics import confusion_matrix
from keras import callbacks
from keras.utils import np_utils
import numpy as np
import hyperopt as hy
import matplotlib.pyplot as plt
from evaluation import metrics
from collections import OrderedDict

from core import embeddings, keras_models
from graph import io


def f_train(params):
	model = getattr(keras_models, model_name)(params, embedding_matrix, max_sent_len, n_out)
	callback_history = model.fit(train_as_indices[:-1],
								 [train_y_properties_one_hot],
								 epochs=20, batch_size=keras_models.model_params['batch_size'], verbose=1,
								 validation_data=(
									 val_as_indices[:-1], val_y_properties_one_hot),
								 callbacks=[callbacks.EarlyStopping(monitor="val_loss", patience=1, verbose=1)])

	predictions = model.predict(val_as_indices[:-1], batch_size=16, verbose=1)
	predictions_classes = np.argmax(predictions, axis=1)
	_, _, acc = metrics.compute_micro_PRF(predictions_classes, val_as_indices[-1])
	return {'loss': -acc, 'status': hy.STATUS_OK}


def evaluate(model, data_input, gold_output):
	predictions = model.predict(data_input, batch_size=keras_models.model_params['batch_size'], verbose=1)
	if len(predictions.shape) == 3:
		predictions_classes = np.argmax(predictions, axis=2)
		train_batch_f1 = metrics.accuracy_per_sentence(predictions_classes, gold_output)
		print("Results (per sentence): ", train_batch_f1)
		train_y_properties_stream = gold_output.reshape(gold_output.shape[0] * gold_output.shape[1])
		predictions_classes = predictions_classes.reshape(predictions_classes.shape[0] * predictions_classes.shape[1])
		class_mask = train_y_properties_stream != 0
		train_y_properties_stream = train_y_properties_stream[class_mask]
		predictions_classes = predictions_classes[class_mask]
	else:
		predictions_classes = np.argmax(predictions, axis=1)
		train_y_properties_stream = gold_output
	accuracy = metrics.accuracy(predictions_classes, train_y_properties_stream)
	micro_scores = metrics.compute_micro_PRF(predictions_classes, train_y_properties_stream,
											 empty_label=keras_models.p0_index)
	print("Results: Accuracy: ", accuracy)
	print("Results: Micro-Average F1: ", micro_scores)
	return predictions_classes, predictions


def error_analysis(model, data_inputs, gold_outputs, out_folder, val_data):
	predictions = model.predict(data_inputs, batch_size=keras_models.model_params['batch_size'], verbose=1)

	predictions_classes = np.argmax(predictions, axis=1)
	true_classes = np.argmax(gold_outputs, axis=1)

	predictions_classes = [keras_models.idx2property.get(i) for i in predictions_classes]
	true_classes = [keras_models.idx2property.get(i) for i in true_classes]

	with open(out_folder+'error.txt', 'w') as f:
		for index, yhat in enumerate(predictions_classes):
			if yhat != true_classes[index]:
				f.write("relationID: " + val_data[index].get('relationID') + '\n')
				f.write("Sentence: " + val_data[index].get('Sentence') + '\n')
				f.write("mentionArg1: " + val_data[index].get('mentionArg1').get('extent') + '\n')
				f.write("mentionArg2: " + val_data[index].get('mentionArg2').get('extent') + '\n')
				f.write("predict: " + yhat + '\n')
				f.write("true: " + true_classes[index] + '\n')
				f.write('\n')

	labels = ['PHYS', 'PART-WHOLE', 'PER-SOC', 'ORG-AFF', 'ART', 'GEN-AFF', 'METONYMY']
	cm = confusion_matrix(true_classes, predictions_classes, labels)
	print(cm)
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(cm)
	plt.title('Confusion matrix of the relation classifier')
	fig.colorbar(cax)
	ax.set_xticklabels([''] + labels)
	ax.set_yticklabels([''] + labels)
	plt.xlabel('Predicted')
	plt.ylabel('True')

	plt.savefig(out_folder + 'confusion_matrix.png')
	plt.clf()


if __name__ == '__main__':
	import argparse
	import glob
	from graph import graph_utils

	DEBUG = 0

	parser = argparse.ArgumentParser()
	parser.add_argument('model_name')
	parser.add_argument('mode', choices=['train', 'optimize', 'train-continue', 'eval', 'summary', 'analysis'])
	parser.add_argument('train_set')
	parser.add_argument('--word_embedding', default='../resource/embeddings/glove/glove.6B.50d.txt')
	# parser.add_argument('val_set')
	parser.add_argument('--models_folder', default="./trainedmodels/")
	parser.add_argument('--earlystop', default=False)
	parser.add_argument('--epoch', default=50, type=int)
	parser.add_argument('--checkpoint', default=False)
	parser.add_argument('--tensorboard', default=False)
	parser.add_argument('--metadata', type=str)
	parser.add_argument('--error_out_folder', default='./error_output/')

	args = parser.parse_args()

	model_name = args.model_name
	mode = args.mode
	error_out_folder = args.error_out_folder

	embedding_matrix, word2idx = embeddings.load(args.word_embedding)
	print("embedding_matrix: " + str(embedding_matrix.shape))

	relationMention_files = glob.glob(args.train_set)
	train_data, val_data, test_data = io.load_relation_from_files(relationMention_files, val_portion=0.1, test_portion=0.1)
	print("Document number: {}".format(len(relationMention_files)))

	if DEBUG == 1:

		print("Training data size: {}".format(len(train_data)))
		print("Validation data size: {}".format(len(val_data)))
		print("Testing data size: {}".format(len(val_data)))

		max_sent_len, max_graph = graph_utils.get_max_sentence_len(train_data)
		print(max_sent_len)
		print(max_graph["id"])

	max_sent_len = keras_models.model_params['max_sent_len']  # 200
	print("Max sentence length set to: {}".format(max_sent_len))

	to_one_hot = np_utils.to_categorical
	graphs_to_indices = keras_models.to_indices

	if "LSTMbaseline" in model_name:
		graphs_to_indices = keras_models.to_indices_with_extracted_entities

	elif "Context" in model_name:
		to_one_hot = embeddings.timedistributed_to_one_hot
		graphs_to_indices = keras_models.to_indices_with_extracted_entities
	elif "CNN" in model_name:
		graphs_to_indices = keras_models.to_indices_with_relative_positions

	train_as_indices = list(graphs_to_indices(train_data, word2idx))
	print("Dataset shapes: {}".format([d.shape for d in train_as_indices]))

	train_data = None

	n_out = len(keras_models.property2idx)  # n_out = number of relation categories
	print("N_out:", n_out)

	val_as_indices = list(graphs_to_indices(val_data, word2idx))
	# val_data = None

	test_as_indices = list(graphs_to_indices(test_data, word2idx))
	test_data = None

	if "train" in mode:
		print("Training the model")
		print("Initialize the model")
		model = getattr(keras_models, model_name)(keras_models.model_params, embedding_matrix, max_sent_len, n_out)
		if "continue" in mode:
			print("Load pre-trained weights")
			model.load_weights(args.models_folder + model_name + ".kerasmodel")

		# sentences_matrix, arg1_matrix, arg2_matrix, y_matrix
		train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)

		val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

		test_y_properties_one_hot = to_one_hot(test_as_indices[-1], n_out)

		cbfunctions = []
		if args.earlystop:
			earlystop = callbacks.EarlyStopping(monitor="val_loss", patience=5, verbose=1)
			cbfunctions.append(earlystop)
		if args.checkpoint:
			checkpoint = callbacks.ModelCheckpoint(
				args.models_folder + model_name + '-{epoch:02d}-{val_loss:.2f}' + ".kerasmodel",
				monitor='val_loss', verbose=1, save_best_only=True)
			cbfunctions.append(checkpoint)
		if args.tensorboard:
			tensorboard = callbacks.TensorBoard(log_dir=args.models_folder + "logs", histogram_freq=True, write_graph=True,
												write_images=False)
			cbfunctions.append(tensorboard)
		callback_history = model.fit(train_as_indices[:-1],
									[train_y_properties_one_hot],
									epochs=args.epoch,
									batch_size=keras_models.model_params['batch_size'],
									verbose=1,
									validation_data=(val_as_indices[:-1], val_y_properties_one_hot),
									callbacks=cbfunctions)

		# Plot training & validation accuracy values
		plt.plot(callback_history.history['acc'])
		plt.plot(callback_history.history['val_acc'])
		plt.title('Model accuracy')
		plt.ylabel('Accuracy')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(args.models_folder + 'accuracy.png')
		plt.clf()
		# plt.show()

		# Plot training & validation loss values
		plt.plot(callback_history.history['loss'])
		plt.plot(callback_history.history['val_loss'])
		plt.title('Model loss')
		plt.ylabel('Loss')
		plt.xlabel('Epoch')
		plt.legend(['Train', 'Validation'], loc='upper left')
		plt.savefig(args.models_folder + 'loss.png')
		# plt.show()
		plt.clf()

		score = model.evaluate(train_as_indices[:-1], train_y_properties_one_hot)
		print("Results on the training set:", score[0], score[1])
		score = model.evaluate(val_as_indices[:-1], val_y_properties_one_hot)
		print("Results on the validation set: ", score[0], score[1])
		score = model.evaluate(test_as_indices[:-1], test_y_properties_one_hot)
		print("Results on the testing set: ", score[0], score[1])

	elif mode == "optimize":
		import optimization_space

		space = optimization_space.space

		train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)
		val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

		trials = hy.Trials()
		best = hy.fmin(f_train, space, algo=hy.rand.suggest, max_evals=10, trials=trials)
		print("Best trial:", best)
		print("Details:", trials.best_trial)
		print("Saving trials.")
		with open("../data/trials/" + model_name + "_final_trials.json", 'w') as ftf:
			json.dump([(t['misc']['vals'], t['result']) for t in trials.trials], ftf)

	elif mode == "eval":

		print("Loading the best model")
		model = getattr(keras_models, model_name)(keras_models.model_params, embedding_matrix, max_sent_len, n_out)
		model.load_weights(args.models_folder + model_name + "-" + args.metadata + ".kerasmodel")

		# print("Results on the training set")
		# evaluate(model, train_as_indices[:-1], train_as_indices[-1])
		# print("Results on the validation set")
		# evaluate(model, val_as_indices[:-1], val_as_indices[-1])
		# print("Results on the test set")
		# evaluate(model, test_as_indices[:-1], test_as_indices[-1])

		# sentences_matrix, arg1_matrix, arg2_matrix, y_matrix
		train_y_properties_one_hot = to_one_hot(train_as_indices[-1], n_out)

		val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

		test_y_properties_one_hot = to_one_hot(test_as_indices[-1], n_out)

		score = model.evaluate(train_as_indices[:-1], train_y_properties_one_hot)
		print("Results on the training set:", score[0], score[1])
		score = model.evaluate(val_as_indices[:-1], val_y_properties_one_hot)
		print("Results on the validation set: ", score[0], score[1])
		score = model.evaluate(test_as_indices[:-1], test_y_properties_one_hot)
		print("Results on the testing set: ", score[0], score[1])

	elif mode == "summary":
		model = getattr(keras_models, model_name)(keras_models.model_params, embedding_matrix, max_sent_len, n_out)
		print(model.summary())

	elif mode == 'analysis':

		print("Loading the best model")
		model = getattr(keras_models, model_name)(keras_models.model_params, embedding_matrix, max_sent_len, n_out)
		model.load_weights(args.models_folder + model_name + "-" + args.metadata + ".kerasmodel")

		val_y_properties_one_hot = to_one_hot(val_as_indices[-1], n_out)

		error_analysis(model, val_as_indices[:-1], val_y_properties_one_hot, error_out_folder, val_data)

