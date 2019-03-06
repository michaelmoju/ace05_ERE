# coding: utf-8
# Copyright (C) 2016 UKP lab
#
# Author: Daniil Sorokin (ukp.tu-darmstadt.de/ukp-home/)
#
import os
import ast, json
import numpy as np
import tensorflow as tf

np.random.seed(1)

from keras import layers, models, optimizers
from keras import backend as K
from keras import regularizers
import tqdm

from core import embeddings
from graph import graph_utils

module_location = os.path.abspath(__file__)
module_location = os.path.dirname(module_location)

with open(os.path.join(module_location, "../model_params.json")) as f:
	model_params = json.load(f)

with open(os.path.join(module_location, "../structure/", model_params["property2idx"])) as f:
	property2idx = ast.literal_eval(f.read())

idx2property = {v: k for k, v in property2idx.items()}

_, position2idx = embeddings.init_random(np.arange(-model_params['max_sent_len'], model_params['max_sent_len']),
										 1, add_all_zeroes=True)

p0_index = 1
MAX_EDGES_PER_GRAPH = 7
POSITION_EMBEDDING_MODE = "mark-bi"
POSITION_VOCAB_SIZE = 3


def model_LSTMbaseline_without_mask(p, embedding_matrix, max_sent_len, n_out):
	print("Parameters:", p)
	# Take sentence encoded as indices and convert it to embeddings
	sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(output_dim=embedding_matrix.shape[1],
									   input_dim=embedding_matrix.shape[0],
									   input_length=max_sent_len, weights=[embedding_matrix],
									   mask_zero=False, trainable=False)(sentence_input)

	word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

	# Take arg1_markers that identify entity positions, convert to position embeddings
	arg1_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='arg1_markers')
	arg1_pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=POSITION_VOCAB_SIZE,
									  input_length=max_sent_len,
									  mask_zero=False,
									  embeddings_regularizer=regularizers.l2(), trainable=True)(arg1_markers)

	# Take arg2_markers that identify entity positions, convert to position embeddings
	arg2_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='arg2_markers')
	arg2_pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=POSITION_VOCAB_SIZE,
									  input_length=max_sent_len,
									  mask_zero=False,
									  embeddings_regularizer=regularizers.l2(), trainable=True)(arg2_markers)

	# Merge word and position embeddings and apply the specified amount of RNN layers
	x = layers.concatenate([word_embeddings, arg1_pos_embeddings, arg2_pos_embeddings])

	for i in range(p["rnn1_layers"] - 1):
		lstm_layer = layers.LSTM(p['units1'], return_sequences=True)
		if p['bidirectional']:
			lstm_layer = layers.Bidirectional(lstm_layer)
		x = lstm_layer(x)

	lstm_layer = layers.LSTM(p['units1'], return_sequences=False)

	if p['bidirectional']:
		lstm_layer = layers.Bidirectional(lstm_layer)

	sentence_vector = lstm_layer(x)

	# Apply softmax
	sentence_vector = layers.Dropout(p['dropout1'])(sentence_vector)
	main_output = layers.Dense(n_out, activation="softmax", name='main_output')(sentence_vector)

	model = models.Model(inputs=[sentence_input, arg1_markers, arg2_markers], outputs=[main_output])
	model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def model_LSTMbaseline(p, embedding_matrix, max_sent_len, n_out):
	print("Parameters:", p)
	# Take sentence encoded as indices and convert it to embeddings
	sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(output_dim=embedding_matrix.shape[1],
									   input_dim=embedding_matrix.shape[0],
									   input_length=max_sent_len, weights=[embedding_matrix],
									   mask_zero=True, trainable=False)(sentence_input)



	word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

	# Take arg1_markers that identify entity positions, convert to position embeddings
	arg1_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='arg1_markers')
	arg1_pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=POSITION_VOCAB_SIZE,
									  input_length=max_sent_len,
									  mask_zero=True,
									  embeddings_regularizer=regularizers.l2(), trainable=True)(arg1_markers)

	# Take arg2_markers that identify entity positions, convert to position embeddings
	arg2_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='arg2_markers')
	arg2_pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=POSITION_VOCAB_SIZE,
									  input_length=max_sent_len,
									  mask_zero=True,
									  embeddings_regularizer=regularizers.l2(), trainable=True)(arg2_markers)

	# Merge word and position embeddings and apply the specified amount of RNN layers
	x = layers.concatenate([word_embeddings, arg1_pos_embeddings, arg2_pos_embeddings])


	for i in range(p["rnn1_layers"] - 1):
		lstm_layer = layers.LSTM(p['units1'], return_sequences=True)
		if p['bidirectional']:
			lstm_layer = layers.Bidirectional(lstm_layer)
		x = lstm_layer(x)

	lstm_layer = layers.LSTM(p['units1'], return_sequences=False)

	if p['bidirectional']:
		lstm_layer = layers.Bidirectional(lstm_layer)

	sentence_vector = lstm_layer(x)

	# Apply softmax
	sentence_vector = layers.Dropout(p['dropout1'])(sentence_vector)
	main_output = layers.Dense(n_out, activation="softmax", name='main_output')(sentence_vector)

	model = models.Model(inputs=[sentence_input, arg1_markers, arg2_markers], outputs=[main_output])
	model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def model_CNN(p, embedding_matrix, max_sent_len, n_out):
	print("Parameters:", p)
	# Take sentence encoded as indices split in three parts and convert it to embeddings
	sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(output_dim=embedding_matrix.shape[1],
									   input_dim=embedding_matrix.shape[0],
									   input_length=max_sent_len, weights=[embedding_matrix],
									   mask_zero=True, trainable=False)(sentence_input)
	word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)


	# Take token markers that identify entity positions, convert to position embeddings
	entity_markers = layers.Input(shape=(2, max_sent_len,), dtype='int8', name='entity_markers')

	pos_embeddings = layers.wrappers.TimeDistributed(
		layers.Embedding(output_dim=p['position_emb'], input_dim=(max_sent_len * 2) + 1, input_length=max_sent_len,
						 mask_zero=False, embeddings_regularizer=regularizers.l2(), trainable=True),
		name='pos_embedding')(entity_markers)

	pos_embeddings = layers.Permute((2, 1, 3))(pos_embeddings)
	pos_embeddings = layers.Reshape((max_sent_len, p['position_emb'] * 2))(pos_embeddings)

	# Merge word and position embeddings and apply the specified amount of CNN layers
	x = layers.concatenate([word_embeddings, pos_embeddings])

	x = MaskedConvolution1D(nb_filter=p['units1'], filter_length=p['window_size'], border_mode='same')(x)
	sentence_vector = MaskedGlobalMaxPooling1D()(x)

	sentence_vector = layers.Lambda(lambda l: K.tanh(l))(sentence_vector)

	# Apply softmax
	sentence_vector = layers.Dropout(p['dropout1'])(sentence_vector)
	main_output = layers.Dense(n_out, activation="softmax", name='main_output')(sentence_vector)

	model = models.Model(input=[sentence_input, entity_markers], output=[main_output])
	model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

	return model


def model_multitask_LSTM(p, embedding_matrix, max_sent_len, n_out):
	print("Parameters:", p)
	# Take sentence encoded as indices and convert it to embeddings
	sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
	word_embeddings = layers.Embedding(output_dim=embedding_matrix.shape[1],
									   input_dim=embedding_matrix.shape[0],
									   input_length=max_sent_len, weights=[embedding_matrix],
									   mask_zero=True, trainable=False)(sentence_input)

	word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

	# Take arg1_markers that identify entity positions, convert to position embeddings
	arg1_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='arg1_markers')
	arg1_pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=POSITION_VOCAB_SIZE,
										   input_length=max_sent_len,
										   mask_zero=True,
										   embeddings_regularizer=regularizers.l2(), trainable=True)(arg1_markers)

	# Take arg2_markers that identify entity positions, convert to position embeddings
	arg2_markers = layers.Input(shape=(max_sent_len,), dtype='int8', name='arg2_markers')
	arg2_pos_embeddings = layers.Embedding(output_dim=p['position_emb'], input_dim=POSITION_VOCAB_SIZE,
										   input_length=max_sent_len,
										   mask_zero=True,
										   embeddings_regularizer=regularizers.l2(), trainable=True)(arg2_markers)

	# Merge word and position embeddings and apply the specified amount of RNN layers
	x = layers.concatenate([word_embeddings, arg1_pos_embeddings, arg2_pos_embeddings])

	for i in range(p["rnn1_layers"] - 1):
		lstm_layer = layers.LSTM(p['units1'], return_sequences=True)
		if p['bidirectional']:
			lstm_layer = layers.Bidirectional(lstm_layer)
		x = lstm_layer(x)

	lstm_layer = layers.LSTM(p['units1'], return_sequences=False)
	entity_lstm_layer = layers.LSTM(p['units1'], return_sequences=True)

	if p['bidirectional']:
		lstm_layer = layers.Bidirectional(lstm_layer)
		entity_lstm_layer = layers.Bidirectional(entity_lstm_layer)

	entity_vector = entity_lstm_layer(x)
	sentence_vector = lstm_layer(x)

	# Apply softmax
	entity_vector = layers.Dropout(p['dropout1'])(entity_vector)
	sentence_vector = layers.Dropout(p['dropout1'])(sentence_vector)
	entity_output = layers.Dense(n_out, activation="softmax", name='main_output')(entity_vector)
	relation_output = layers.Dense(n_out, activation="softmax", name='main_output')(sentence_vector)

	model = models.Model(inputs=[sentence_input], outputs=[entity_output, relation_output])
	model.compile(optimizer=p['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

	return model

def masked_categorical_crossentropy(y_true, y_pred):
	mask = K.equal(y_true[..., 0], K.variable(1))
	mask = 1 - K.cast(mask, K.floatx())

	loss = K.categorical_crossentropy(y_true, y_pred) * mask
	return loss


def model_ContextSum(p, embedding_matrix, max_sent_len, n_out):
	print("Parameters:", p)

	# Take sentence encoded as indices and convert it to embeddings
	sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
	# Repeat the input N times for each edge
	x = layers.RepeatVector(MAX_EDGES_PER_GRAPH)(sentence_input)
	word_embeddings = layers.wrappers.TimeDistributed(
		layers.Embedding(output_dim=embedding_matrix.shape[1], input_dim=embedding_matrix.shape[0],
						 input_length=max_sent_len, weights=[embeddings],
						 mask_zero=True, trainable=False))(x)
	word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

	# Take token markers that identify entity positions, convert to position embeddings
	entity_markers = layers.Input(shape=(MAX_EDGES_PER_GRAPH, max_sent_len,), dtype='int8', name='entity_markers')
	pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
																	  input_dim=POSITION_VOCAB_SIZE,
																	  input_length=max_sent_len,
																	  mask_zero=True,
																	  embeddings_regularizer=regularizers.l2(),
																	  trainable=True))(entity_markers)

	# Merge word and position embeddings and apply the specified amount of RNN layers
	for i in range(p["rnn1_layers"] - 1):
		lstm_layer = layers.LSTM(p['units1'], return_sequences=True)
		if p['bidirectional']:
			lstm_layer = layers.Bidirectional(lstm_layer)
		x = layers.wrappers.TimeDistributed(lstm_layer)(x)
	lstm_layer = layers.LSTM(p['units1'], return_sequences=False)
	if p['bidirectional']:
		lstm_layer = layers.Bidirectional(lstm_layer)
	sentence_matrix = layers.wrappers.TimeDistributed(lstm_layer)(x)

	# Take the vector of the sentences with the target entity pair
	layers_to_concat = []
	num_units = p['units1'] * (2 if p['bidirectional'] else 1)
	for i in range(MAX_EDGES_PER_GRAPH):
		sentence_vector = layers.Lambda(lambda l: l[:, i], output_shape=(num_units,))(sentence_matrix)
		if i == 0:
			context_vectors = layers.Lambda(lambda l: l[:, i + 1:], output_shape=(MAX_EDGES_PER_GRAPH - 1, num_units))(
				sentence_matrix)
		elif i == MAX_EDGES_PER_GRAPH - 1:
			context_vectors = layers.Lambda(lambda l: l[:, :i], output_shape=(MAX_EDGES_PER_GRAPH - 1, num_units))(
				sentence_matrix)
		else:
			context_vectors = layers.Lambda(lambda l: K.concatenate([l[:, :i], l[:, i + 1:]], axis=1),
											output_shape=(MAX_EDGES_PER_GRAPH - 1, num_units))(sentence_matrix)
		context_vector = GlobalSumPooling1D()(context_vectors)
		edge_vector = layers.concatenate([sentence_vector, context_vector])
		edge_vector = layers.Reshape((1, num_units * 2))(edge_vector)
		layers_to_concat.append(edge_vector)
	edge_vectors = layers.Concatenate(1)(layers_to_concat)

	# Apply softmax
	edge_vectors = layers.Dropout(p['dropout1'])(edge_vectors)
	main_output = layers.wrappers.TimeDistributed(layers.Dense(n_out, activation="softmax", name='main_output'))(
		edge_vectors)

	model = models.Model(inputs=[sentence_input, entity_markers], outputs=[main_output])
	model.compile(optimizer=p['optimizer'], loss=masked_categorical_crossentropy, metrics=['accuracy'])

	return model


def model_ContextWeighted(p, embedding_matrix, max_sent_len, n_out):
	print("Parameters:", p)

	# Take sentence encoded as indices and convert it to embeddings
	sentence_input = layers.Input(shape=(max_sent_len,), dtype='int32', name='sentence_input')
	# Repeat the input N times for each edge
	x = layers.RepeatVector(MAX_EDGES_PER_GRAPH)(sentence_input)
	word_embeddings = layers.wrappers.TimeDistributed(
		layers.Embedding(output_dim=embedding_matrix.shape[1], input_dim=embedding_matrix.shape[0],
						 input_length=max_sent_len, weights=[embedding_matrix],
						 mask_zero=True, trainable=False))(x)
	word_embeddings = layers.Dropout(p['dropout1'])(word_embeddings)

	# Take token markers that identify entity positions, convert to position embeddings
	entity_markers = layers.Input(shape=(MAX_EDGES_PER_GRAPH, max_sent_len,), dtype='int8', name='entity_markers')
	pos_embeddings = layers.wrappers.TimeDistributed(layers.Embedding(output_dim=p['position_emb'],
																	  input_dim=POSITION_VOCAB_SIZE,
																	  input_length=max_sent_len,
																	  mask_zero=True,
																	  embeddings_regularizer=regularizers.l2(),
																	  trainable=True))(entity_markers)

	# Merge word and position embeddings and apply the specified amount of RNN layers
	x = layers.concatenate([word_embeddings, pos_embeddings])
	for i in range(p["rnn1_layers"] - 1):
		lstm_layer = layers.LSTM(p['units1'], return_sequences=True)
		if p['bidirectional']:
			lstm_layer = layers.Bidirectional(lstm_layer)
		x = layers.wrappers.TimeDistributed(lstm_layer)(x)
	lstm_layer = layers.LSTM(p['units1'], return_sequences=False)
	if p['bidirectional']:
		lstm_layer = layers.Bidirectional(lstm_layer)
	sentence_matrix = layers.wrappers.TimeDistributed(lstm_layer)(x)

	### Attention over ghosts ###
	layers_to_concat = []
	num_units = p['units1'] * (2 if p['bidirectional'] else 1)
	for i in range(MAX_EDGES_PER_GRAPH):
		# Compute a memory vector for the target entity pair
		sentence_vector = layers.Lambda(lambda l: l[:, i], output_shape=(num_units,))(sentence_matrix)
		target_sentence_memory = layers.Dense(num_units,
											  activation="linear", use_bias=False)(sentence_vector)
		if i == 0:
			context_vectors = layers.Lambda(lambda l: l[:, i + 1:],
											output_shape=(MAX_EDGES_PER_GRAPH - 1, num_units))(sentence_matrix)
		elif i == MAX_EDGES_PER_GRAPH - 1:
			context_vectors = layers.Lambda(lambda l: l[:, :i],
											output_shape=(MAX_EDGES_PER_GRAPH - 1, num_units))(sentence_matrix)
		else:
			context_vectors = layers.Lambda(lambda l: K.concatenate([l[:, :i], l[:, i + 1:]], axis=1),
											output_shape=(MAX_EDGES_PER_GRAPH - 1, num_units))(sentence_matrix)
		# Compute the score between each memory and the memory of the target entity pair
		sentence_scores = layers.Lambda(lambda inputs: K.batch_dot(inputs[0],
																   inputs[1], axes=(1, 2)),
										output_shape=(MAX_EDGES_PER_GRAPH,))([target_sentence_memory, context_vectors])
		sentence_scores = layers.Activation('softmax')(sentence_scores)

		# Compute the final vector by taking the weighted sum of context vectors and the target entity vector
		context_vector = layers.Lambda(lambda inputs: K.batch_dot(inputs[0], inputs[1], axes=(1, 1)),
									   output_shape=(num_units,))([context_vectors, sentence_scores])
		edge_vector = layers.concatenate([sentence_vector, context_vector])
		edge_vector = layers.Reshape((1, num_units * 2))(edge_vector)
		layers_to_concat.append(edge_vector)

	edge_vectors = layers.concatenate(layers_to_concat, axis=1)

	# Apply softmax
	edge_vectors = layers.Dropout(p['dropout1'])(edge_vectors)
	main_output = layers.wrappers.TimeDistributed(layers.Dense(n_out, activation="softmax", name='main_output'))(
		edge_vectors)

	model = models.Model(inputs=[sentence_input, entity_markers], outputs=[main_output])
	optimizer = optimizers.Adam(lr=0.001)
	model.compile(optimizer=optimizer, loss=masked_categorical_crossentropy, metrics=['accuracy'])

	return model


class GlobalSumPooling1D(layers.Layer):

	def __init__(self, **kwargs):
		super(GlobalSumPooling1D, self).__init__(**kwargs)
		self.input_spec = [layers.InputSpec(ndim=3)]

	def compute_output_shape(self, input_shape):
		return input_shape[0], input_shape[2]

	def call(self, x, mask=None):
		return K.sum(x, axis=1)


class MaskedConvolution1D(layers.Convolution1D):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(MaskedConvolution1D, self).__init__(**kwargs)

	def compute_mask(self, x, mask=None):
		return mask


class MaskedGlobalMaxPooling1D(layers.pooling._GlobalPooling1D):
	def __init__(self, **kwargs):
		self.supports_masking = True
		super(MaskedGlobalMaxPooling1D, self).__init__(**kwargs)

	def call(self, x, mask=None):
		if mask is None:
			return K.max(x, axis=1)
		else:
			if (K.backend() == 'tensorflow'):
				import tensorflow as tf
				return K.max(tf.where(mask[:, :, np.newaxis], x, -np.inf), axis=1)
			else:
				print("theano")
				return K.max(K.switch(mask[:, :, np.newaxis], x, -np.inf), axis=1)

	def compute_mask(self, x, mask=None):
		return None


def to_indices(graphs, word2idx):
	max_sent_len = model_params['max_sent_len']
	num_edges = sum(1 for g in graphs for e in g['edgeSet'])
	sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
	entity_matrix = np.zeros((num_edges, max_sent_len), dtype="int8")
	y_matrix = np.zeros(num_edges, dtype="int16")
	index = 0
	for g in tqdm.tqdm(graphs, ascii=True):
		token_sent_ids = embeddings.get_idx_sequence(g["tokens"], word2idx)
		if len(token_sent_ids) > max_sent_len:
			token_sent_ids = token_sent_ids[:max_sent_len]
		for edge in g["edgeSet"]:
			if edge['kbID'] not in property_blacklist:
				left_border, right_border = graph_utils.get_sentence_boundaries(g["tokens"], edge)
				entity_markers = [m for _, m in graph_utils.get_entity_indexed_vector(g["tokens"], edge,
																					  mode=POSITION_EMBEDDING_MODE)][
								 left_border:right_border]
				token_sent_ids = token_sent_ids[left_border:right_border]
				sentences_matrix[index, :len(token_sent_ids)] = token_sent_ids
				entity_matrix[index, :len(token_sent_ids)] = entity_markers[:len(token_sent_ids)]
				_, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
				property_kbid = property2idx.get(property_kbid, property2idx[embeddings.all_zeroes])
				y_matrix[index] = property_kbid
				index += 1
	return [sentences_matrix, entity_matrix, y_matrix]


def to_indices_with_extracted_entities(graphs, word2idx):
	"""
	:param graphs:
	:param word2idx:
	:return:
		sentences_matrix[graph(sentence)_index, token_ids_in_sentence] = token_ids_in_word2idx
		entity_matrix[graph(sentence)_index, edge(relation)_index, token_ids_in_sentence] = entity_marker(1~4)('The', 4)
		y_matrix[graph(sentence)_index, edge(relation)_index] = relation_type (relation_type)
	"""
	max_sent_len = model_params['max_sent_len']  # 200
	sentences_matrix = np.zeros((len(graphs), max_sent_len), dtype="int32")  # (sentence_number, sentence_len)
	arg1_matrix = np.zeros((len(graphs), max_sent_len), dtype="int8")
	arg2_matrix = np.zeros((len(graphs), max_sent_len), dtype="int8")
	y_matrix = np.zeros((len(graphs), 1), dtype="int16")  # relation type 1~7

	for index, g in enumerate(tqdm.tqdm(graphs, ascii=True)):

		token_wordvec_ids = embeddings.get_idx_sequence(g["Tokens"], word2idx)
		sentences_matrix[index, :len(token_wordvec_ids)] = token_wordvec_ids

		arg1_matrix[index, :len(token_wordvec_ids)] = 1
		arg2_matrix[index, :len(token_wordvec_ids)] = 1

		arg1_matrix[index, range(g["mentionArg1"]["start"], g["mentionArg1"]["end"]+1)] = 2
		arg2_matrix[index, range(g["mentionArg2"]["start"], g["mentionArg2"]["end"]+1)] = 2

		relation_type = g["relationType"]
		relation_type_id = property2idx.get(relation_type)
		y_matrix[index] = relation_type_id

	return sentences_matrix, arg1_matrix, arg2_matrix, y_matrix


def to_indices_with_relative_positions(graphs, word2idx):
	max_sent_len = model_params['max_sent_len']
	num_edges = len([e for g in graphs for e in g['edgeSet']])
	sentences_matrix = np.zeros((num_edges, max_sent_len), dtype="int32")
	entity_matrix = np.zeros((num_edges, 2, max_sent_len), dtype="int8")
	y_matrix = np.zeros(num_edges, dtype="int16")
	index = 0
	max_entity_index = max_sent_len - 1
	for g in tqdm.tqdm(graphs, ascii=True):
		token_ids = embeddings.get_idx_sequence(g["tokens"], word2idx)
		if len(token_ids) > max_sent_len:
			token_ids = token_ids[:max_sent_len]
		for edge in g["edgeSet"]:
			sentences_matrix[index, :len(token_ids)] = token_ids
			_, property_kbid, _ = graph_utils.edge_to_kb_ids(edge, g)
			property_kbid = property2idx[property_kbid]
			entity_vector = graph_utils.get_entity_indexed_vector(token_ids, edge, mode="position")
			entity_vector = [
				(-max_entity_index if m1 < -max_entity_index else max_entity_index if m1 > max_entity_index else m1,
				 -max_entity_index if m2 < -max_entity_index else max_entity_index if m2 > max_entity_index else m2) for
				_, m1, m2 in entity_vector]
			entity_matrix[index, :, :len(token_ids)] = [[position2idx[m] for m, _ in entity_vector],
														[position2idx[m] for _, m in entity_vector]]

			y_matrix[index] = property_kbid
			index += 1
	return [sentences_matrix, entity_matrix, y_matrix]


def softmax(x):
	e_x = np.exp(x)
	return e_x / e_x.sum(axis=0)


if __name__ == "__main__":
	print("test")