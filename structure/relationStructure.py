import numpy as np


def find_type():
	type = ""
	subType = ""

	return type, subType


class Relation:
	def __init__(self, id, type, subType, arg):
		self.id = id
		self.type = type
		self.subType = subType

		self.arg1 = Entity(arg[0]["id"], arg[0]["extent"], arg[0]["start"], arg[0]["end"])
		self.arg2 = Entity(arg[1]["id"], arg[1]["extent"], arg[1]["start"], arg[1]["end"])


class Entity:
	def __init__(self, id, extent, start, end):
		self.id = id
		self.type, self.subType = find_type()
		self.extent = extent
		self.span = (start, end)

	def find_position(self, sentence):
		position = np.zeros(1, len(sentence))
		for i in range(self.span[0], self.span[1]+1):
			position[1, i] = 1
		return position

	def encode_type_marker(self, position):
		for token in position:
			if token:
				return np.ones(3, 1)
			else:
				return np.zeros(3, 1)
