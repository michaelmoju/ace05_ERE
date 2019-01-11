
from std import *
from Runner import Runner
import jpype
import glob
from collections import OrderedDict

from SSQA import Docxer, Dotter, Plotter, Scanner


class Entity:
	def __init__(self):
		self.id = ''
		self.type = ''
		self.subType = ''
		self.mentions = []

	def set(self, id, type, subType, mentions):
		self.id = id
		self.type = type
		self.subType = subType
		self.mentions = mentions

	def match(self, type):
		if self.type == type:
			return True
		else:
			return False

	def dump(self):
		pass


class EntityMention:
	def __init__(self):
		self.id = ''
		self.extent = ''
		self.position = None

	def set(self, id, extent, position):
		self.id = id
		self.extent = extent
		self.position = position



