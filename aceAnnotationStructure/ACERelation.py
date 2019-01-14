class Relation:
	def __init__(self):
		self.relationID = ''
		self.type = ''
		self.subType = ''
		self.arg1 = ''
		self.arg2 = ''

	def match(self, type):
		if self.type == type:
			return True
		else:
			return False


class RelationMention(Relation):
	def __init__(self, ReMentionArg1, ReMentionArg2, position):
		super().__init__()
		self.mentionID = ''
		self.extent = ''
		self.position = position
		self.mentionArg1 = ReMentionArg(ReMentionArg1, position)
		self.mentionArg2 = ReMentionArg(ReMentionArg2, position)

	def set(self, id, extent, relationID, arg1, arg2, type, subType):
		self.mentionID = id
		self.extent = extent
		super().relationID = relationID
		super().arg1 = arg1
		super().arg2 = arg2
		super().type = type
		super().subType = subType


class ReMentionArg:
	def __init__(self, extent, position):
		self.mentionArgID = ''
		self.extent = extent
		self.position = position

	def setID(self, mentionArgID):
		self.mentionArgID = mentionArgID



