class Entity:
	def __init__(self):
		self.entityID = ''
		self.type = ''
		self.subType = ''

	def match(self, type):
		if self.type == type:
			return True
		else:
			return False


class EntityMention(Entity):
	def __init__(self, extent, position):
		super().__init__()
		self.mentionID = ''
		self.extent = extent
		self.position = position

	def set(self, mentionID, entityID, type, subType):
		self.mentionID = mentionID
		super().entityID = entityID
		super().type = type
		super().subType = subType
