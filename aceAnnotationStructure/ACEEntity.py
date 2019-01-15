class Entity:
	def __init__(self):
		self.entityID = ''
		self.type = ''
		self.subType = ''


class EntityMention(Entity):
	def __init__(self, extent, position):
		super().__init__()
		self.mentionID = ''
		self.extent = extent
		self.position = position

	def set(self, mentionID, entityID, type, subType):
		self.mentionID = mentionID
		self.entityID = entityID
		self.type = type
		self.subType = subType
