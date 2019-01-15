class Event:
	def __init__(self):
		self.eventID = ''
		self.type = ''
		self.subType = ''
		self.args = []


class EventArg:
	def __init__(self, id, role):
		self.argID = id
		self.argRole = role


class EventMention(Event):
	def __init__(self, extent, position, anchor, mentionArgs):
		super().__init__()
		self.mentionID = ''
		self.extent = extent
		self.position = position
		self.anchor = anchor
		self.mentionArgs = []
		for mentionArg in mentionArgs:
			self.mentionArgs.append(EvMentionArg(mentionArg['role'], mentionArg['extent']))

	def setID(self, mentionID, eventID):
		self.mentionID = mentionID
		self.eventID = eventID

	def setType(self, type):
		self.type = type

	def setSubType(self, subType):
		self.subType = subType

	def setArgs(self, eventArgList):
		for eventArg in eventArgList:
			self.args.append(EventArg(eventArg['id'], eventArg['role']))


class EvMentionArg:
	def __init__(self, role, extent):
		self.id = ''
		self.role = role
		self.extent = extent

	def setID(self, id):
		self.id = id
