"""
Date: 2018/12/26
Version: 0
Last update: 2019/1/14
Author: Moju Wu
"""

from std import *
from Runner import Runner
import jpype
import glob

from collections import OrderedDict

from SSQA import Docxer, Dotter, Plotter, Scanner

from aceAnnotationStructure.ACEEntity import *
from aceAnnotationStructure.ACERelation import *
from aceAnnotationStructure.ACEEvent import *

_log = Logger(__name__)

# =================================== Global =======================================#

ENTITY_MATCH_COUNT, ENTITY_COUNT, ENTITY_BENCH_COUNT = 0, 0, 0
RELATION_MATCH_COUNT, RELATION_COUNT, RELATION_BENCH_COUNT = 0, 0, 0
EVENT_MATCH_COUNT, EVENT_COUNT, EVENT_BENCH_COUNT = 0, 0, 0

PREPRO_DIR = '/media/moju/data/work/ace05_ERE/data/prepro/'
ENTITY_DIR = '/media/moju/data/work/ace05_ERE/data/entity/'
RELATION_DIR = '/media/moju/data/work/ace05_ERE/data/relation/'


def _init_java(args):
	myPaths = []
	for iArg in args:
		lprint(iArg)
		if '::' in iArg:
			myDir, *myJars = iArg.split('::')
			for iJar in myJars:
				myPath = '{}/{}'.format(myDir, iJar)
				if '*' in myPath:
					for iPath in glob.glob(myPath):
						myPaths.append(iPath)
				else:
					myPaths.append(myPath)
		else:
			myPaths.append(iArg)
	myClassPath = os.pathsep.join(myPaths)
	jpype.startJVM(jpype.getDefaultJVMPath(), "-ea", "-Dfile.encoding=UTF8", "-Djava.class.path=%s" % myClassPath)


def _read_src(fp):
	doc = json.load(fp)
	out = [doc["docID"], doc["documentID"], doc["sentences"]]
	return out


def _read_bench(fp):
	# [doc["docID"], doc["entityList"], doc["relationList"], doc["eventList"]]
	doc = json.load(fp)
	lprint(doc['docID'])

	entityMentions = []
	for entityObj in doc['entityList']:
		for entityMentionObj in entityObj['entityMentionList']:
			myMention = EntityMention(entityMentionObj['extent'], entityMentionObj['position'])
			myMention.set(entityMentionObj['id'], entityObj['entityID'], entityObj['entityType'],
						  entityObj['entitySubType'])
			entityMentions.append(myMention)

	relationMentions = []
	for relationObj in doc['relationList']:
		for relationMentionObj in relationObj['relationMentionList']:
			myMention = RelationMention(relationMentionObj['mentionArg1']['extent'],
										relationMentionObj['mentionArg2']['extent'],
										relationMentionObj['position'])
			myMention.mentionArg1.setID(relationMentionObj['mentionArg1']['id'])
			myMention.mentionArg2.setID(relationMentionObj['mentionArg2']['id'])
			myMention.set(relationMentionObj['id'], relationMentionObj['extent'], relationObj['relationID'],
						  relationObj['relationArg1'], relationObj['relationArg2'], relationObj['relationType'],
						  relationObj['relationSubType'])
			relationMentions.append(myMention)

	eventMentions = []
	for eventObj in doc['eventList']:
		for eventMentionObj in eventObj['eventMentionList']:
			myMention = EventMention(eventMentionObj['extent'], eventMentionObj['position'], eventMentionObj['anchor'],
									 eventMentionObj['eventMentionArgList'])
			myMention.setID(eventMentionObj['id'], eventObj['eventID'])
			myMention.setType(eventObj['eventType'])
			myMention.setSubType(eventObj['eventSubType'])
			myMention.setArgs(eventObj['eventArgList'])
			eventMentions.append(myMention)

	return entityMentions, relationMentions, eventMentions


def _read_ere(fp):
	doc = json.loads(fp)
	out = [doc["entityList"], doc["relationList"], doc["eventList"]]
	return out


def _fmeasure(matchedCount, ieLen, benchLen):
	p = matchedCount / ieLen
	r = matchedCount / benchLen
	fmeasure = 2 * p * r / (p + r)
	return fmeasure


def _evaluate(bench, ere):
	entityMentions_bench, relationMentions_bench, eventMentions_bench = bench
	entityMentions, relationMentions, eventMentions = ere

	global ENTITY_MATCH_COUNT, ENTITY_COUNT, ENTITY_BENCH_COUNT

	# evaluate entity
	for entityMention in entityMentions:
		for entityMention_bench in entityMentions_bench:
			if entityMention.position == entityMention_bench.position:
				if (entityMention.extent == entityMention_bench.extent) & \
						(entityMention.type == entityMention_bench.type):
					ENTITY_MATCH_COUNT += 1
	ENTITY_COUNT += len(entityMentions)
	ENTITY_BENCH_COUNT += len(entityMentions_bench)

	global RELATION_MATCH_COUNT, RELATION_COUNT, RELATION_BENCH_COUNT

	# evaluate relation
	for relationMention in relationMentions:
		for relationMention_bench in relationMentions_bench:
			if relationMention.position == relationMention_bench.position:
				if (relationMention.mentionArg1 == relationMention_bench.mentionArg1) & \
						(relationMention.mentionArg2 == relationMention_bench.mentionArg2) & \
						(relationMention.type == relationMention_bench.type):
					RELATION_MATCH_COUNT += 1
	RELATION_COUNT += len(relationMentions)
	RELATION_BENCH_COUNT += len(relationMentions_bench)

	global EVENT_MATCH_COUNT, EVENT_COUNT, EVENT_BENCH_COUNT

	# evaluate event
	for eventMention in eventMentions:
		for eventMention_bench in eventMentions_bench:
			if eventMention.position == eventMention_bench.position:
				if eventMention.type == eventMention_bench.type:
					EVENT_MATCH_COUNT += 1
	EVENT_COUNT += len(eventMentions)
	EVENT_BENCH_COUNT += len(eventMentions_bench)


class _Runner(Runner):
	def __init__(self, arg):
		super().__init__(arg)
		self.cmd = getattr(self, 'cmd__{}'.format(self.arg.cmd))

	def run_one(self, fs, plg=False, elg=False):
		if plg or elg:
			return self.cmd(fs, plg, elg)
		else:
			return self.cmd(fs)

	@staticmethod
	def cmd__test_read(fs):
		# sfh, bfh = fs  # r:source, r:benchmark
		bfh, = fs
		# docIDS, documentID, sentences = _read_src(sfh)
		entityMentions, relationMentions, eventMentions = _read_bench(bfh)
		lprint(entityMentions[0].type)

	def cmd__test_eval(self, fhs, plg=False, elg=False):
		if plg:
			return
		# ----
		bfh, erefh = fhs
		bench, ere = _read_bench(bfh), _read_ere(erefh)
		# ----
		if elg:
			lprint(_fmeasure(ENTITY_MATCH_COUNT, ENTITY_COUNT, ENTITY_BENCH_COUNT))

		_evaluate(bench, ere)


def get_runner(arg):
	myCmds = arg.cmd.split('.')
	if len(myCmds) == 1:
		return _Runner(arg)
	elif len(myCmds) > 2:
		arg.cmd = myCmds.pop()
		import importlib
		# myModule = 'SSQA.{}.Sys'.format('.'.join(myCmds))
		# lprint(myModule)
		# lprint(getattr(importlib.import_module('SSQA.{}.Sys'.format('.'.join(myCmds))), 'Sys'))
		return getattr(importlib.import_module('SSQA.{}.Sys'.format('.'.join(myCmds))), 'Sys')(arg)
	else:
		mySys, arg.cmd = myCmds
		if mySys == 'Docxer':
			return Docxer(arg)
		elif mySys == 'Dotter':
			return Dotter(arg)
		elif mySys == 'Plotter':
			return Plotter(arg)
		elif mySys == 'Scanner':
			return Scanner(arg)
		elif mySys in ['SysCyutB1', 'SysCyutB1e']:
			from SSQA.Sys.Cyut.Sys import SysB1 as Sys
			return Sys(arg, allow_null_evid=(mySys[-1] != 'e'))
		elif mySys in ['SysCyutB2', 'SysCyutB2S', 'SysCyutB2e']:
			from SSQA.Sys.Cyut.Sys import SysB2 as Sys
			return Sys(arg, simplified=(mySys == 'SysCyutB2S'), allow_null_evid=(mySys[-1] != 'e'))
		elif mySys == 'SysV0':
			from SSQA.Sys.V0.Sys import Sys
			return Sys(arg)
		elif mySys == 'SysV1R1':
			from SSQA.Sys.V1.R1.Sys import Sys
			return Sys(arg)
		elif mySys.startswith('SysV1R2'):
			from SSQA.Sys.V1.R2.Sys import Sys
			return Sys(arg)
		else:
			raise ValueError()


def main(arg):
	if arg.jvm:
		_init_java(arg.jvm)
	myRunner = get_runner(arg)
	myRunner.run_all()


if __name__ == '__main__':
	import argparse

	ap = argparse.ArgumentParser()
	# ----
	ap.add_argument('-gvb', default='S:', help='Graphviz bin dir')
	ap.add_argument('-occ', help='OpenCC home dir')
	ap.add_argument('-cns', help='CoreNlp Server URL (http://localhost:9000)')
	ap.add_argument('-cwn', help='Chinese WordNet dir')
	ap.add_argument('-jvm', nargs='*', help='Run Java Virtual Machine via JPype')
	# ----
	ap.add_argument('-rfs', nargs='*', help='Runner: File-Spec')
	ap.add_argument('-rpe', nargs='*', help='Runner: Prologue and Epilogue')
	ap.add_argument('-reo', help='Runner: Epilogue Output File')
	ap.add_argument('-rno', action='store_true', help='Runner: Not-Open')
	ap.add_argument('-rlc', nargs='*', help='Runner: Looping-Char')
	ap.add_argument('-rip', nargs='*', help='Runner: Including Pattern')
	ap.add_argument('-rxp', nargs='*', help='Runner: eXcluding Pattern')
	ap.add_argument('-rpg', nargs='*', help='Runner: Parameter Grid')
	ap.add_argument('-rsa', nargs='*', help='Runner: Skipper Arguments')
	ap.add_argument('-rpa', help='Runner: Pool Arguments')  # worker-number[:time-out[:poll-gap[:listen]]]
	ap.add_argument('-rja', nargs='*', help='Runner: Job-Arguments')
	# ----
	ap.add_argument('-cmd', required=True, help='Command')
	ap.add_argument('-echo', action='store_true', help='Echo command only')
	# ----
	ap.add_argument('-force', action='store_true', help='Overwrite existing output files')
	ap.add_argument('-log', help='Output log file')
	ap.add_argument('-llv', default='50', help='Logging level')
	ap.add_argument('-v', type=int, default=0, help='Verbose level')
	myArg = ap.parse_args()

	if myArg.log:
		# myFormat = '%(asctime)s %(filename)s(%(lineno)d): %(message)s'
		myFormat = '%(asctime)s: %(message)s'
		logging.basicConfig(handlers=[log_w(myArg.log)], level=str2llv(myArg.llv), format=myFormat)
		_log.log(100, ' '.join(sys.argv))
	else:
		myFormat = '%(filename)s(%(lineno)d): %(message)s'
		logging.basicConfig(level=str2llv(myArg.llv), format=myFormat)
		_log.log(100, ' '.join(sys.argv))

	main(myArg)
