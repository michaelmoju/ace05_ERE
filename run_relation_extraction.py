"""
Date: 2018/12/26
Version: 0
Last update: 2019/1/11
Author: Moju Wu
"""


from std import *
from Runner import Runner
import jpype
import glob
from collections import OrderedDict

from SSQA import Docxer, Dotter, Plotter, Scanner

_log = Logger(__name__)

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
	doc = json.loads(fp)
	out = [doc["docID"], doc["entityList"], doc["relationList"], doc["eventList"]]
	return out


def _read_ere(fp):
	doc = json.loads(fp)
	out = [doc["entityList"], doc["relationList"], doc["eventList"]]
	return out


def _fmeasure(matchedCount, ieLen, benchLen):
	p = matchedCount/ieLen
	r = matchedCount/benchLen
	fmeasure = 2*p*r / (p+r)
	return fmeasure


def _evaluate(bench, ere):
	_, entityListB, relationListB, eventListB = bench
	entity, relation, event = ere

	entityMatchCount = 0
	# evaluate entity
	for ientity in entity:
		for ientityB in entityListB:
			if ientityB.type == entity.type:
				entityMatchCount += 1

	fmeasureEntity = _fmeasure(entityMatchCount, len(entity), len(entityListB))



	# evaluate relation

	# evaluate event


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
		sfh, bfh = fs  # r:source, r:benchmark
		docIDS, documentID, sentences = _read_src(sfh)
		docIDB, entityList, relationList = _read_bench(bfh)

	@staticmethod
	def cmd__test_eval(self, fs):
		erefh, bfh = fs  # r:entity/relation/event r:benchmark
		_evaluate(_read_bench(bfh), _read_ere(erefh))


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
