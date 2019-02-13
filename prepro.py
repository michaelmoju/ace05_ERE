"""
Date: 2019/2/11
Version: 0
Last update: 2019/2/11
Author: Moju Wu
"""

from std import *

import xml.etree.ElementTree as ET
import json

from collections import OrderedDict

PREPRO_DIR = '/media/moju/data/work/ace05_ERE/data/prepro/'

def conv2json(index, tokens):
	orders = ["index", "tokens"]
	contents = dict()
	contents["index"] = index
	contents["tokens"] = tokens
	contents = OrderedDict([(key, contents[key]) for key in orders])  ## conver a ordered-dict according the orders
	return contents


def parse_source(fp):

	tree = ET.parse(fp)
	doc = tree.getroot()
	docid = ''
	content = ''
	category = ''

	if 'bn' in fp:
		category = 'bn'
		assert doc[0].tag == 'DOCID'
		docid = doc[0].text

		assert doc[3].tag == 'BODY'
		for child in doc[3]:
			if child.tag == 'TEXT':
				text = child
				turn = text[0]
				content = turn.text
	elif 'nw' in fp:
		category = 'nw'
		assert doc[0].tag == 'DOCID'
		docid = doc[0].text

		assert doc[3].tag == 'BODY'
		for child in doc[3]:
			if child.tag == 'TEXT':
				text = child
				content = text.text

	return docid, content, category


def prepro_file(fpath, nlp, props):

	docid, content, category = parse_source(fpath)

	annotation = nlp.annotate(content.rstrip('\n').replace('\n', ' '), properties=props)
	annotation = json.loads(annotation)

	return annotation, category


def prepro_files(fdir):
	from stanfordcorenlp import StanfordCoreNLP

	nlp = StanfordCoreNLP(r'/media/moju/data/Data/codeTest/stanford-corenlp-full-2018-02-27')
	props = {'annotators': 'tokenize,ssplit', 'pipelineLanguage': 'en', 'outputFormat': 'json'}

	f_count = 0
	for fname in os.listdir(fdir):
		if fname.endswith(".sgm"):
			f_count += 1

			annotation, category = prepro_file(fdir+fname, nlp, props)

			out_list = []
			token_list = []

			for sentence in annotation["sentences"]:
				index = sentence["index"]
				for token in sentence["tokens"]:
					token_list.append(token["word"])
				out_list.append(conv2json(index, token_list))
				token_list = []

			out_file_name = fname[:-4]
			with open(PREPRO_DIR+category+'/'+out_file_name+'.prepro.json', 'w') as f:
				json.dump(out_list, f, indent=4)
	lprint("f_count=", f_count)
	nlp.close()


if __name__ == '__main__':
	import argparse

	ap = argparse.ArgumentParser()

	ap.add_argument('-dir', help='directory to pre-process')
	myArg = ap.parse_args()

	if myArg.dir:
		prepro_files(myArg.dir)
