import json, sys, csv
import os.path as path
from collections import Counter, defaultdict
from statistics import stdev, mean
from constants import *

# data/processed/mimic3ScrBaseline.csv
def getIDs(fnin):
	ids = set()
	with open(fnin, 'r') as fin:
		for line in fin:
			line = line.strip()
			sline = line.replace("\"",'').split(',')
			try:
				if float(int(sline[2])) >= 18:
					ids.add(tuple(sline[:2]))
			except:
				pass
	return ids

def filterOnIDs(fnin, ids, force=False):
	fnout = path.join(
		PROCESSED_PATH,
		'FILTERED_' + path.basename(fnin)
		)
	if not force and path.isfile(fnout):
		print('File already exits!')
		return fnout
		
	with open(fnin, 'r') as fce:
		with open(fnout, 'w') as fout:
			i = 0
			c = 0
			for l in fce:
				i += 1
				sl = l.strip()
				sline = sl.replace("\"",'').split(',')
				if tuple(sline[1:3]) in ids:
					fout.write(l)
					c += 1
				if i%100000 == 0:
					sys.stdout.write('\rOn Line: %d' % i)
					sys.stdout.flush()
			print('\nLines Kept: ', c, '/', i)
	
	return fnout

def countItems(fnin, deffn, icol=4, dIncCols=[2]):
	fnout = path.join(
		PROCESSED_PATH,
		'ITEMCOUNTS_' + path.basename(fnin)
		)

	counts = Counter()
	with open(fnin, 'r') as fin:
		i = 0
		c = 0
		for l in fin:
			i += 1
			sl = l.strip()
			sline = sl.replace("\"",'').split(',')
			counts[sline[icol]] += 1
			if i%10000 == 0:
				sys.stdout.write('\rOn Line: %d' % i)
				sys.stdout.flush()
		print('\n')

	with open(deffn, 'r') as d:
		with open(fnout, 'w') as fout:
			csvfout = csv.writer(fout)
			csvfout.writerow(['ITEMID', 'LABEL'] + ['']*(len(dIncCols)-1) + ['COUNT'])
			i = 0
			for l in d:
				i += 1
				sline = l.strip().replace("\"",'').split(',')
				iid = sline[1]
				if iid in counts:
					csvfout.writerow([iid] + [sline[x] for x in dIncCols] + [counts[iid]])
				if i%100:
					sys.stdout.write('\rOn Line: %d' % i)
					sys.stdout.flush()
			print('\n')
			
	return fnout

def getDictStats(counts):
	out = defaultdict(dict)
	print
	for i in counts:
		v = counts[i].values()
		# print(i)
		
		out[i]['num'] = len(v)
		out[i]['avg'] = mean(v)
		out[i]['min'] = min(v)
		out[i]['max'] = max(v)
		try:
			out[i]['sd'] = stdev(v)
		except:
			out[i]['sd'] = -1
	return out

def getPatientItemStats(fnItems, fnin, icol=4, minObv=1000):
	perpt = {}
	perenc = {}
	with open(fnItems, 'r') as fItems:
		fItems.readline()
		for li in fItems:
			sli = li.strip().replace("\"",'').split(',')
			# print(sli[-1])
			iid, ic = sli[0], int(sli[-1])
			if ic > minObv:
				perpt[iid] = Counter()
				perenc[iid] = Counter()

	with open(fnin, 'r') as fin:
		for l in fin:
			sl = l.strip().replace("\"",'').split(',')
			iid = sl[icol]
			if iid in perpt:
				ptid = sl[1]
				encid = (sl[1], sl[2])
				perpt[iid][ptid] += 1
				perenc[iid][encid] += 1

	fnoutpt = path.join(
		PROCESSED_PATH,
		'ITEMCOUNTS_PTSTATS_' + path.basename(fnin)#.split('.')[0] + '.json'
		)
	fnoutenc = path.join(
		PROCESSED_PATH,
		'ITEMCOUNTS_ENCSTATS_' + path.basename(fnin)#.split('.')[0] + '.json'
		)

	perptstats, perencstats = getDictStats(perpt), getDictStats(perenc)
	with open(fnoutpt, 'w') as foutpt:
		# foutpt.write(repr(perpt))
		foutpt.write(json.dumps(perptstats))
	with open(fnoutenc, 'w') as foutenc:
		# foutenc.write(repr(perenc))
		foutenc.write(json.dumps(perencstats))
	
	dstr = lambda d: ', %d, %d, %d, %d, %d\n' % (d['num'], d['min'], d['avg'], d['max'], d['sd'])
	with open(fnItems, 'r') as fItems:
		with open(fnoutpt, 'w') as foutpt:
			with open(fnoutenc, 'w') as foutenc:
				h = fItems.readline()[:-1]+", NUM, MIN, AVG, MAX, SD\n"
				foutpt.write(h)
				foutenc.write(h)
				for l in fItems:
					l = l.strip()
					sl = l.replace("\"",'').split(',')
					if sl[0] in perptstats:
						foutpt.write(l + dstr(perptstats[sl[0]]))
						foutenc.write(l + dstr(perencstats[sl[0]]))

	return fnoutpt, fnoutenc