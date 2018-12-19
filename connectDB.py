import mysql.connector, json, sys, csv, time
import os.path as path
from collections import defaultdict, Counter
from statistics import stdev, mean
from constants import *

def timerDecorator(function):
    def wrapper(*args, **kwargs):
        t0 = time.time()
        out = function(*args, **kwargs)
        t1 = time.time()
        print("Function \"%s\" took %f s" % (function.__name__, (t1 - t0)))
        return out
    return wrapper

@timerDecorator
def countItems(table, cursor, icol=4, string=False):
    counts = defaultdict(Counter)
    i = 0
    for row in cursor:
        i += 1
        if string:
            counts[str(row[icol])][str(row[1:3])] += 1
        else:
            counts[row[icol]][row[1:3]] += 1

        if i%10000 == 0:
            sys.stdout.write('\rOn row: %d' % i)
            sys.stdout.flush()
    print('\n')

    return counts

def genCountCSV(counts, table, deffn, dIncCols=[2]):
    fnout = path.join(
    PROCESSED_PATH,
    'ITEMSTATS_' + table + '.csv'
    )

    with open(deffn, 'r') as d:
        with open(fnout, 'w') as fout:
            csvfout = csv.writer(fout)
            csvfout.writerow(['ITEMID', 'LABEL'] + ['']*(len(dIncCols)-1) + ['COUNT', 'NUM', 'MIN', 'AVG', 'MAX', 'SD'])
            i = 0
            for l in d:
                i += 1
                sline = l.strip().replace("\"",'').split(',')
                iid = sline[1]
                if iid in counts:
                    csvfout.writerow(
                        [iid]
                        + [sline[x] for x in dIncCols]
                        + [counts[iid]['count'], counts[iid]['num'], counts[iid]['min'], counts[iid]['avg'], counts[iid]['max'], counts[iid]['sd']])
                if i%100:
                    sys.stdout.write('\rOn Line: %d' % i)
                    sys.stdout.flush()
            print('\n')
            
def getDictStats(counts):
	out = defaultdict(dict)
	print
	for i in counts:
		v = counts[i].values()
		# print(i)
		out[i]['count'] = sum(v)
		out[i]['num'] = len(v)
		out[i]['avg'] = mean(v)
		out[i]['min'] = min(v)
		out[i]['max'] = max(v)
		try:
			out[i]['sd'] = stdev(v)
		except:
			out[i]['sd'] = -1
	return out


table = "FILTERED_CHARTEVENTS"
defs = 'D_ITEMS.csv'
icol = 4
dIncCols = [2]

# table = "FILTERED_LABEVENTS"
# defs = 'D_LABITEMS.csv'
# icol = 3
# dIncCols = [2,3,4]

cfgs = None
with open('.dbconfig.json') as cf:
    cfgs = json.loads(cf.read())
mydb = mysql.connector.connect(**cfgs)
cursor = mydb.cursor()
cursor.execute("SELECT * FROM %s" % table)

fnj = path.join(
    PROCESSED_PATH,
    'ITEMCOUNTS_' + table + '.json'
)
counts = countItems(table, cursor, icol=icol, string=True)
with open(fnj, 'w') as fj:
    fj.write(json.dumps(counts))
# with open(fnj, 'r') as fj:
#     counts = json.loads(fj.read())
countstats = getDictStats(counts)
genCountCSV(countstats, table, path.join(DEFINITIONS_PATH, defs), dIncCols=dIncCols)