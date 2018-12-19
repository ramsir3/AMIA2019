#!/usr/bin/env python3

import mysql.connector, json, csv, time, sys, os.path
ap = os.path.dirname(os.path.abspath(__file__)).rsplit('/', maxsplit=1, )[0]
print(ap)
sys.path.append(ap)
from constants import *

query = """
SELECT `time`, COUNT(`time`) FROM `TIMED_CHARTEVENTS` GROUP BY `time` ORDER BY `time`
"""

# query = None
nargs = len(sys.argv)
if query != None and nargs > 1:
    cfgs = None
    with open('.dbconfig.json') as cf:
        cfgs = json.loads(cf.read())
    mydb = mysql.connector.connect(**cfgs)
    cursor = mydb.cursor()

    if nargs >= 1:
        outfn = sys.argv[1]

    t0 = time.time()
    with open(os.path.join(PROCESSED_PATH, outfn), 'w') as outf:
        print(query)
        cursor.execute(query)
        coutf = csv.writer(outf)
        for l in cursor:
            coutf.writerow(l)
    t1 = time.time()
    print("took %f s" % (t1 - t0))

    cursor.close()
    mydb.close()
