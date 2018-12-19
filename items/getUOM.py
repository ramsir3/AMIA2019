#!/usr/bin/env python3

import mysql.connector, json, csv, time, sys, os.path
from collections import defaultdict
ap = os.path.dirname(os.path.abspath(__file__)).rsplit('/', maxsplit=1, )[0]
print(ap)
sys.path.append(ap)
from constants import *

nargs = len(sys.argv)
if nargs > 2:
    cfgs = None
    with open('.dbconfig.json') as cf:
        cfgs = json.loads(cf.read())
    mydb = mysql.connector.connect(**cfgs)
    cursor = mydb.cursor()

    infn = sys.argv[1]
    table = sys.argv[2].upper()

    items = set()
    with open(infn, 'r') as inf:
        cinf = csv.reader(inf, delimiter='\t')
        cinf.__next__() #throw away header
        for row in cinf:
            if row[0] != '':
                items.add(int(row[0]))

    items_dict = {k: set() for k in items}
    t0 = time.time()
    query = """
    SELECT DISTINCT
        ITEMID,
        VALUEUOM
    FROM `FILTERED_%s`
    """ % table
    print(query)
    cursor.execute(query)
    for l in cursor:
        if l[0] in items:
            items_dict[l[0]].add(l[1])
    t1 = time.time()
    print("took %f s" % (t1 - t0))

    with open(infn, 'r') as inf:
        with open(os.path.join(PROCESSED_PATH, infn), 'w') as outf:
            cinf = csv.reader(inf, delimiter='\t')
            coutf = csv.writer(outf)
            cinf.__next__() #throw away header
            for row in cinf:
                if row[0] != '':
                    coutf.writerow(row + list(items_dict[int(row[0])]))

    cursor.close()
    mydb.close()
