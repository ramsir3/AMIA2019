#!/usr/bin/python3

import mysql.connector, json, csv, time, sys, os.path
from collections import defaultdict
ap = os.path.dirname(os.path.abspath(__file__)).rsplit('/', maxsplit=1, )[0]
print(ap)
sys.path.append(ap)
from constants import *

# query = """
# SELECT DISTINCT ITEMID, VALUEUOM FROM `FILTERED_LABEVENTS` WHERE ITEMID IN (50861,50862,50863,50867,50868,51137,50878,51143,51463,51144,50802,51146,50882,50885,51464,51466,50893,50804,50902,50806,50911,50910,50912,51082,51200,51476,50920,51214,50808,50931,51478,50809,51221,50810,51222,50811,51233,51237,50812,51484,50813,50954,51087,51486,50956,51244,51246,50960,51248,51249,51250,51251,51252,51254,51255,51256,51487,50816,50817,50818,50819,50820,51491,50970,51265,50821,50971,50822,51492,51274,51275,51493,51277,51279,50983,50824,51100,51498,50800,50825,50826,51003,51006,51506,51508,51514,50828,51516,51301,51519)
# """
# query = """
#         SELECT
#             `time`,
#             COUNT(STAGE) AS STAGE%d 
#         FROM `simple_stage`
#         WHERE STAGE=%d 
#         GROUP BY `time`
#         """ % 0
# # query = None
# nargs = len(sys.argv)
# if query != None and nargs > 1:
#     cfgs = None
#     with open('.dbconfig.json') as cf:
#         cfgs = json.loads(cf.read())
#     mydb = mysql.connector.connect(**cfgs)
#     cursor = mydb.cursor()

#     if nargs >= 1:
#         outfn = sys.argv[1]

#     t0 = time.time()
#     with open(os.path.join(PROCESSED_PATH, outfn), 'w') as outf:
#         print(query)
#         cursor.execute(query)
#         coutf = csv.writer(outf)
#         for l in cursor:
#             coutf.writerow(l)
#     t1 = time.time()
#     print("took %f s" % (t1 - t0))

#     cursor.close()
#     mydb.close()


cfgs = None
with open('.dbconfig.json') as cf:
    cfgs = json.loads(cf.read())
mydb = mysql.connector.connect(**cfgs)
cursor = mydb.cursor()

outfn = 'timecounts.csv'
t0 = time.time()
with open(os.path.join(PROCESSED_PATH, outfn), 'w') as outf:
    coutf = csv.writer(outf)
    counts = defaultdict(lambda: defaultdict(lambda: 0))
    for c in range(4):
        query = """
            SELECT
                `time`,
                COUNT(STAGE) AS STAGE%d 
            FROM `simple_stage`
            WHERE STAGE=%d 
            GROUP BY `time`
            """ % (c, c)

        print(query)
        cursor.execute(query)
        for l in cursor:
            counts[l[0]][c] = l[1]

    coutf.writerow(["HOUR", "STAGE 0", "STAGE 1", "STAGE 2", "STAGE 3"])
    for h in counts:
        coutf.writerow([h,counts[h][0],counts[h][1],counts[h][2],counts[h][3]])
t1 = time.time()