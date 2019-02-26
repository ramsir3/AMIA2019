#!/usr/bin/python3

import mysql.connector, json, csv, time, sys, os.path, pickle
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

# outfn = 'timecounts.csv'
# t0 = time.time()
# with open(os.path.join(PROCESSED_PATH, outfn), 'w') as outf:
#     coutf = csv.writer(outf)
#     counts = defaultdict(lambda: defaultdict(lambda: 0))
#     for c in range(4):
#         query = """
#             SELECT
#                 `time`,
#                 COUNT(STAGE) AS STAGE%d 
#             FROM `simple_stage`
#             WHERE STAGE=%d 
#             GROUP BY `time`
#             """ % (c, c)

#         print(query)
#         cursor.execute(query)
#         for l in cursor:
#             counts[l[0]][c] = l[1]

#     coutf.writerow(["HOUR", "STAGE 0", "STAGE 1", "STAGE 2", "STAGE 3"])
#     for h in counts:
#         coutf.writerow([h,counts[h][0],counts[h][1],counts[h][2],counts[h][3]])
# t1 = time.time()

q = """
SELECT ITEMID, AVG(average) FROM `AVG_TIMED_LABEVENTS`
WHERE ITEMID IN (
    50912, 51006, 51221, 51265, 51301, 51222, 51249, 51248, 51250, 51279, 51277,
    50971, 50983, 50902, 50882, 50868, 50931, 50960, 50970, 50893, 51237, 51274,
    51275, 50820, 51491, 51498, 51244, 51254, 51256, 51146, 51200, 50802, 50804,
    50821, 50818, 50813, 50861, 50878, 51492, 50885, 50863, 51484, 51514, 51478,
    50862, 51508, 51506, 51466, 51464, 51487, 51519, 51516, 51493, 51486, 51476,
    50822, 50808, 50910, 50809, 50911, 50800, 51463, 50954, 50810, 50811, 50824,
    50817, 50812, 50956, 51003, 50806, 50825, 50816, 50867, 51144, 51214, 50826,
    51087, 50828, 50920, 51233, 51082, 51246, 50819, 51143, 51251, 51255, 51137,
    51252, 51100
    )
GROUP BY ITEMID
"""

q = """
SELECT ITEMID, AVG(average), COUNT(average) FROM `AVG_TIMED_CHARTEVENTS`
WHERE ITEMID IN (
    762, 763, 3580, 3581, 3582, 226512, 226531, 224639, 920, 1394, 4187, 4188,
    3485, 3486, 226707, 226730, 455, 51, 225309, 442, 6701, 8441, 8368, 225310,
    8440, 8555, 678, 677, 223761, 679, 676, 223762, 226329, 227054, 614, 618,
    619, 1635, 1884, 220210, 224688, 224689, 224690, 211, 220045, 646, 834,
    6719, 220277
    )
GROUP BY ITEMID
"""

outfn = 'avgchart.pickle'
t0 = time.time()
with open(os.path.join(PROCESSED_PATH, outfn), 'wb') as outf:
    al = {}
    coutf = csv.writer(outf)
    print(q)
    cursor.execute(q)
    for l in cursor:
        al[l[0]] = l[1:]


    pickle.dump(al, outf)


t1 = time.time()


