#!/usr/bin/env python3

import os.path as path
import mysql.connector, sys, json, time, csv
from collections import OrderedDict
from constants import *
from itemsids import ALL_GT5

ITEM_IDS = ALL_GT5

def getItemRowDict(defval='?'):
    return OrderedDict.fromkeys(ITEM_IDS, defval)

def getItems(tables, row, hour, cursor):
    idict = getItemRowDict()
    for t in tables:
        cursor.execute(
            """
            SELECT
                `ITEMID`,
                `average`
            FROM `AVG_TIMED_%s`
            WHERE time=%s AND SUBJECT_ID=%d AND HADM_ID=%d
            """ %(t, hour, row[0], row[1])
        )
        for l in cursor:
            if l[0] in idict:
                idict[l[0]] = l[1]
    return list(idict.values())

def getDemos(row, hour, cursor):
    cursor.execute(
        """
        SELECT
            `age`,
            `GENDER`
        FROM `cohort`
        WHERE SUBJECT_ID=%d AND HADM_ID=%d
        """ % (row[0], row[1])
    )
    demo = cursor.fetchone()
    cursor.execute(
        """
        SELECT
            `time`,
            `STAGE`
        FROM `cohort_hour_staged_nonzero`
        WHERE SUBJECT_ID=%d AND HADM_ID=%d AND time BETWEEN %s AND %d
        """ % (row[0], row[1], hour, 48 + int(hour))
    )
    label = cursor.fetchone()
    if label == None:
        label = [0, 0]
    return ([demo[0], 0 if demo[1] == 'F' else 1, label[0]], [label[1]])

def collectHour(hour, table, cursor1, cursor2, fnout, limit=(0, 1000)):
    q = """
        SELECT
            `SUBJECT_ID`,
            `HADM_ID`,
            `ADMISSION_TYPE`,
            `ADMISSION_LOCATION`,
            `INSURANCE`,
            `LANGUAGE`,
            `RELIGION`,
            `MARITAL_STATUS`,
            `ETHNICITY`,
            `DIAGNOSIS`
        FROM `FILTERED_ADMISSIONS`
        """
    q = """
        SELECT
            `SUBJECT_ID`,
            `HADM_ID`
        FROM `FILTERED_ADMISSIONS`
        """
    if limit != None:
        if type(limit) == tuple:
            q += "\nLIMIT %s, %s" % limit
        else:
            q += "\nLIMIT %s" % limit

    with open(fnout, 'w') as fout:
        cfout = csv.writer(fout)
        cfout.writerow([
            "SUBJECT_ID",
            "HADM_ID",
            "AGE",
            "GENDER",
            "TSTAGE"
        ] + ITEM_IDS + ["STAGE"])
        t0 = time.time()
        print(q)
        cursor1.execute(q)
        for row in cursor1:
            items = getItems(["CHARTEVENTS", "LABEVENTS"], row, hour, cursor2)
            demos, label = getDemos(row, hour, cursor2)
            cfout.writerow(list(row)+demos+items+label)
        t1 = time.time()
        print("took %f s" % (t1 - t0))

cfgs = None
with open('.dbconfig.json') as cf:
    cfgs = json.loads(cf.read())
mydb = mysql.connector.connect(**cfgs)
cursor1 = mydb.cursor(buffered=True)
cursor2 = mydb.cursor(buffered=True)

nargs = len(sys.argv)
hour = None
table = None
if nargs > 1:
    try:
        _ = int(sys.argv[-1])
        hour = sys.argv[-1]
    except:
        raise ValueError()

collectHour(hour, table, cursor1, cursor2, path.join(PROCESSED_PATH, "HOUR_%05d.csv" % int(hour)), limit=None)

cursor1.close()
cursor2.close()
mydb.close()
