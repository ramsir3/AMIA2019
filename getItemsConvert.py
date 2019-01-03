#!/usr/bin/env python3

import os.path as path
import mysql.connector, json, csv, time, sys
from collections import OrderedDict
from constants import *

# All item ids (CE & LE)
ITEM_IDS = CE_ITEM_LABELS + TOP_90_LE_ITEM_IDS

# Get item values from tables other than CE
def getOtherItems(tables, row, hour, cursor, idict):
    for t in tables:
        cursor.execute(
            """
            SELECT
                `ITEMID`,
                `VALUENUM`
            FROM `TIMED_%s`
            WHERE time=%s AND SUBJECT_ID=%d AND HADM_ID=%d
            """ %(t, hour, row[0], row[1])
        )
        for l in cursor:
            if l[0] in idict:
                idict[l[0]].append(l[1])
    return idict

# Get item values from CE and perform unit conversions
def getCEItems(row, hour, cursor, idict):
    cursor.execute(
        """
        SELECT
            `ITEMID`,
            `VALUENUM`
        FROM `TIMED_%s`
        WHERE time=%s AND SUBJECT_ID=%d AND HADM_ID=%d
        """ %('CHARTEVENTS', hour, row[0], row[1])
    )
    for l in cursor:
        for cat in ITEM_IDS_UOM:
            if l[0] in ITEM_IDS_UOM[cat]:
                idict[cat].append(CONVERSIONS[cat](l[1], ITEM_IDS_UOM[cat][l[0]]))
    return idict

# call getCEItems & getOtherItems, store them in a dict, and return the avg in the hour
# if there is no observation for the item in the hour insert a '?'
def getItems(tables, row, hour, cursor):
    idict = OrderedDict((i, list()) for i in ITEM_IDS)
    idict = getCEItems(row, hour, cursor, idict)
    idict = getOtherItems(tables[1:], row, hour, cursor, idict)

    out = []
    return_bool = False
    for v in idict.values():
        if len(v) == 0:
            out.append('?')
        else:
            return_bool = True
            out.append(sum(v)/len(v))

    # if no items were collected return None
    if return_bool:
        return out
    else:
        return None

# get demographic information, the label, and the time of the staging (TSTAGE)
# if the subject is AKI negative then label and TSTAGE = 0
# Male is coded as 1, Female is 0
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
    return ([label[0], demo[0], 0 if demo[1] == 'F' else 1], [label[1]])

# get items given hour
def collectHour(hour, cursor1, cursor2, fnout, limit=(0, 1000)):
    # q = """
    #     SELECT
    #         `SUBJECT_ID`,
    #         `HADM_ID`,
    #         `ADMISSION_TYPE`,
    #         `ADMISSION_LOCATION`,
    #         `INSURANCE`,
    #         `LANGUAGE`,
    #         `RELIGION`,
    #         `MARITAL_STATUS`,
    #         `ETHNICITY`,
    #         `DIAGNOSIS`
    #     FROM `FILTERED_ADMISSIONS`
    #     """
    # query for subject ids and hadm ids in the cohort
    q = """
        SELECT
            `SUBJECT_ID`,
            `HADM_ID`,
            `ETHNICITY`
        FROM `FILTERED_ADMISSIONS`
        """
    if limit != None:
        if type(limit) == tuple:
            q += "\nLIMIT %d, %d" % limit
        else:
            q += "\nLIMIT %d" % limit

    #  write to an output file
    with open(fnout, 'w') as fout:
        cfout = csv.writer(fout)
        cfout.writerow([ # write headers
            "SUBJECT_ID",
            "HADM_ID",
            "TSTAGE",
            "AGE",
            "GENDER",
            "ETHNICITY",
        ] + ITEM_IDS + ["STAGE"])
        t0 = time.time()
        print(q)
        cursor1.execute(q)
        # for every subject & hadm id get items in CE & LE in the given hour
        for row in cursor1:
            items = getItems(["CHARTEVENTS", "LABEVENTS"], row, hour, cursor2)
            if items != None:
                demos, label = getDemos(row, hour, cursor2)
                cfout.writerow(list(row[:2])+demos+list(row[2:])+items+label)
        t1 = time.time()
        print("took %f s" % (t1 - t0))

# user error proofing the command line arguments
nargs = len(sys.argv)
hour = None
if nargs > 1:
    try:
        _ = int(sys.argv[-1])
        hour = sys.argv[-1]
    except:
        raise ValueError()
else:
    raise ValueError('hour is required')

# open a connection to the SQL server usin the configs in .dbconfig.json
cfgs = None
with open('.dbconfig.json') as cf:
    cfgs = json.loads(cf.read())
mydb = mysql.connector.connect(**cfgs)

# get 2 cursors to execute the queries
cursor1 = mydb.cursor(buffered=True)
cursor2 = mydb.cursor(buffered=True)

# run the function to get the csv output
collectHour(hour, cursor1, cursor2, path.join(DATA_PATH, "HOUR_%05d.csv" % int(hour)), limit=None)

# close SQL connections
cursor1.close()
cursor2.close()
mydb.close()
