#!/usr/bin/python3

import os.path as path
import mysql.connector, json, csv, time, sys, datetime
from collections import OrderedDict, defaultdict
from constants import DATA_PATH, PROCESSED_PATH, CE_ITEM_LABELS, TOP_90_LE_ITEM_IDS, ITEM_IDS_UOM, CONVERSIONS, VALIDATIONS
from lablabels import LAB_LABELS

# All item ids (CE & LE)
ITEM_IDS = CE_ITEM_LABELS + TOP_90_LE_ITEM_IDS
ITEM_LABELS = CE_ITEM_LABELS + [LAB_LABELS[i] for i in TOP_90_LE_ITEM_IDS]

# Get item values from tables other than CE
def getOtherItems(tables, row, hour, cursor, idict, prior=False):
    for t in tables:
        if prior:
            q = """
                SELECT
                    `ITEMID`,
                    `VALUENUM`
                FROM `TIMED_%s`
                WHERE time<%s AND SUBJECT_ID=%d AND HADM_ID=%d
                """ %(t, hour, row[0], row[1])
            cursor.execute(q)
            for l in cursor:
                if l[0] in idict:
                    idict[l[0]].append(l[1])
            q = """
                SELECT
                    `ITEMID`,
                    `VALUENUM`
                FROM `TIMED_%s`
                WHERE SUBJECT_ID=%d AND HADM_ID IN (
                    SELECT HADM_ID
                    FROM `FILTERED_ADMISSIONS`
                    WHERE SUBJECT_ID=%d AND ADMITTIME<(
                        SELECT ADMITTIME 
                        FROM `FILTERED_ADMISSIONS`
                        WHERE SUBJECT_ID=%d AND HADM_ID=%d
                    )
                )
                """ % (t, row[0], row[0], row[0], row[1])
            cursor.execute(q)
            for l in cursor:
                if l[0] in idict:
                    idict[l[0]].append(l[1])
        else:
            q = """
                SELECT
                    `ITEMID`,
                    `VALUENUM`,
                    `time`
                FROM `TIMED_%s`
                WHERE time>0 AND time<=%s AND SUBJECT_ID=%d AND HADM_ID=%d
                """ %('CHARTEVENTS', hour, row[0], row[1])
            cursor.execute(q)
            latest = {}
            for l in cursor:
                if l[0] not in latest:
                    latest[l[0]] = (l[2], [l[1]])
                else:
                    if l[2] > latest[l[0]][0]:
                        latest[l[0]] = (l[2], [l[1]])
                    if l[2] == latest[l[0]][0]:
                        latest[l[0]][1].append(l[1])
            for k, v in latest.items():
                if k in idict:
                    idict[k] = v[1]
    return idict

# Get item values from CE and perform unit conversions
def getCEItems(row, hour, cursor, idict, prior=False):
    if prior:
        q = """
            SELECT
                `ITEMID`,
                `VALUENUM`
            FROM `TIMED_%s`
            WHERE time<%s AND SUBJECT_ID=%d AND HADM_ID=%d
            """ % ('CHARTEVENTS', hour, row[0], row[1])
        cursor.execute(q)
        for l in cursor:
            for cat in ITEM_IDS_UOM:
                if l[0] in ITEM_IDS_UOM[cat]:
                    idict[cat].append(CONVERSIONS[cat](l[1], ITEM_IDS_UOM[cat][l[0]]))       
        q = """
            SELECT
                `ITEMID`,
                `VALUENUM`
            FROM `TIMED_%s`
            WHERE SUBJECT_ID=%d AND HADM_ID IN (
                SELECT HADM_ID
                FROM `FILTERED_ADMISSIONS`
                WHERE SUBJECT_ID=%d AND ADMITTIME<(
                    SELECT ADMITTIME 
                    FROM `FILTERED_ADMISSIONS`
                    WHERE SUBJECT_ID=%d AND HADM_ID=%d
                )
            )
            """ % ('CHARTEVENTS', row[0], row[0], row[0], row[1])
        cursor.execute(q)
        for l in cursor:
            for cat in ITEM_IDS_UOM:
                if l[0] in ITEM_IDS_UOM[cat]:
                    idict[cat].append(CONVERSIONS[cat](l[1], ITEM_IDS_UOM[cat][l[0]]))    
    else:
        q = """
            SELECT
                `ITEMID`,
                `VALUENUM`,
                `time`
            FROM `TIMED_%s`
            WHERE time<=%s AND SUBJECT_ID=%d AND HADM_ID=%d
            """ %('CHARTEVENTS', hour, row[0], row[1])
        cursor.execute(q)
        latest = {}
        for l in cursor:
            if l[0] not in latest:
                latest[l[0]] = (l[2], [l[1]])
            else:
                if l[2] > latest[l[0]][0]:
                    latest[l[0]] = (l[2], [l[1]])
                if l[2] == latest[l[0]][0]:
                    latest[l[0]][1].append(l[1])

        for itemid, values in latest.items():
            for cat in ITEM_IDS_UOM:
                if itemid in ITEM_IDS_UOM[cat]:
                    for v in values[1]:
                        idict[cat].append(CONVERSIONS[cat](v, ITEM_IDS_UOM[cat][itemid]))      
    return idict

def validate(itemid, value, margin):
    if itemid in VALIDATIONS:
        low = None if VALIDATIONS[itemid][0] == None else VALIDATIONS[itemid][0] * margin
        hi = None if VALIDATIONS[itemid][1] == None else VALIDATIONS[itemid][1] * (1 + margin)
        if low != None and hi != None:
            return value >= low and value <= hi
        elif low == None and hi != None:
            return value <= hi
        elif low != None and hi == None:
            return value >= low
    return True

def clean(idict, margin=0.5):
    return_bool = False
    for k, v in idict.items():
        if len(v) == 0:
            idict[k] = '?'
        else:
            return_bool = True
            rv = sum(v)/len(v)
            if validate(k, rv, margin):
                idict[k] = rv
            else:
                idict[k] = '!'# + str(rv)
    return return_bool, idict

# call getCEItems & getOtherItems, store them in a dict, and return the avg in the hour
# if there is no observation for the item in the hour insert a '?'
def getItems(tables, row, hour, cursor):
    margin = 0.5
    idict = OrderedDict((i, list()) for i in ITEM_IDS)
    idict = getCEItems(row, hour, cursor, idict)
    idict = getOtherItems(tables[1:], row, hour, cursor, idict)
    return_bool, idict = clean(idict, margin)

    if return_bool:            
        pdict = OrderedDict((i, list()) for i in ITEM_IDS)
        pdict = getCEItems(row, hour, cursor, pdict, True)
        pdict = getOtherItems(tables[1:], row, hour, cursor, pdict, True)
        _, pdict = clean(pdict, margin)
    # if no items were collected return None
        return list(pdict.values()), list(idict.values())
    else:
        return None, None

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

    plabel = None
    if int(hour) > 1:
        cursor.execute(
            """
            SELECT
                `time`,
                `STAGE`
            FROM `cohort_hour_staged_nonzero`
            WHERE SUBJECT_ID=%d AND HADM_ID=%d AND time BETWEEN %d AND %d
            """ % (row[0], row[1], int(hour) - 1, 47 + int(hour))
        )
        plabel = cursor.fetchone()
    else:
        cursor.execute(
            """
            SELECT
                `time`,
                `STAGE`
            FROM `cohort_hour_staged_nonzero`
            WHERE SUBJECT_ID=%d AND HADM_ID=%d AND time < %d
            """ % (row[0], row[1], int(hour))
        )
        plabel = cursor.fetchall()
        if len(plabel) > 0:
            plabel = plabel[-1]
        
    if label == None:
        label = [0, 0]
    if plabel == None or len(plabel) == 0:
        plabel = [0, 0]
    return ([demo[0], 0 if demo[1] == 'F' else 1], list(plabel), list(label))

# get items given hour
def collectHour(hour, cursor1, cursor2, fnout, limit=(3, 100)):
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
            q += "        LIMIT %d, %d" % limit
        else:
            q += "        LIMIT %d" % limit

    #  write to an output file
    with open(fnout, 'w') as fout:
        cfout = csv.writer(fout)
        cfout.writerow([ # write headers
            "SUBJECT_ID",
            "HADM_ID",
            "AGE",
            "GENDER",
            "ETHNICITY",
        ] + ['P ' + i for i in ITEM_LABELS] + ["P TSTAGE", "P STAGE"] + ITEM_LABELS + ["TSTAGE", "STAGE"])
        t0 = time.time()
        print(q)
        cursor1.execute(q)
        # for every subject & hadm id get items in CE & LE in the given hour
        for row in cursor1:
            pitems, items = getItems(["CHARTEVENTS", "LABEVENTS"], row, hour, cursor2)
            if items != None:
                demos, plabel, label = getDemos(row, hour, cursor2)
                cfout.writerow(list(row[:2])+demos+list(row[2:])+pitems+plabel+items+label)
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


#set output dir
outpath = PROCESSED_PATH
outpath = path.join(DATA_PATH, 'test')

# run the function to get the csv output
print(datetime.datetime.now())
collectHour(hour, cursor1, cursor2, path.join(outpath, "HOUR_%05d.csv" % int(hour)), limit=None)

# close SQL connections
cursor1.close()
cursor2.close()
mydb.close()
