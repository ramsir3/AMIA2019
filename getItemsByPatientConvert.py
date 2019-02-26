#!/usr/bin/python3

import os.path as path
import mysql.connector, json, csv, time, sys, datetime, argparse
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
        return ['?' for i in ITEM_IDS], ['?' for i in ITEM_IDS]

# get demographic information, the label, and the time of the staging (TSTAGE)
# if the subject is AKI negative then label and TSTAGE = 0
# Male is coded as 1, Female is 0
def getDemos(row, hour, cursor, labelHours):
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
    label = []
    for h in labelHours:
        cursor.execute(
            """
            SELECT
                `time`,
                `STAGE`
            FROM `cohort_hour_staged_nonzero`
            WHERE SUBJECT_ID=%d AND HADM_ID=%d AND time BETWEEN %s AND %d
            """ % (row[0], row[1], hour, h + int(hour))
        )
        label.append(cursor.fetchone())

    plabel = []
    if int(hour) > 1:
        for h in labelHours:
            cursor.execute(
                """
                SELECT
                    `time`,
                    `STAGE`
                FROM `cohort_hour_staged_nonzero`
                WHERE SUBJECT_ID=%d AND HADM_ID=%d AND time BETWEEN %d AND %d
                """ % (row[0], row[1], int(hour) - 1, h - 1 + int(hour))
            )
            plabel.append(cursor.fetchone())
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
        plabels = cursor.fetchall()
        if len(plabels) > 0:
            plabel = [plabels[-1] for h in labelHours]

    label = [f(l) if l != None else f([0,0]) for l in label for f in (lambda x: x[s] for s in [0,1])]
    if len(plabel) > 0:
        plabel = [f(l) if l != None else f([0,0]) for l in plabel for f in (lambda x: x[s] for s in [0,1])]
    else:
        plabel = [0 for l in label]

    return ([demo[0], 0 if demo[1] == 'F' else 1], plabel, label)

# get items given hour
def collectPatient(cursor1, cursor2, outpath, n_patients=[100], hours=24, labelHours=[48]):
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
    if n_patients != None:
        if len(n_patients) == 2:
            q += "        LIMIT %d, %d" % tuple(n_patients)
        elif len(n_patients) == 1:
            q += "        LIMIT %d" % n_patients[0]
        else:
            raise ValueError('npatients must have length 1 or 2')

    # for every subject & hadm id get items in CE & LE in the given hour
    print(q)
    cursor1.execute(q)

    for row in cursor1:
        fnout = path.join(outpath, "S%d_H%d.csv" % (row[0], row[1]))
        print("S%d_H%d.csv" % (row[0], row[1]), 'START', datetime.datetime.now().strftime('%Y-%m-%d %X'), end=' ')
        sys.stdout.flush()
        l0 = time.perf_counter() 
        #  write to an output file
        with open(fnout, 'w') as fout:
            cfout = csv.writer(fout)
            # write headers
            cfout.writerow(
                [ 
                    "HOUR",
                    "SUBJECT_ID",
                    "HADM_ID",
                    "ETHNICITY",
                    "AGE",
                    "GENDER",
                ] + ['P ' + i for i in ITEM_LABELS]
                # [f(n) for n in a for f in (lambda x: 'a%d'%x, lambda x: 'b%d'%x)]
                + [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["PTSTAGE", "PSTAGE"])]
                + ITEM_LABELS
                + [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["TSTAGE", "STAGE"])]
            )

            for hour in range(1, hours+1):
                pitems, items = getItems(["CHARTEVENTS", "LABEVENTS"], row, hour, cursor2)
                if items != None:
                    demos, plabel, label = getDemos(row, hour, cursor2, labelHours)
                    cfout.writerow([hour]+list(row[:2])+list(row[2:])+demos+pitems+plabel+items+label)
        print('DONE', datetime.datetime.now().strftime('%Y-%m-%d %X'), 'TOOK %.3fs' % (time.perf_counter() - l0))


# user error proofing the command line arguments
parser = argparse.ArgumentParser(description='Get MIMIC visit data by hour')
parser.add_argument('hours', metavar='H', type=int, help='number of hours of data to collect')
parser.add_argument('labelHours', metavar='L', type=int, nargs='+', help='hours at which labels will be collected')
parser.add_argument('-n', '--npatients', type=int, nargs='+', help='number of patients to limit the queries to')
parser.add_argument('-o', '--outdir', type=str, help='output directory (default: data/bypt/)')
args = parser.parse_args()
# print(args)

# open a connection to the SQL server usin the configs in .dbconfig.json
cfgs = None
with open('.dbconfig.json') as cf:
    cfgs = json.loads(cf.read())
mydb = mysql.connector.connect(**cfgs)

# get 2 cursors to execute the queries
cursor1 = mydb.cursor(buffered=True)
cursor2 = mydb.cursor(buffered=True)


# set output dir
# outpath = PROCESSED_PATH
outpath = path.join(DATA_PATH, 'bypt')
if args.outdir != None:
    outpath = args.outdir

# run the function to get the csv output
print('START', datetime.datetime.now().strftime('%Y-%m-%d %X'))
collectPatient(cursor1, cursor2, outpath, n_patients=args.npatients, hours=args.hours, labelHours=args.labelHours)
print('END', datetime.datetime.now().strftime('%Y-%m-%d %X'))

# close SQL connections
cursor1.close()
cursor2.close()
mydb.close()
