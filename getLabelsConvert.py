#!/usr/bin/python3

import os.path as path
import mysql.connector, json, csv, time, sys, datetime
from collections import OrderedDict, defaultdict
from constants import DATA_PATH, PROCESSED_PATH, CE_ITEM_LABELS, TOP_90_LE_ITEM_IDS, ITEM_IDS_UOM, CONVERSIONS, VALIDATIONS
from lablabels import LAB_LABELS


# get demographic information, the label, and the time of the staging (TSTAGE)
# if the subject is AKI negative then label and TSTAGE = 0
# Male is coded as 1, Female is 0
def getDemos(row, hour, cursor, labelHours):
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

    return (plabel, label)

# get items given hour
def collectHour(hour, cursor1, cursor2, fnout, labelHours=[6, 12, 24, 36, 48, 72], limit=(3, 100)):
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
            `HADM_ID`
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
            ]
            + [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["PTSTAGE", "PSTAGE"])]
            + [f(h) for h in labelHours for f in (lambda x: '%s%d'%(s,x) for s in ["TSTAGE", "STAGE"])]
        )

        t0 = time.time()
        print(q)
        cursor1.execute(q)
        # for every subject & hadm id get items in CE & LE in the given hour
        for row in cursor1:
            plabel, label = getDemos(row, hour, cursor2, labelHours)
            cfout.writerow(list(row)+plabel+label)
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
# outpath = path.join(DATA_PATH, 'hour')
outpath = path.join(DATA_PATH, 'hourlabel')
limit = None
# limit = 10

# run the function to get the csv output
print(datetime.datetime.now())
collectHour(hour, cursor1, cursor2, path.join(outpath, "HOUR_%05d.csv" % int(hour)), limit=limit)

# close SQL connections
cursor1.close()
cursor2.close()
mydb.close()
