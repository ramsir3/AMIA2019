#!/usr/bin/env python3

import mysql.connector, json, time, sys

queries = ["""
CREATE TABLE FILTERED_ADMISSIONS
SELECT d.* FROM (
    (
        SELECT *
        FROM ramsandbox.`d_ids`
    ) i
    INNER JOIN
    (
        SELECT *
        FROM mimic3.`ADMISSIONS`
    ) d
    ON (i.HADM_ID = d.HADM_ID)
    AND (i.SUBJECT_ID = d.SUBJECT_ID)
)
""",]

# queries = ["""
# CREATE TABLE TIMED_LABEVENTS
# SELECT c.*, a.ADMITTIME AS ADMITTIME, TIMESTAMPDIFF(HOUR, a.ADMITTIME, c.CHARTTIME) AS 'time'
# FROM
# (
# 	(
# 		SELECT *
# 		FROM `FILTERED_ADMISSIONS`
# 	) a
# 	INNER JOIN
# 	(
# 		SELECT *
# 		FROM `FILTERED_LABEVENTS`
# 		WHERE VALUENUM IS NOT NULL
# 	) c
# 	ON (a.HADM_ID = c.HADM_ID) AND (a.SUBJECT_ID = c.SUBJECT_ID)
# )
# ORDER BY time
# ""","""
# CREATE INDEX main_idx ON TIMED_LABEVENTS(SUBJECT_ID,HADM_ID,ITEMID,time)
# """,]

# queries = ["""
# CREATE TABLE TIMED_CHARTEVENTS
# SELECT c.*, a.ADMITTIME AS ADMITTIME, TIMESTAMPDIFF(HOUR, a.ADMITTIME, c.CHARTTIME) AS 'time'
# FROM
# (
# 	(
# 		SELECT *
# 		FROM `FILTERED_ADMISSIONS`
# 	) a
# 	INNER JOIN
# 	(
# 		SELECT *
# 		FROM `FILTERED_CHARTEVENTS`
# 		WHERE VALUENUM IS NOT NULL
# 	) c
# 	ON (a.HADM_ID = c.HADM_ID) AND (a.SUBJECT_ID = c.SUBJECT_ID)
# )
# ORDER BY time
# ""","""
# CREATE INDEX main_idx ON TIMED_CHARTEVENTS(SUBJECT_ID,HADM_ID,ITEMID,time)
# """,]

row = [-1,-1]
hour = -1
queries = ["""
SELECT
    `time`,
    `STAGE`
FROM `cohort_hour_staged_nonzero`
WHERE SUBJECT_ID=%d AND HADM_ID=%d AND time BETWEEN %d AND %d
""" % (row[0], row[1], int(hour) - 1, 47 + int(hour))
]

# queries = None
nargs = len(sys.argv)
if queries != None or nargs > 1:
    cfgs = None
    with open('.dbconfig.json') as cf:
        cfgs = json.loads(cf.read())
    mydb = mysql.connector.connect(**cfgs)
    cursor = mydb.cursor()

    if nargs == 2:
        if sys.argv[1] == "ps":
            queries = ["SHOW PROCESSLIST"]
    if nargs == 3:
        if sys.argv[1] == "k":
            queries = ["KILL %s" % sys.argv[2]]

    for q in queries:
        t0 = time.time()
        print(q)
        cursor.execute(q)
        l = cursor.fetchone()
        # for l in cursor:
        #     pass
        print(l)
        t1 = time.time()
        print("took %f s" % (t1 - t0))

    cursor.close()

