#!/usr/bin/env python3

import mysql.connector, json, time, sys

query = """
EXPLAIN
SELECT SUBJECT_ID, HADM_ID, ITEMID, time, COUNT(VALUENUM) AS 'count', AVG(VALUENUM) AS 'average', MIN(VALUENUM) AS 'min', MAX(VALUENUM) AS 'max', VALUEUOM
FROM
TIMED_LABEVENTS
GROUP BY SUBJECT_ID, HADM_ID, ITEMID, time
"""
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
# CREATE TABLE AVG_TIMED_LABEVENTS
# SELECT SUBJECT_ID, HADM_ID, ITEMID, time, COUNT(VALUENUM) AS 'count', AVG(VALUENUM) AS 'average', MIN(VALUENUM) AS 'min', MAX(VALUENUM) AS 'max', VALUEUOM
# FROM
# TIMED_LABEVENTS
# GROUP BY SUBJECT_ID, HADM_ID, ITEMID, time
# ""","""
# CREATE INDEX main_idx ON TIMED_LABEVENTS(SUBJECT_ID,HADM_ID,ITEMID,time)
# """,]

# queries = ["""
# CREATE INDEX ale_main_idx ON AVG_TIMED_LABEVENTS(SUBJECT_ID,HADM_ID,ITEMID,time)
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
# CREATE INDEX ce_main_idx ON TIMED_CHARTEVENTS(SUBJECT_ID,HADM_ID,ITEMID,time)
# ""","""
# CREATE TABLE AVG_TIMED_CHARTEVENTS
# SELECT SUBJECT_ID, HADM_ID, ITEMID, time, COUNT(VALUENUM) AS 'count', AVG(VALUENUM) AS 'average', MIN(VALUENUM) AS 'min', MAX(VALUENUM) AS 'max', VALUEUOM
# FROM
# TIMED_CHARTEVENTS
# GROUP BY SUBJECT_ID, HADM_ID, ITEMID, time
# ""","""
# CREATE INDEX ace_main_idx ON AVG_TIMED_CHARTEVENTS(SUBJECT_ID,HADM_ID,ITEMID,time)
# """,]

queries = ["""
CREATE TABLE AVG_TIMED_CHARTEVENTS
SELECT SUBJECT_ID, HADM_ID, ITEMID, time, COUNT(VALUENUM) AS 'count', AVG(VALUENUM) AS 'average', MIN(VALUENUM) AS 'min', MAX(VALUENUM) AS 'max', VALUEUOM
FROM
TIMED_CHARTEVENTS
GROUP BY SUBJECT_ID, HADM_ID, ITEMID, time
""","""
CREATE INDEX ace_main_idx ON AVG_TIMED_CHARTEVENTS(SUBJECT_ID,HADM_ID,ITEMID,time)
""",]

# queries = [query]
queries = None
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
        for l in cursor:
            print(l)
        t1 = time.time()
        print("took %f s" % (t1 - t0))

    cursor.close()

