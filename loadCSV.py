import os.path as path
import mysql.connector, sys, json, time
from constants import *

with open(path.join(PROCESSED_PATH, 'cohort_hour_staged.csv'), 'r') as csv_data:
    with open('.dbconfig.json') as cf:
        cfgs = json.loads(cf.read())
        mydb = mysql.connector.connect(**cfgs)
        cursor = mydb.cursor()

        i = 0
        t0 = time.time()
        for row in csv_data:
            i += 1
            if i % 10000 == 0:
                print("row: ", i)
            cursor.execute(
                "INSERT INTO cohort_hour_staged(SUBJECT_ID, HADM_ID, age, ETHNICITY, GENDER, time, SCr, VALUEUOM, BASELINE, RATIO, STAGE) VALUES(%s)" % row
                )
        #close the connection to the database.
        mydb.commit()
        cursor.close()
        t1 = time.time()
        print("took %f s" % (t1 - t0))
        print("total rows: ", i)
