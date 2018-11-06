from filterTables import *
from constants import *

files = [('CHARTEVENTS.csv', 'D_ITEMS.csv'), ('LABEVENTS.csv', 'D_LABITEMS.csv')]
countFiles = [('FILTERED_CHARTEVENTS.csv', 'ITEMCOUNTS_FILTERED_CHARTEVENTS.csv'), ('FILTERED_LABEVENTS.csv', 'ITEMCOUNTS_FILTERED_LABEVENTS.csv')]
ids = getIDs(path.join(PROCESSED_PATH, 'mimic3ScrBaseline.csv'))

data, defs = files[1] #select the data you want
ff = filterOnIDs(path.join(RAW_PATH, data), ids)
countItems(ff, path.join(DEFINITIONS_PATH, defs), icol=3, dIncCols=[2,3,4])

# data, counts = countFiles[1] #select the data you want
# getPatientItemStats(path.join(PROCESSED_PATH, counts), path.join(PROCESSED_PATH, data), icol=3, minObv=0)