import os

DATA_PATH = os.path.join(os.getcwd(), 'data')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
DEFINITIONS_PATH = os.path.join(DATA_PATH, 'definitions')


TOP_50_CE_ITEM_IDS = [
    829, 811, 813, 791, 781, 837, 212, 161, 432, 184, 723, 454, 674, 617, 828,
    479, 787, 788, 478, 425, 428, 599, 593,  27, 644, 861, 707,  80, 919, 210,
    704, 814, 211, 742,  31, 833, 646, 618, 198,  84,  82,  85,  86,  83,  88,
    159, 54, 32, 87, 706
]

TOP_50_LE_ITEM_IDS = [
    50912, 51006, 51221, 51265, 51301, 51222, 51249, 51248, 51250, 51279, 51277,
    50971, 50983, 50902, 50882, 50868, 50931, 50960, 50970, 50893, 51237, 51274,
    51275, 50820, 51491, 51498, 51244, 51254, 51256, 51146, 51200, 50802, 50804,
    50821, 50818, 50813, 50861, 50878, 51492, 50885, 50863, 51484, 51514, 51478,
    50862, 51508, 51506, 51466, 51464, 51487
]
CONVERSIONS = {
    'WEIGHT':           lambda v, u: v * 0.453592 if u == 'lbs' else v, # WEIGHT
    'HEIGHT':           lambda v, u: v * 2.54     if u == 'in'  else v, # HEIGHT
    'SYSTOLIC BP':      lambda v, u: v, # SYSTOLIC BP
    'DIASTOLIC BP':     lambda v, u: v, # DIASTOLIC BP
    'TEMPERATURE':      lambda v, u: (v-32) * (5/9) if u == 'F' else v, # TEMPERATURE
    'RESPIRATORY RATE': lambda v, u: v, # RESPIRATORY RATE
    'HEART RATE':       lambda v, u: v, # HEART RATE
    'SPO2':             lambda v, u: v, # SPO2
}

ITEM_IDS_UOM = {
    'WEIGHT': {
        226512: 'kg',
        762: 'kg',
        763: 'kg',
        226531: 'lbs',
        224639: 'kg',
}, 'HEIGHT': {
        920: 'in',
        226730: 'cm',
}, 'SYSTOLIC BP': {
        455: 'mmHg',
        51: 'mmHg',
        225309: 'mmHg',
        442: 'mmHg',
        6701: 'mmHg',
}, 'DIASTOLIC BP': {
        8441: 'mmHg',
        8368: 'mmHg',
        225310: 'mmHg',
        8440: 'mmHg',
        8555: 'mmHg',
}, 'TEMPERATURE': {
        678: 'F',
        677: 'C',
        223761: 'F',
        679: 'F',
        676: 'C',
        # 645: ,
        # 591: ,
        223762: 'C',
        226329: 'C',
        # 597: ,
        227054: 'F',
}, 'RESPIRATORY RATE': {
        618: 'bpm',
        220210: 'bpm',
        619: 'bpm',
        224689: 'bpm',
        224690: 'bpm',
        224688: 'bpm',
}, 'HEART RATE': {
        211: 'bpm',
        220045: 'bpm',
}, 'SPO2': {
        646: '%',
        6719: '%',
        220277: '%',
    #}, 'BUN': {
    #     225624: 'mg/dl',
    #     1162: 'mg/dl',
    #}, 'CREATININE': {
    #     220615: 'mg/dl',
    #     1525: 'mg/dl',
    #}, 'GLUCOSE': {
    #     220621: 'mg/dl',
    #     1529: 'mg/dl',
    #     # 807: ,
    #     # 225664: ,
    #     226537: 'mg/dl',
    #}, 'ALBUMIN': {
    #     227456: 'g/dl',
    #     1521: ,
    #}, 'CARBON DIOXIDE': {
    #     787: 'mg/dl',
    #}, 'CALCIUM': {
    #     225625: 'mg/dl',
    #     1522: 'mg/dl',
    #}, 'IONIZED CALCIUM': {
    #     816: ,
    #     225667: 'mmol/l',
    #}, 'SODIUM
    #     220645: 'mEg/l',
    #     1536: ,
    #     226534: 'mEg/l',
    #}, 'POTASSIUM
    #     227442: 'mEg/l',
    #     1535: ,
    #     227464: 'mEg/l',
    #}, 'CHLORIDE
    #     220602: 'mEg/l',
    #     1523: ,
    #     226536: 'mEg/l',
    #}, 'TOTAL BILIRUBIN': {
    #     225690: 'mg/dl',
    #     1538: ,
    #}, 'DIRECT BILIRUBIN': {
    #     225651: 'mg/dl',
    #     1527: ,
    #}, 'ALT': {
    #     769: ,
    #     220644: 'iu/l',
    #}, 'ALKALINE PHOSPHATE': {
    #     773: ,
    #     225612: 'iu/l',
    #}, 'AST': {
    #     770: ,
    #     220587: 'iu/l',
    #}, 'HEMOGLOBIN': {
    #     814: 'gm/dl',
    #     220228: 'g/dl',
    #}, 'PLATELETS': {
    #     828: ,
    #     227457: 'k/ul',
    #}, 'WBC': {
    #     220546: 'k/ul',
    #     1542: 'k/ul',
    #}, 'AMMONIA': {
    #     220580: ,
    #     1060: ,
    #     774: ,
    #     2687: ,
    #}, 'CK': {
    #     225634: ,
    #     227445: ,
    #     225628: ,
    #}, 'LIPASE': {
    #     225672: ,
    #     820: ,
    #     2412: ,
    #}, 'TROPONIN': {
    #     227429: ,
    #     851: ,
    #}, 'ANION GAP': {
    #     227073: ,
    #}, 'BNP': {
    #     227446: ,
    #     7294: ,
    }
}

CE_ITEM_LABELS = [
    'WEIGHT',
    'HEIGHT',
    'SYSTOLIC BP',
    'DIASTOLIC BP',
    'TEMPERATURE',
    'RESPIRATORY RATE',
    'HEART RATE',
    'SPO2',
]