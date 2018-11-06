# (
#     CASE
#     WHEN (scr_info.ETHNICITY LIKE 'BLACK%') AND (scr_info.GENDER = 'F') THEN ROUND(POW(75/(0.742*1.21*186*ROUND(POW(scr_info.age, -0.203), 5)), -1/1.154), 5)
#     WHEN scr_info.GENDER = 'F' THEN ROUND(POW(75/(0.742*186*ROUND(POW(scr_info.age, -0.203), 5)), -1/1.154), 5)
#     WHEN scr_info.ETHNICITY LIKE 'BLACK%' THEN ROUND(POW(75/(1.21*186*ROUND(POW(scr_info.age, -0.203), 5)), -1/1.154), 5)
#     ELSE ROUND(POW(75/(186*ROUND(POW(scr_info.age, -0.203), 5)), -1/1.154), 5)
#     END
# ) AS 'baseline_Scr'

def calcBaseline(age, ethnicity, sex, scr):
	# print(age, ethnicity, sex, scr)
	s = 0.742 if sex == "F" else 1
	e = 1.21 if "BLACK" in ethnicity else 1
	bl = (75/(s*e*186*(float(age)**-0.203)))**-(1/1.154)
	return bl, float(scr)/bl

def run(fnin, fnout, debug=False):
	with open(fnin) as fin:
		with open(fnout, 'w') as fout:
			i = 0
			for line in fin:
				i += 1
				line = line.strip()
				if debug: print(line)
				sline = line.replace("\"",'').split(',')

				if debug: print(sline)
				# try:
				bl, r = calcBaseline(sline[2], sline[3], sline[4], sline[6])
				line = "%s,\"%f\",\"%f\"\n"%(line,bl,r)
				if debug: print(line)
				fout.write(line)
				if debug and i%100 == 0:
					print(i)
				# except:
				# 	print("error on line", i, ":")
				# 	print(line)
				
				if debug and i == 10:
					break
			print("total lines:", i)

run('./data/raw/mimic3Scr.csv', './data/processed/mimic3ScrBaseline.csv')
# run('./data/raw/mimic3Scr.csv', './data/processed/mimic3ScrBaseline.csv', debug=True)