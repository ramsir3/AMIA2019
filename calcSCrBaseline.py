
def calcBaseline(age, ethnicity, sex, scr):
	# print(age, ethnicity, sex, scr)
	s = 0.742 if sex == "F" else 1
	e = 1.21 if "BLACK" in ethnicity else 1
	bl = (75/(s*e*186*(float(age)**-0.203)))**-(1/1.154)
	r = float(scr)/bl
	stage = 0
	if r >= 3.0:
		stage = 3
	elif r >= 2.0:
		stage = 2
	elif r >= 1.5:
		stage = 1
	return bl, r, stage

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
				bl, r, stage = calcBaseline(sline[2], sline[3], sline[4], sline[6])
				line = "%s,\"%f\",\"%f\",\"%d\"\n"%(line,bl,r,stage)
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

run('./data/raw/cohort_hour.csv', './data/processed/cohort_hour_staged.csv')
# run('./data/raw/mimic3Scr.csv', './data/processed/mimic3ScrBaseline.csv', debug=True)