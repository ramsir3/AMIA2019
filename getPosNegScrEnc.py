from collections import defaultdict
import json

uniqueEncPos = set()
uniqueEncNeg = set()
counts = defaultdict(lambda: defaultdict(int))

e = u = i = 0
with open('./data/processed/mimic3ScrBaseline.csv') as fin:
	for line in fin:
		i += 1
		line = line.strip()
		# print(line)
		sline = line.replace("\"",'').split(',')
		# print(sline)
		idcols = tuple(sline[:2])
		scrbr = float(sline[9])
		# try:
		if float(int(sline[2])) >= 18:
			if scrbr >= 1.5:
				uniqueEncPos.add( (sline[0],sline[1]) )
				counts[idcols]['pos'] += 1
				if scrbr >= 3.0:
					counts[idcols]['stage3'] += 1
				elif scrbr >= 2.0:
					counts[idcols]['stage2'] += 1
				else:
					counts[idcols]['stage1'] += 1
			else:
				uniqueEncNeg.add( (sline[0],sline[1]) )
				counts[idcols]['neg'] += 1
		else:
			u += 1
			# if i%100 == 0:
			# 	print(i)
		# except:
		# 	print("error on line:", i)
		# 	e += 1
		# if i == 10:
		# 	break
	print(i, u)

with open('./data/processed/mimic3ScrEnc.csv', 'w') as fout:
	fout.write("subject_id,hadm_id,neg_aki,pos_aki,stage1,stage2,stage3\n")
	for k, v in counts.items():
		line = "%s,%s,%d,%d,%d,%d,%d\n" \
			%(k[0],k[1],v['neg'],v['pos'],v['stage1'],v['stage2'],v['stage3'])
		fout.write(line)

# countvalues = counts.values()
# stats = {
# 	'total enc': len([key for key in counts]),
# 	'num icu': len([key for key in counts if key[2] != 'NULL']),
# 	'num non icu': len([key for key in counts if key[2] == 'NULL']),
# 	'num enc w/ pos': len([cv for cv in countvalues if cv['pos'] != 0]),
# 	'num enc w/o pos': len([cv for cv in countvalues if cv['pos'] == 0]),
# 	'num enc w/ neg': len([cv for cv in countvalues if cv['neg'] != 0]),
# 	'num enc w/o neg': len([cv for cv in countvalues if cv['neg'] == 0]),	
# }

# with open('./data/processed/mimic3ScrEncStats.json', 'w') as fj:
# 	fj.write(json.dumps(stats))

print(len(uniqueEncPos))
print(len(uniqueEncNeg))
