from collections import defaultdict


uniqueEncPos = set()
uniqueEncNeg = set()
stats = defaultdict(lambda: defaultdict(int))

i = 0
with open('./data/processed/mimic3ScrBaseline.csv') as fin:
	for line in fin:
		i += 1
		line = line.strip()
		# print(line)
		sline = line.replace("\"",'').split(',')
		# print(sline)
		try:
			if float(sline[11]) >= 1.5:
				uniqueEncPos.add( (sline[0],sline[1]) )
				stats[tuple(sline[:3])]['pos'] += 1
			else:
				uniqueEncNeg.add( (sline[0],sline[1]) )
				stats[tuple(sline[:3])]['neg'] += 1

			# if i%100 == 0:
			# 	print(i)
		except:
			print("error on line:", i)
		# if i == 10:
		# 	break
	print(i)

with open('./data/processed/mimic3HighScrEnc.csv', 'w') as fout:
	for k, v in stats.items():
		line = "%s,%s,%s,%d,%d\n"%(k[0],k[1],k[2],v['pos'],v['neg'])
		fout.write(line)

print(len(uniqueEncPos))
print(len(uniqueEncNeg))
