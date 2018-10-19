
uniqueEncPos = set()
uniqueEncNeg = set()

i = 0
with open('./data/processed/mimic3ScrBaseline.csv') as fin:
	with open('./data/processed/mimic3HighScrEnc.csv', 'w') as fout:
		for line in fin:
			i += 1
			line = line.strip()
			# print(line)
			sline = line[1:-1].split("\",\"")
			# print(sline)
			try:
				if float(sline[10]) >= 1.5:
					uniqueEnc.add((sline[0],sline[1]))
					line = "%s\n"%line
					fout.write(line)
				if i%100 == 0:
					print(i)
			except:
				print("error on line", i)

print(len(uniqueEnc))