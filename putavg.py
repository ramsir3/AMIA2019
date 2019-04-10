import csv

out = []
with open('labelandavglabs.csv', 'r') as outf:
    with open('data/processed/avglabs.csv', 'r') as inf:
        cof = csv.reader(outf)
        cif = csv.reader(inf)
        avgs = {int(r[0]): r[1] for r in cif}
        
        for r in cof:
            r.append(avgs.get(int(r[0]), 'nan'))
            out.append(r)

with open('labelandavglabsout.csv', 'w') as outf:
    csv.writer(outf).writerows(out)


'51506'

