from collections import Counter
from matplotlib 

c = Counter()
with open('p.csv') as f:
	for line in f:
		c[int(line)] += 1

print(c)