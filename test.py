from collections import OrderedDict

rp = OrderedDict.fromkeys([1,2,3,4,5], '?')
print(rp.values())

d = [
    (1,'a'),
    (2,'b'),
    (3,'c'),
    (4,'d'),
    (5,'e'),
    (3,'f'),
]

for k,v in d:
    rp[k] = v
    print(rp.values())