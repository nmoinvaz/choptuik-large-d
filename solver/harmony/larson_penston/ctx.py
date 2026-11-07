import re, sys
t = open(sys.argv[1]).read()
n = int(sys.argv[2])
for k in sys.argv[3:]:
    for m in list(re.finditer(re.escape(k), t))[:n]:
        print('>>', k, m.start(), ':', t[max(0, m.start()-300):m.start()+400]); print()
