import json
import sys



out=json.load(open(sys.argv[1],"r"))

f = open("output.molden","w")

for i in out["Molden"]:
    f.write(i)
f.close()


del out["Molden"]

json.dump(out,open("output.json","w"),indent=4)


