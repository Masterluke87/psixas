import sys
import pickle

Spec  = sys.argv[1]
Delta = sys.argv[2]


X = pickle.load(open(Spec,"rb"))
print("1st Ex(TP) : {}".format(float(X["En"][0])*27.211385))
    
E = [x.split(":")[1].split()[0] for x in open(Delta,"r").readlines() if "EXCITATION" in x][0]
print("Delta-SCF  : {}".format(float(E)*27.211385 )) 

print("Shift      : {}".format((float(E)-float(X["En"][0]))*27.211385))

