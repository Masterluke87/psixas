import json
import sys
import pickle
import copy

def findShift(spectrum, excitation):
    """
    Finds the shift, if the given excitation energy is equal to 0.0
    the shift is also zero
    """
    if excitation==0.0:
        return 0.0

    f = pickle.load(open(spectrum,"rb"))
    firstExcitation =  f["En"][0]*27.211385
    print("1st Exc TP-DFT : {} [eV]".format(firstExcitation))
    shift = (excitation-firstExcitation)
    return shift

def findMoldenHeader(moldenFile):
    Molden = open(moldenFile).readlines()
    idx = [c for c,x in enumerate(Molden) if "[MO]" in x][0]
    header = Molden[:idx]
    header.append("[MO]\n")
    return header
 
 

def findCoreOrbital(moldenFile):
    """
    Find one the corebital (occupation less than 1.0, but greater than 0.0)
    """
    Molden = open(moldenFile).readlines()
    idx = [c for c,x in enumerate(Molden) if "[MO]" in x][0]
    moPart = Molden[idx+1:]
    idx = [c for c,x in enumerate(moPart) if "Sym" in x]
    coreOrb = ""
    for i in range(1,len(idx)):
        X = "".join((moPart[idx[i-1]:idx[i]]))
        if "Beta" in X:
            X2 = X.split("\n")
            occ = float([x.split("=")[1] for x in X2 if "Occup=" in x][0])

            if ((occ>0.0) and (occ<1.0)):
                coreEn = float([x.split("=")[1] for x in X2 if "Ene=" in x][0])
                coreOrb = X
    return coreOrb,coreEn
    
    




inputf  = json.load(open(sys.argv[1],"r"))

print("Number of excitation centers: {}".format(len(inputf["center"])))

for i in inputf["center"]:
    print("Center: {}".format(i["label"]))
    print("Number of Spectra: {}".format(len(i["spectra"])))
    i["CoreOrb"],i["CoreEn"] =  findCoreOrbital(i["spectra"][0]["molden"])
    for c,j in enumerate(i["spectra"]):
        print("\nSpectrum {} at center {}:".format(c,i["label"]))
        print("Molden File    : {}".format(j["molden"]))
        print("Spectrum File  : {}".format(j["spectrum"]))
        print("Type           : {}".format(j["type"]))
        print("1st Exc D-KS   : {} [eV]".format(j["excitaion"]))
        j["shift"] = findShift(j["spectrum"],j["excitaion"])
        j["core"] = float(findCoreOrbital(j["molden"])[1])*27.211385
        print("Shift          : {} [eV]".format(j["shift"]))
        print("Core Orbital E : {} [eV]".format(j["core"]))

"""
Now we have to sort the excitation centers wrt. to energies and also give them
an id...
"""
inputf["center"] = sorted(inputf["center"], key = lambda x: x["CoreEn"])
counter = 0
for i in inputf["center"]:
    for j in i["spectra"]:
        j["ID"] = counter
        counter = counter+1
    

"""
Grep the molden Header
"""
print("Takeing moleden header from:{}".format(inputf["center"][0]["spectra"][0]["molden"]))
inputf["moldenHeader"] = findMoldenHeader(inputf["center"][0]["spectra"][0]["molden"])

"""
1. Iterate over all centers and find the empty Beta orbitals.
2. Set their energies to the exitation energies and apply the shift
"""

Orbs = []
for i in inputf["center"]:
   for j in i["spectra"]:
       Molden = open(j["molden"]).readlines()
       idx = [c for c,x in enumerate(Molden) if "[MO]" in x][0]
       moPart = Molden[idx+1:]
       idx = [c for c,x in enumerate(moPart) if "Sym" in x]
       for k in range(1,len(idx)):
           X = "".join((moPart[idx[k-1]:idx[k]]))
           if "Beta" in X:
               X2 = X.split("\n")
               occ = float([x.split("=")[1] for x in X2 if "Occup=" in x][0])
               if (occ==0.0):
                   En = float([x.split("=")[1] for x in X2 if "Ene=" in x][0])
                   for c,l in enumerate(X2):
                       if "Ene=" in l:
                           X2[c]=" Ene="+str((En*27.211385-j["core"])+float(j["shift"]))
                       if "Sym=" in l:
                           X2[c]=" Sym="+str(j["ID"])
      
                   Orbs.append("\n".join(X2))
       
       X = "".join((moPart[idx[k]:]))
       if "Beta" in X:
           X2 = X.split("\n")
           occ = float([x.split("=")[1] for x in X2 if "Occup=" in x][0])
           if (occ ==0.0):
               En = float([x.split("=")[1] for x in X2 if "Ene=" in x][0])
               for c,l in enumerate(X2):
                   if "Ene=" in l:
                       X2[c]=" Ene="+str((En*27.211385-j["core"])+float(j["shift"]))
                   if "Sym=" in l:
                       X2[c]=" Sym="+str(j["ID"])
               Orbs.append("\n".join(X2))
       

inputf["Molden"] = copy.deepcopy(inputf["moldenHeader"])
X = ([x.split("\n") for x in Orbs])
print(X)
for i in inputf["center"]:
    inputf["Molden"].extend([x+"\n" for x in i["CoreOrb"].split("\n") if x!=""])
for i in X:
    inputf["Molden"].extend([x+"\n" for x in i if x!=""])

json.dump(inputf,open("output.json","w"),indent=4)
data = {} 

f = open("output.molden","w")
for i in inputf["Molden"]:
    f.write(i)







"""
print("Number of CMD-line arguments: {}".format(len(sys.argv)-1))

if (((len(sys.argv)-1) % 3)!=0):
    print("Wrong number of arguments! It should be 3*N")
    exit()

N = int((len(sys.argv)-1)/3)


data = {}
data["center"] = []
data["spec"] = []

for x in range(N):
    print("Center: {}\n".format(x)+9*"=")
    print("SPEC  : {}\nMOLDEN: {}\n1stExc: {}".format(sys.argv[3*x+1],sys.argv[3*x+2],sys.argv[3*x+3]))
    X = pickle.load(open(sys.argv[3*x+1] ,"rb"))
    shift=(float(sys.argv[3*x+3])-float(X["En"][0])*27.211385)
    print("SHIFT : {}\n".format(shift))
    data["center"].append({"spectrum" :sys.argv[3*x+1],
                           "molden":sys.argv[3*x+2],
                           "excitation":sys.argv[3*x+3],
                           "shift" : shift})


Orbs = []

for center in data["center"]:
    X = pickle.load(open(center["spectrum"],"rb"))
    Molden = open(center["molden"]).readlines()
    idx = [c for c,x in enumerate(Molden) if "[MO]" in x][0]
    header = Molden[:idx]
    header.append("[MO]\n")
    moPart = Molden[idx+1:]
    idx = [c for c,x in enumerate(moPart) if "Sym" in x]

    for i in range(1,len(idx)):
        X = "".join((moPart[idx[i-1]:idx[i]]))
        if "Beta" in X:
            X2 = X.split("\n")
            occ = float([x.split("=")[1] for x in X2 if "Occup=" in x][0])

            if ((occ>0.0) and (occ<1.0)):
                coreEn = float([x.split("=")[1] for x in X2 if "Ene=" in x][0])
                Orbs.append(X)
    
            if (occ==0.0):
                En = float([x.split("=")[1] for x in X2 if "Ene=" in x][0])
                for c,j in enumerate(X2):
                    if "Ene=" in j:
                        X2[c]=" Ene="+str((En-coreEn)*27.211385+float(center["shift"]))
    
                Orbs.append("\n".join(X2))

    X = "".join((moPart[idx[i]:]))
    if "Beta" in X:
        X2 = X.split("\n")
        occ = float([x.split("=")[1] for x in X2 if "Occup=" in x][0])
        if (occ ==0.0):
            En = float([x.split("=")[1] for x in X2 if "Ene=" in x][0])
            for c,j in enumerate(X2):
                if "Ene=" in j:
                    X2[c]=" Ene= "+str((En-coreEn)*27.211385)
            Orbs.append("\n".join(X2))
    center["CoreEnergy"] = coreEn


data["center"] = sorted(data["center"], key=lambda x: x['CoreEnergy'])
for c,i in enumerate(data["center"]):
    i["ID"] = c
    

f = open("spec.molden","w")
f.write("".join(header))
for i in Orbs:
    f.write(i)
f.close()


for spectra in data["center"]:
    X = pickle.load(open(spectra["spectrum"],"rb"))
    for E,X,Y,Z in zip(X["En"],X["Dx"],X["Dy"],X["Dz"]):
        data["spec"].append({"ID" : spectra["ID"],
                             "shift" : float(spectra["shift"]),
                             "En" : float(E*27.211385),
                             "Dx" : float(X),
                             "Dy" : float(Y),
                             "Dz" : float(Z),
                             "Tot": float(X**2+Y**2+Z**2)})


data["spec"] = sorted(data["spec"], key=lambda x: x['En'])

with open("spec.json","w") as outfile:
    json.dump(data,outfile,indent=4)
"""



