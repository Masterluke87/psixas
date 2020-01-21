import numpy as np
import psi4
import pickle
import os
from .kshelper import printHeader

def CalcSpec(mol,func):
    printHeader("X-Ray Absorption Spectrum Calculation")
    Files = [] 
    prefix = psi4.core.get_local_option("PSIXAS","PREFIX")

    if os.path.exists(prefix+"_exorbs.npz"):
        Files.append(prefix+"_exorbs.npz")
    if os.path.exists(prefix+"_exorbsAug.npz"):
        Files.append(prefix+"_exorbsAug.npz")

    psi4.core.print_out("\nFound {} files with orbital information".format(len(Files)))

    for i in Files:
        psi4.core.print_out("\nUsing orbitals, occupations from file: {}  \n\n".format(i))
        Ca = np.load(i)["Ca"]
        Cb = np.load(i)["Cb"]
        occa = np.load(i)["occa"]
        occb = np.load(i)["occb"]
        epsa = np.load(i)["epsa"]
        epsb = np.load(i)["epsb"]
        orbitals = np.load(i, allow_pickle=True)["orbitals"]

        psi4.core.print_out("Occupation pattern: \n")
        printOccupation("Alpha",occa,15)
        printOccupation("Beta ",occb,15)

        D = np.load(i)["D"]
        Dx,Dy,Dz = D[0],D[1],D[2]
        spec = {}

        if ("b" in [x['spin'] for x in orbitals]):
            psi4.core.print_out("\nBETA orbitals")
            orbI = ([c for c,x in enumerate(occb) if x != 1.0][0])
            psi4.core.print_out("\nInitial Orbital: {:d}".format(orbI))
            Ci = Cb[orbI]
            orbF = ([c for c,x in enumerate(occb) if (x != 1.0) and (c!=orbI)])
            psi4.core.print_out("\nFinal Orbitals: {} \n\n".format(str(orbF)))
            Cf = Cb[orbF]

            Mx = Cb.T @ Dx @ Cb
            My = Cb.T @ Dy @ Cb
            Mz = Cb.T @ Dz @ Cb

            spec["En"] = epsb[orbF] - epsb[orbI]
            spec["Dx"] = Mx[orbI,orbF]
            spec["Dy"] = My[orbI,orbF]
            spec["Dz"] = Mz[orbI,orbF]


            psi4.core.print_out("\nTransition-Potential excitation energies and transition dipole moments\n")
            psi4.core.print_out("\n{:>16} {:>3}->{:>3} {:>16} {:>16} {:>16} \n".format("Energy [eV]","i","f","<i|x|f>","<i|y|f>","<i|z|f>"))
            psi4.core.print_out("="*(16*4+7*2+5)+"\n")

            for e,f,x,y,z in zip(spec["En"],orbF,spec["Dx"],spec["Dy"],spec["Dz"]):
                psi4.core.print_out("{:>16.8f} {:>3}->{:>3} {:>16.8f} {:>16.8f} {:>16.8f} \n".format(e*27.211386,str(orbI),str(f),x,y,z))

            if "Aug" in i:
                with open(prefix+'_bAug.spectrum', 'wb') as handle:
                    pickle.dump(spec, handle, protocol=pickle.HIGHEST_PROTOCOL)
                psi4.core.print_out(("\n{}"+"_bAug.spectrum written.. \n\n").format(prefix))
            else:
                with open(prefix+'_b.spectrum', 'wb') as handle:
                    pickle.dump(spec, handle, protocol=pickle.HIGHEST_PROTOCOL)
                psi4.core.print_out(("\n{}"+"_b.spectrum written.. \n\n").format(prefix))

        if ("a" in [x['spin'] for x in orbitals]):
            raise Exception("Alpha orbitals not yet implemented, please use beta orbitals")

def printOccupation(title,occs,width):
    psi4.core.print_out("\n{}: \n".format(title))
    for i in range(int(len(occs)/width)):
        psi4.core.print_out(("|"+"{:^4}|"*width).format(*[x for x in range(i*width,i*width+ width)]))
        psi4.core.print_out(("\n|"+"{:3.2f}|"*width+"\n\n").format(*[x for x in occs[i*width:i*width+width]]))
    
    psi4.core.print_out(("|"+"{:^4}|"*(len(occs) % width)).format(*[x for x in range(int(len(occs)/width)*width,len(occs))]))
    psi4.core.print_out(("\n|"+"{:3.2f}|"*(len(occs) % width)+"\n\n").format(*[x for x in occs[int(len(occs)/width)*width:]]))

 


