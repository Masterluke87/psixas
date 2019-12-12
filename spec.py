import numpy as np
import pdb
import psi4
import pickle


def CalcSpec(mol,func):
    psi4.core.print_out("\n\nX-Ray Absorption Spectrum Calculation:\n"+38*"="+"\n\n")
    prefix = psi4.core.get_local_option("PSIXAS","PREFIX")


    psi4.core.print_out("Using orbitals, occupations from file: {}  \n".format(prefix+"_exorbs.npz"))

    Ca = np.load(prefix+"_exorbs.npz")["Ca"]
    Cb = np.load(prefix+"_exorbs.npz")["Cb"]

    occa = np.load(prefix+"_exorbs.npz")["occa"]
    occb = np.load(prefix+"_exorbs.npz")["occb"]

    epsa = np.load(prefix+"_exorbs.npz")["epsa"]
    epsb = np.load(prefix+"_exorbs.npz")["epsb"]

    orbitals = np.load(prefix+"_exorbs.npz", allow_pickle=True)["orbitals"]


    psi4.core.print_out("Occupation pattern: \n")

    printOccupation("Alpha",occa,15)
    printOccupation("Beta ",occb,15)

    psi4.core.be_quiet()
    wfn   = psi4.core.Wavefunction.build(mol,psi4.core.get_global_option('BASIS'))
    mints = psi4.core.MintsHelper(wfn.basisset())
    psi4.core.reopen_outfile()

    D = mints.ao_dipole()
    Dx,Dy,Dz = np.asarray(D[0]),np.asarray(D[1]),np.asarray(D[2])

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


        with open(prefix+'_b.spectrum', 'wb') as handle:
            pickle.dump(spec, handle, protocol=pickle.HIGHEST_PROTOCOL)
        psi4.core.print_out(("\n{}"+"_b.spectrum written.. \n\n").format(prefix))

    if ("a" in [x['spin'] for x in orbitals]):
        print ("ALFA orbitals")

def printOccupation(title,occs,width):
    psi4.core.print_out("\n{}: \n".format(title))
    for i in range(int(len(occs)/width)):
        psi4.core.print_out(("|"+"{:^4}|"*width).format(*[x for x in range(i*width,i*width+ width)]))
        psi4.core.print_out(("\n|"+"{:3.2f}|"*width+"\n\n").format(*[x for x in occs[i*width:i*width+width]]))
    
    psi4.core.print_out(("|"+"{:^4}|"*(len(occs) % width)).format(*[x for x in range(int(len(occs)/width)*width,len(occs))]))
    psi4.core.print_out(("\n|"+"{:3.2f}|"*(len(occs) % width)+"\n\n").format(*[x for x in occs[int(len(occs)/width)*width:]]))

 


