import numpy as np
import pdb
import psi4
import pickle


def CalcSpec(mol,func):
    print ("X-Ray spectrum calculation:")
    prefix = psi4.core.get_local_option("PSIXAS","PREFIX")


    print ("Using orbitals, occupations from file: {}".format(prefix+"_exorbs.npz"))

    Ca = np.load(prefix+"_exorbs.npz")["Ca"]
    Cb = np.load(prefix+"_exorbs.npz")["Cb"]

    occa = np.load(prefix+"_exorbs.npz")["occa"]
    occb = np.load(prefix+"_exorbs.npz")["occb"]

    epsa = np.load(prefix+"_exorbs.npz")["epsa"]
    epsb = np.load(prefix+"_exorbs.npz")["epsb"]

    orbitals = np.load(prefix+"_exorbs.npz", allow_pickle=True)["orbitals"]


    print ("Occupation pattern: ")
    print (occa)
    print (occb)

    wfn   = psi4.core.Wavefunction.build(mol,psi4.core.get_global_option('BASIS'))
    mints = psi4.core.MintsHelper(wfn.basisset())

    D = mints.ao_dipole()
    Dx,Dy,Dz = np.asarray(D[0]),np.asarray(D[1]),np.asarray(D[2])

    spec = {}

    if ("b" in [x['spin'] for x in orbitals]):
        print ("BETA orbitals")
        orbI = ([c for c,x in enumerate(occb) if x != 1.0][0])
        print ("Initial Orbital: {:d}".format(orbI))
        Ci = Cb[orbI]
        orbF = ([c for c,x in enumerate(occb) if (x != 1.0) and (c!=orbI)])
        print ("Final Orbitals: "+str(orbF))
        Cf = Cb[orbF]

        Mx = Cb.T @ Dx @ Cb
        My = Cb.T @ Dy @ Cb
        Mz = Cb.T @ Dz @ Cb

        spec["En"] = epsb[orbF] - epsb[orbI]
        spec["Dx"] = Mx[orbI,orbF]
        spec["Dy"] = My[orbI,orbF]
        spec["Dz"] = Mz[orbI,orbF]

        with open(prefix+'_b.spectrum', 'wb') as handle:
            pickle.dump(spec, handle, protocol=pickle.HIGHEST_PROTOCOL)


    if ("a" in [x['spin'] for x in orbitals]):
        print ("ALFA orbitals")
