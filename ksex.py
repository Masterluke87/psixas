# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 18:30:31 2018

@author: luke

Module to perform excited state calculations
"""
from .kshelper import diag_H,Timer, ACDIIS,printHeader
import numpy as np
import os
import psi4
import time
import logging
import psi4.driver.qcdb as qcdb
import itertools
import copy
import pickle

def DFTExcitedState(mol,func,orbitals,**kwargs):
    """
    Perform unrestrictred Kohn-Sham excited state calculation
    """
    printHeader("Entering Excited State Kohn-Sham",1)

    options = {
        "PREFIX"    : psi4.core.get_local_option("PSIXAS","PREFIX"),
        "E_CONV"    : float(psi4.core.get_local_option("PSIXAS","E_EX_CONV")),
        "D_CONV"    : float(psi4.core.get_local_option("PSIXAS","D_EX_CONV")),
        "MAXITER"   : int(psi4.core.get_local_option("PSIXAS","MAXITER")), 
        "BASIS"     : psi4.core.get_global_option('BASIS'),
        "GAMMA"     : float(psi4.core.get_local_option("PSIXAS","DAMP")),
        "VSHIFT"    : float(psi4.core.get_local_option("PSIXAS","VSHIFT")),
        "DIIS_LEN"  : int(psi4.core.get_local_option("PSIXAS","DIIS_LEN")),
        "DIIS_MODE" : psi4.core.get_local_option("PSIXAS","DIIS_MODE"),
        "DIIS_EPS"  : float(psi4.core.get_local_option("PSIXAS","DIIS_EPS")),
        "MIXMODE"   : "DAMP",
        "USE_AUG"   : psi4.core.get_local_option("PSIXAS","USE_AUG")  } 

    """
    STEP 1: Read in ground state orbitals or restart from previous
    """
    if (os.path.isfile(options["PREFIX"]+"_exorbs.npz")):
        psi4.core.print_out("\nRestarting Calculation from: {}".format(options["PREFIX"]+"_exorbs.npz")) 
        Ca = np.load(options["PREFIX"]+"_exorbs.npz")["Ca"]
        Cb = np.load(options["PREFIX"]+"_exorbs.npz")["Cb"]
    else:
        Ca = np.load(options["PREFIX"]+"_gsorbs.npz")["Ca"]
        Cb = np.load(options["PREFIX"]+"_gsorbs.npz")["Cb"]

    """
    Grep the coefficients for later overlap
    """
    for i in orbitals:
        if i["spin"]=="b":
            i["C"] = Cb[:,i["orb"]]
        elif i["spin"]=="a":
            i["C"] = Ca[:,i["orb"]]
        else:
            raise Exception("Orbital has non a/b spin!")

    printHeader("Basis Set:",2)
    wfn   = psi4.core.Wavefunction.build(mol,options["BASIS"])
    aux     = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", psi4.core.get_global_option('BASIS'))

    psi4.core.be_quiet()
    mints = psi4.core.MintsHelper(wfn.basisset())

    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = np.zeros((mints.nbf(),mints.nbf()))

    H = T+V
    if wfn.basisset().has_ECP():
        ECP = mints.ao_ecp()
        H += ECP                

    A = mints.ao_overlap()
    A.power(-0.5,1.e-16)
    A = np.asarray(A)

    Enuc = mol.nuclear_repulsion_energy()
    Eold = 0.0
    SCF_E = 100.0
    nbf    = wfn.nso()
    nalpha = wfn.nalpha()
    nbeta  = wfn.nbeta()

    Va = psi4.core.Matrix(nbf,nbf)
    Vb = psi4.core.Matrix(nbf,nbf)

    sup = psi4.driver.dft.build_superfunctional(func, False)[0]
    sup.set_deriv(2)
    sup.allocate()

    Vpot = psi4.core.VBase.build(wfn.basisset(), sup, "UV")
    Vpot.initialize()
   
    uhf   = psi4.core.UHF(wfn,sup)  #This object is needed to write out a molden file later
    psi4.core.reopen_outfile()
    """
    Form initial denisty
    """
    occa = np.zeros(Ca.shape[0])
    occb = np.zeros(Cb.shape[0])

    occa[:nalpha] = 1
    occb[:nbeta]  = 1

    Cocca = psi4.core.Matrix(nbf, nbf)
    Coccb = psi4.core.Matrix(nbf, nbf)

    Cocca.np[:]  = Ca
    Coccb.np[:]  = Cb

    for i in orbitals:
        if i["spin"]=="b":
            """
            Check if this is still the largest overlap
            """
            ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
            if i["orb"] != np.argmax(ovl):
                print ("index changed from {:d} to {:d}".format(i["orb"],np.argmax(ovl)))
                i["orb"] = np.argmax(ovl)

            """
            Set occupation and overlap
            """
            i["ovl"] = np.max(ovl)
            occb[i["orb"]] = i["occ"]


        elif i["spin"]=="a":
            """
            Check if this is still the largest overlap
            """
            ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Cocca,S))
            if i["orb"] != np.argmax(ovl):
                print ("index changed from {:d} to {:d}".format(i["orb"],np.argmax(ovl)))
                i["orb"] = np.argmax(ovl)
            """
            Set occupation and overlap
            """
            i["ovl"] = np.max(ovl)
            occa[i["orb"]] = i["occ"]

    for i in range(nbf):
        Cocca.np[:,i] *= np.sqrt(occa[i])
        Coccb.np[:,i] *= np.sqrt(occb[i])


    Da     = Cocca.np @ Cocca.np.T
    Db     = Coccb.np @ Coccb.np.T
    
    printHeader("Molecule:",2)
    mol.print_out()
    printHeader("XC & JK-Info:",2)
    jk = psi4.core.JK.build(wfn.basisset(),aux,jk_type=psi4.core.get_local_option("SCF","SCF_TYPE"))
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    jk.initialize()
    jk.C_left_add(Cocca)
    jk.C_left_add(Coccb)

    psi4.core.print_out(sup.description())
    psi4.core.print_out(sup.citation())
    psi4.core.print_out("\n\n")
    jk.print_header()

    Da_m = psi4.core.Matrix(nbf,nbf)
    Db_m = psi4.core.Matrix(nbf,nbf)
    
    diis = ACDIIS(max_vec=options["DIIS_LEN"])
    printHeader("Starting SCF:",2)
    psi4.core.print_out("""{:>10} {:8.2E}
{:>10} {:8.2E}
{:>10} {:8.4f}
{:>10} {:8.2E}
{:>10} {:8.4f}
{:>10} {:8d}
{:>10} {:8d}\n
""".format(
    "E_CONV:",options["E_CONV"],
    "D_CONV:",options["D_CONV"],
    "DAMP:",options["GAMMA"],
    "DIIS_EPS:",options["DIIS_EPS"],
    "VSHIFT:",options["VSHIFT"],
    "MAXITER:",options["MAXITER"],
    "DIIS_LEN:",options["DIIS_LEN"],
    "DIIS_MODE:",options["DIIS_MODE"]))

 
    psi4.core.print_out("\nInitial orbital occupation pattern:\n\n")
    psi4.core.print_out("Index|Spin|Occ|Ovl|Freeze\n"+25*"-")
    for i in orbitals:
        psi4.core.print_out("\n{:^5}|{:^4}|{:^3}|{:^3}|{:^6}".format(i["orb"],i["spin"],i["occ"],'Yes' if i["DoOvl"] else 'No','Yes' if i["frz"] else 'No'))
    psi4.core.print_out("\n\n")

    psi4.core.print_out(("{:^3} {:^14} {:^11} {:^11} {:^11} {:^5} {:^5} | {:^"+str(len(orbitals)*5)+"}| {:^11} {:^5}\n").format("#IT", "Escf",
         "dEscf","Derror","DIIS-E","na","nb",
         "OVL","MIX","Time"))
    psi4.core.print_out("="*(87+5*len(orbitals))+"\n")

    diis_counter = 0
    myTimer = Timer()
    for SCF_ITER in range(1, options["MAXITER"] + 1):
        myTimer.addStart("SCF")

        myTimer.addStart("JK")
        jk.compute()
        myTimer.addEnd("JK")

        myTimer.addStart("buildFock")
        Da_m.np[:] = Da
        Db_m.np[:] = Db
        Vpot.set_D([Da_m,Db_m])
        myTimer.addStart("compV")
        Vpot.compute_V([Va,Vb])
        myTimer.addEnd("compV")

        Ja = np.asarray(jk.J()[0])
        Jb = np.asarray(jk.J()[1])
        Ka = np.asarray(jk.K()[0])
        Kb = np.asarray(jk.K()[1])

        if SCF_ITER>1 :
            FaOld = np.copy(Fa)
            FbOld = np.copy(Fb)

        Fa = H + (Ja + Jb) - Vpot.functional().x_alpha()*Ka + Va
        Fb = H + (Ja + Jb) - Vpot.functional().x_alpha()*Kb + Vb

        myTimer.addEnd("buildFock")
        """
        Fock matrix constructed: freeze orbitals if needed
        """
        myTimer.addStart("Freeze")        
        FMOa = Ca.T @ Fa @ Ca
        FMOb = Cb.T @ Fb @ Cb


        CaInv = np.linalg.inv(Ca)
        CbInv = np.linalg.inv(Cb)

        for i in orbitals:
            if i['frz'] == True:
                if i["spin"]=="b":
                    idx = i["orb"]
                    FMOb[idx,:idx]     = 0.0
                    FMOb[idx,(idx+1):] = 0.0
                    FMOb[:idx,idx]     = 0.0
                    FMOb[(idx+1):,idx] = 0.0
                elif i["spin"]=="a":
                    FMOa[idx,:idx]     = 0.0
                    FMOa[idx,(idx+1):] = 0.0
                    FMOa[:idx,idx]     = 0.0
                    FMOa[(idx+1):,idx] = 0.0
        """
        VSHIFT 
        """        
        idxs = [c for c,x in enumerate(occa) if (x ==0.0) and (c>=nalpha)]
        FMOa[idxs,idxs] += options["VSHIFT"]
        idxs = [c for c,x in enumerate(occb) if (x ==0.0) and (c>=nbeta)]
        FMOb[idxs,idxs] += options["VSHIFT"]

        Fa = CaInv.T @ FMOa @ CaInv
        Fb = CbInv.T @ FMOb @ CbInv

        myTimer.addEnd("Freeze")
        """
        END FREEZE
        """

        """
        DIIS/MIXING
        """
        myTimer.addStart("MIX")      
        
        diisa_e = np.ravel(A.T@(Fa@Da@S - S@Da@Fa)@A)
        diisb_e = np.ravel(A.T@(Fb@Db@S - S@Db@Fb)@A)
        diis.add(Fa,Fb,Da,Db,np.concatenate((diisa_e,diisb_e)))

        if ("DIIS" in options["MIXMODE"]) and (SCF_ITER>1):
            (Fa,Fb) = diis.extrapolate(DIISError)
            diis_counter += 1
            if (diis_counter >= 2*options["DIIS_LEN"]):
                diis.reset()
                diis_counter = 0
                psi4.core.print_out("Resetting DIIS\n")

        elif (options["MIXMODE"] == "DAMP") and (SCF_ITER>1):
            # Use Damping to obtain the new Fock matrices
            Fa = (1-options["GAMMA"]) * np.copy(Fa) + (options["GAMMA"]) * FaOld
            Fb = (1-options["GAMMA"]) * np.copy(Fb) + (options["GAMMA"]) * FbOld
        
        myTimer.addEnd("MIX")    
        """
        END DIIS/MIXING
        """

        """
        CALC energy
        """
        myTimer.addStart("calcE")

        one_electron_E  = np.sum(Da * H)
        one_electron_E += np.sum(Db * H)
        coulomb_E       = np.sum(Da * (Ja+Jb))
        coulomb_E      += np.sum(Db * (Ja+Jb))

        alpha       = Vpot.functional().x_alpha()
        exchange_E  = 0.0
        exchange_E -= alpha * np.sum(Da * Ka)
        exchange_E -= alpha * np.sum(Db * Kb)

        XC_E = Vpot.quadrature_values()["FUNCTIONAL"]

        SCF_E = 0.0
        SCF_E += Enuc
        SCF_E += one_electron_E
        SCF_E += 0.5 * coulomb_E
        SCF_E += 0.5 * exchange_E
        SCF_E += XC_E
        myTimer.addEnd("calcE")
        
        # Diagonalize Fock matrix
        myTimer.addStart("Diag")
        Ca,epsa = diag_H(Fa, A)
        Cb,epsb = diag_H(Fb, A)
        myTimer.addEnd("Diag")

        DaOld = np.copy(Da)
        DbOld = np.copy(Db)


        """
        New orbitals obtained set occupation numbers
        """
        myTimer.addStart("SetOcc")

        Cocca.np[:]  = Ca
        Coccb.np[:]  = Cb


        occa[:] = 0.0
        occb[:] = 0.0

        occa[:nalpha] = 1.0  #standard aufbau principle occupation
        occb[:nbeta]  = 1.0


        for i in orbitals:
            if i["spin"]=="b":
                """
                Overlap
                """
                #calculate the Overlapp with all other orbitals
                ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
                #User wants to switch the index if higher overlap is found
                if i["DoOvl"] ==True :
                    if i["orb"] != np.argmax(ovl):
                        i["orb"] = np.argmax(ovl)
                    i["ovl"] = np.max(ovl)
                else:
                    #just calculate the overlap to assess the character
                    i["ovl"] = ovl[i["orb"]]
                #Modify the occupation vector
                occb[i["orb"]] = i["occ"]

            elif i["spin"]=="a":
                """
                Check if this is still the largest overlap
                """
                ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Cocca,S))
                if i["DoOvl"] ==True :
                    if i["orb"] != np.argmax(ovl):
                        i["orb"] = np.argmax(ovl) # set index to the highest overlap
                    i["ovl"] = np.max(ovl)
                else:
                    i["ovl"] = ovl[i["orb"]]
                #Modify the occupation vector
                occa[i["orb"]] = i["occ"]
                

        for i in range(nbf):
            Cocca.np[:,i] *= np.sqrt(occa[i])
            Coccb.np[:,i] *= np.sqrt(occb[i])
        
        Da     = Cocca.np @ Cocca.np.T
        Db     = Coccb.np @ Coccb.np.T
        myTimer.addEnd("SetOcc")

        DError = (np.sum((DaOld-Da)**2)**0.5 + np.sum((DbOld-Db)**2)**0.5)
        EError = (SCF_E - Eold)
        DIISError = (np.sum(diisa_e**2)**0.5 + np.sum(diisb_e**2)**0.5)

        myTimer.addEnd("SCF")
        psi4.core.print_out(("{:3d} {:14.8f} {:11.3E} {:11.3E} {:11.3E} {:5.1f} {:5.1f} | "+"{:4.2f} "*len(orbitals)+"| {:^11} {:5.2f} {:2d}  \n").format(
            SCF_ITER,
            SCF_E,
            EError,
            DError,
            DIISError,
            np.sum(Da*S),
            np.sum(Db*S),
            *[x["ovl"] for x in orbitals],
            options["MIXMODE"],
            myTimer.getTime("SCF"),
            len(diis.Fa)))
        psi4.core.flush_outfile()
        myTimer.printAlltoFile("timers.ksex")
        
        if (abs(DIISError) < options["DIIS_EPS"]):
            options["MIXMODE"] = options["DIIS_MODE"]
        else:
            options["MIXMODE"] = "DAMP"  
        
        
        if (abs(EError) < options["E_CONV"]) and (abs(DError)<options["D_CONV"]):
            if (options["VSHIFT"] != 0.0):
                psi4.core.print_out("Converged but Vshift was on... removing Vshift..\n")
                options["VSHIFT"] = 0.0
            else:
                break


        Eold = SCF_E   



        if SCF_ITER == options["MAXITER"]:
            psi4.core.clean()
            np.savez(options["PREFIX"]+'_exorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb,orbitals=orbitals)
            raise Exception("Maximum number of SCF cycles exceeded.")

    psi4.core.print_out("\n\n{:>20} {:12.8f} [Ha] \n".format("FINAL EX SCF ENERGY:",SCF_E))
    gsE = psi4.core.get_variable('GS ENERGY')
    if gsE!=0.0:
        psi4.core.print_out("{:>20} {:12.8f} [Ha] \n".format("EXCITATION ENERGY:",SCF_E-gsE))
        psi4.core.print_out("{:>20} {:12.8f} [eV] \n\n".format("EXCITATION ENERGY:",(SCF_E-gsE)*27.211385))

    psi4.core.print_out("\nFinal orbital occupation pattern:\n\n")
    psi4.core.print_out("Index|Spin|Occ|Ovl|Freeze|Comment\n"+34*"-")
    for i in orbitals:
        Comment = "-"
        if i["DoOvl"]:
            psi4.core.print_out("\n{:^5}|{:^4}|{:^3}|{:^3}|{:^6}|{:^7}".format(i["orb"],i["spin"],i["occ"],'Yes' if i["DoOvl"] else 'No','Yes' if i["frz"] else 'No',Comment))
        else:
            if i["spin"]=="b":
                #calculate the Overlap with all other orbitals
                ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
                idx = np.argmax(ovl)
            elif  i["spin"]=="a":
                ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Cocca,S))
                idx = np.argmax(ovl)
            Comment = " Found by overlap"
            psi4.core.print_out("\n{:^5}|{:^4}|{:^3}|{:^3}|{:^6}|{:^7}".format(idx,i["spin"],i["occ"],'Yes' if i["DoOvl"] else 'No','Yes' if i["frz"] else 'No',Comment))

    
    if options["USE_AUG"]:
        """
        What are we doing here?:
        [1]: Find the excitation center -> exci
        [2]: Rebuild the normal basis set and add the new functions
        [3]: Copy C to Caug and rebuild and diagonalize the Fock matrix
        """

        from .kshelper import augBas 

        printHeader("Using Augmentation Basis Set:",2)
        
        if not (wfn.basisset().has_puream()):
            raise Exception("Please use a basis set with spherical gaussian basis functions...")


        D = mints.ao_dipole()
        Dx,Dy,Dz = np.asarray(D[0]),np.asarray(D[1]),np.asarray(D[2])
        orbI = ([c for c,x in enumerate(occb) if x != 1.0][0])
        Contribs = [[x,0] for x in range(mol.natom())]

        for i in range(mol.natom()):
            for c,j in enumerate(Cb[:,orbI]):
                if wfn.basisset().function_to_center(c)==i:
                    Contribs[i][1] += j**2

        exci = sorted(Contribs,key=lambda f: f[1])[-1][0]
        psi4.core.print_out("\nExc. center index   : {}".format(exci))

        if mol.fsymbol(exci) in ["LI","BE","B","C","N","O","F","NE"]:
            psi4.core.print_out("\nSym -> 1st row      : {}".format(mol.fsymbol(exci)))
            B = augBas["firstRow"]
        else:
            psi4.core.print_out("\nSym -> 2nd row      : {}".format(mol.fsymbol(exci)))
            B = augBas["secondRow"]

       
        
        qcmol   = qcdb.Molecule.from_dict(mol.to_dict())
        qcbasis,qcdict = qcdb.libmintsbasisset.BasisSet.pyconstruct(qcmol,'BASIS',"","ORBITAL",options["BASIS"],return_dict=True)
      
        bfBefore = []
        bfAfter = []
        for i in qcdict["shell_map"][:exci+1]:
            for j in i:
                if isinstance(j,list):
                    bfBefore.append(j)
        
        for i in qcdict["shell_map"][exci+1:]:
            for j in i:
                if isinstance(j,list):
                    bfAfter.append(j)

        
        nbefore = sum([2*x[0]+1 for x in bfBefore]) #for puream basis sets
        nafter  = sum([2*x[0]+1 for x in bfAfter])  #for puream basis sets
                    
        psi4.core.print_out("\nBasis #nbf          : {}".format(wfn.basisset().nbf()))
        psi4.core.print_out("\n#nbf - before insert: {}".format(nbefore))
        psi4.core.print_out("\n#nbf - after  insert: {}".format(nafter))

        if ((nbefore+nafter) != wfn.basisset().nbf()):
            raise Exception("Something is wrong with the Basis sets... Stopping")

        for i in B:
            qcdict["shell_map"][exci].append(i)
    
        aug_basis = psi4.core.BasisSet.construct_from_pydict(mol,qcdict,qcdict["puream"])
        aug_mints = psi4.core.MintsHelper(aug_basis)


        psi4.core.print_out("\naugBasis #nbf       : {}".format(aug_basis.nbf()))

        Saug = np.asarray(aug_mints.ao_overlap())
        Taug = np.asarray(aug_mints.ao_kinetic())
        Vaug = np.asarray(aug_mints.ao_potential())
        Haug = np.zeros((aug_mints.nbf(),mints.nbf()))

        e,v = np.linalg.eigh(Saug)
        psi4.core.print_out("\n\nSmin     : {:8.2E} ".format(np.min(e)))
        psi4.core.print_out("\nS-cutoff : {:8.2E} ".format(1.0E-6))
        psi4.core.print_out("\nRemoving Linear dependencies")

        idx = np.where(e>1.0E-6)

        e = np.diag(1/np.sqrt(e[idx]))
        vtmp = v[:,idx[0]]
        Aaug = vtmp@e
        
        Haug = Taug+Vaug
        if aug_basis.has_ECP():
            ECP = aug_mints.ao_ecp()
            Haug += ECP                
        
        augNbf = Aaug.shape[1]
        psi4.core.print_out("\nNew basis size: {}\n\n".format(augNbf))   

        augCaOcc = psi4.core.Matrix(aug_basis.nbf(),augNbf) 
        augCbOcc = psi4.core.Matrix(aug_basis.nbf(),augNbf) 

        idx = list(range(nbefore)) + list(range(aug_basis.nbf()-nafter,aug_basis.nbf()))
        for c,i in enumerate(occa):
            if i>0.0:
                augCaOcc.np[:,c][idx] = Ca[:,c]
        
        for c,i in enumerate(occb):
            if i>0.0:
                augCbOcc.np[:,c][idx] = Cb[:,c]
        
        for i in range(nbf):
                augCaOcc.np[:,i] *= np.sqrt(occa[i])
                augCbOcc.np[:,i] *= np.sqrt(occb[i])

        augDa = augCaOcc.np @ augCaOcc.np.T
        augDb = augCbOcc.np @ augCbOcc.np.T
        augVa = psi4.core.Matrix(aug_basis.nbf(),aug_basis.nbf())
        augVb = psi4.core.Matrix(aug_basis.nbf(),aug_basis.nbf())

        augSup = psi4.driver.dft.build_superfunctional(func, False)[0]
        augSup.set_deriv(2)
        augSup.allocate()

        augVpot = psi4.core.VBase.build(aug_basis, augSup, "UV")
        augVpot.initialize()


        psi4.core.set_global_option("SCF_TYPE","DIRECT") 
        psi4.core.set_local_option("SCF","SCF_TYPE","DIRECT") 

        psi4.core.be_quiet()
        augJk = psi4.core.JK.build(aug_basis)
        augJk.print_header()
        glob_mem = psi4.core.get_memory()/8
        augJk.set_memory(int(glob_mem*0.6))
        augJk.initialize()
        augJk.C_left_add(augCaOcc)
        augJk.C_left_add(augCbOcc)

        psi4.core.reopen_outfile()

        Da_m = psi4.core.Matrix(aug_basis.nbf(),aug_basis.nbf())
        Db_m = psi4.core.Matrix(aug_basis.nbf(),aug_basis.nbf())

        Da_m.np[:] = augDa
        Db_m.np[:] = augDb


        augJk.compute()

        augVpot.set_D([Da_m,Db_m])
        augVpot.compute_V([augVa,augVb])

        Ja = np.asarray(augJk.J()[0])
        Jb = np.asarray(augJk.J()[1])
        Ka = np.asarray(augJk.K()[0])
        Kb = np.asarray(augJk.K()[1])

        Fa = Haug + (Ja + Jb) - Vpot.functional().x_alpha()*Ka + augVa
        Fb = Haug + (Ja + Jb) - Vpot.functional().x_alpha()*Kb + augVb


        myTimer.addStart("Diag")
        augCa,augEpsa = diag_H(Fa, Aaug)
        augCb,augEpsb = diag_H(Fb, Aaug)
        myTimer.addEnd("Diag")


        
        one_electron_E  = np.sum(augDa * Haug)
        one_electron_E += np.sum(augDb * Haug)
        coulomb_E       = np.sum(augDa * (Ja+Jb))
        coulomb_E      += np.sum(augDb * (Ja+Jb))

        alpha       = augVpot.functional().x_alpha()
        exchange_E  = 0.0
        exchange_E -= alpha * np.sum(augDa * Ka)
        exchange_E -= alpha * np.sum(augDb * Kb)

        XC_E = augVpot.quadrature_values()["FUNCTIONAL"]

        SCF_E = 0.0
        SCF_E += Enuc
        SCF_E += one_electron_E
        SCF_E += 0.5 * coulomb_E
        SCF_E += 0.5 * exchange_E
        SCF_E += XC_E

        
        augOcca = np.zeros(augCa.shape[1])
        augOccb = np.zeros(augCb.shape[1])

        augOcca[:nalpha] = 1.0 
        augOccb[:nbeta]  = 1.0

        augOrbitals = copy.deepcopy(orbitals)

        for orb,aug in zip(orbitals,augOrbitals):
            aug["C"]      = np.zeros(aug_basis.nbf())
            aug["C"][idx] = orb["C"]    

        for i in augOrbitals:
            if i["spin"]=="b":
                """
                Overlap
                """
                #calculate the Overlapp with all other orbitals
                ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],augCb,Saug))
                #User wants to switch the index if higher overlap is found
                if i["DoOvl"] ==True :
                    if i["orb"] != np.argmax(ovl):
                        i["orb"] = np.argmax(ovl)
                    i["ovl"] = np.max(ovl)
                else:
                    #just calculate the overlap to assess the character
                    i["ovl"] = ovl[i["orb"]]
                    #Modify the occupation vector
                augOccb[i["orb"]] = i["occ"]

            elif i["spin"]=="a":
                """
                Check if this is still the largest overlap
                """
                ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],augCa,Saug))
                if i["DoOvl"] ==True :
                    if i["orb"] != np.argmax(ovl):
                        i["orb"] = np.argmax(ovl) # set index to the highest overlap
                    i["ovl"] = np.max(ovl)
                else:
                    i["ovl"] = ovl[i["orb"]]
                    #Modify the occupation vector
                augOcca[i["orb"]] = i["occ"]

        Daug = np.asarray([np.asarray(x) for x in aug_mints.ao_dipole()])
        Dx,Dy,Dz = Daug[0],Daug[1],Daug[2]

        np.savez(options["PREFIX"]+'_exorbsAug',D=Daug,Ca=augCa,Cb=augCb,occa=augOcca,occb=augOccb,epsa=augEpsa,epsb=augEpsb,orbitals=orbitals)


    D = np.asarray([np.asarray(x) for x in mints.ao_dipole()])
    OCCA = psi4.core.Vector(nbf)
    OCCB = psi4.core.Vector(nbf)
    OCCA.np[:] = occa
    OCCB.np[:] = occb
        

    uhf.Ca().np[:] = Ca
    uhf.Cb().np[:] = Cb

    uhf.epsilon_a().np[:] = epsa
    uhf.epsilon_b().np[:] = epsb

    uhf.occupation_a().np[:] = occa
    uhf.occupation_b().np[:] = occb

    mw = psi4.core.MoldenWriter(uhf)
    mw.write(options["PREFIX"]+'_ex.molden',uhf.Ca(),uhf.Cb(),uhf.epsilon_a(),uhf.epsilon_b(),OCCA,OCCB,True)
    psi4.core.print_out("\nMoldenfile written")
    np.savez(options["PREFIX"]+'_exorbs',D=D,Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb,orbitals=orbitals)

    psi4.core.set_variable('CURRENT ENERGY', SCF_E)
