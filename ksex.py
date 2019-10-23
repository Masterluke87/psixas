# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 18:30:31 2018

@author: luke

Module to perform excited state calculations
"""
from .kshelper import diag_H,Timer, ACDIIS
import numpy as np
import os
import psi4
import time
import logging

def DFTExcitedState(mol,func,orbitals,**kwargs):
    """
    Perform unrestrictred Kohn-Sham excited state calculation
    """
    psi4.core.print_out("\nEntering Excited State Kohn-Sham:\n"+33*"="+"\n\n")

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
        "MIXMODE"   : "DAMP"}

    """
    STEP 1: Read in ground state orbitals or restart from previous
    """
    if (os.path.isfile(options["PREFIX"]+"_exorbs.npz")):
        psi4.core.print_out("Restarting Calculation from: {} \n\n".format(options["PREFIX"]+"_exorbs.npz")) 
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

    #This object is needed to write out a molden file later
    uhf   = psi4.core.UHF(wfn,sup)
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

    jk = psi4.core.JK.build(wfn.basisset(),aux,jk_type=psi4.core.get_local_option("SCF","SCF_TYPE"))
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    jk.initialize()
    jk.C_left_add(Cocca)
    jk.C_left_add(Coccb)

    Da_m = psi4.core.Matrix(nbf,nbf)
    Db_m = psi4.core.Matrix(nbf,nbf)
    
    diis = ACDIIS(max_vec=options["DIIS_LEN"])

    psi4.core.print_out("""
Starting SCF:
"""+13*"="+"""\n
{:>10} {:8.2E}
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
                #calculate the Overlapp with all other orbitals
                ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Coccb,S))
                idx = np.argmax(ovl)
            elif  i["spin"]=="a":
                ovl = np.abs(np.einsum('m,nj,mn->j',i["C"],Cocca,S))
                idx = np.argmax(ovl)
            Comment = " Found by overlap"
            psi4.core.print_out("\n{:^5}|{:^4}|{:^3}|{:^3}|{:^6}|{:^7}".format(idx,i["spin"],i["occ"],'Yes' if i["DoOvl"] else 'No','Yes' if i["frz"] else 'No',Comment))




    psi4.core.print_out("\n\n")




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

    OCCA.print_out()
    OCCB.print_out()

    mw = psi4.core.MoldenWriter(uhf)
    mw.write(options["PREFIX"]+'_ex.molden',uhf.Ca(),uhf.Cb(),uhf.epsilon_a(),uhf.epsilon_b(),OCCA,OCCB,True)
    psi4.core.print_out("\n\n Moldenfile written\n")
    np.savez(options["PREFIX"]+'_exorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb,orbitals=orbitals)

    psi4.core.set_variable('CURRENT ENERGY', SCF_E)
