# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 01:57:59 2018

@author: luke
"""
import psi4
import numpy as np
from .kshelper import diag_H,ACDIIS,Timer,printHeader
import os.path
import time
import logging
import pdb

def DFTGroundState(mol,func,**kwargs):
    """
    Perform unrestrictred Kohn-Sham
    """
    printHeader("Entering Ground State Kohn-Sham")

    options = {
        "PREFIX"    : psi4.core.get_local_option("PSIXAS","PREFIX"),
        "E_CONV"    : float(psi4.core.get_local_option("PSIXAS","E_GS_CONV")),
        "D_CONV"    : float(psi4.core.get_local_option("PSIXAS","D_GS_CONV")),
        "MAXITER"   : int(psi4.core.get_local_option("PSIXAS","MAXITER")),
        "BASIS"     : psi4.core.get_global_option('BASIS'),
        "GAMMA"     : float(psi4.core.get_local_option("PSIXAS","DAMP")),
        "DIIS_LEN"  : int(psi4.core.get_local_option("PSIXAS","DIIS_LEN")),
        "DIIS_MODE" : psi4.core.get_local_option("PSIXAS","DIIS_MODE"),
        "DIIS_EPS"  : float(psi4.core.get_local_option("PSIXAS","DIIS_EPS")),
        "MIXMODE"   : "DAMP"}

    printHeader("Basis Set:",2)
    wfn   = psi4.core.Wavefunction.build(mol,options["BASIS"])
    aux   = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", options["BASIS"],puream=wfn.basisset().has_puream())

    sup = psi4.driver.dft.build_superfunctional(func, False)[0]
    psi4.core.be_quiet()
    mints = psi4.core.MintsHelper(wfn.basisset())
    
    
    sup.allocate()

    uhf   = psi4.core.UHF(wfn,sup)
    psi4.core.reopen_outfile()
    
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

    nbf    = wfn.nso()
    nalpha = wfn.nalpha()
    nbeta  = wfn.nbeta()

    Va = psi4.core.Matrix(nbf,nbf)
    Vb = psi4.core.Matrix(nbf,nbf)

    Vpot = psi4.core.VBase.build(wfn.basisset(), sup, "UV")
    Vpot.initialize()

    """
    Read or Core Guess
    """    
    Cocca       = psi4.core.Matrix(nbf, nalpha)
    Coccb       = psi4.core.Matrix(nbf, nbeta)
    if (os.path.isfile(options["PREFIX"]+"_gsorbs.npz")):
        psi4.core.print_out("\nRestarting Calculation \n")
        Ca = np.load(options["PREFIX"]+"_gsorbs.npz")["Ca"]
        Cb = np.load(options["PREFIX"]+"_gsorbs.npz")["Cb"]
        Cocca.np[:]  = Ca[:, :nalpha]
        Da     = Ca[:, :nalpha] @ Ca[:, :nalpha].T
        Coccb.np[:]  = Cb[:, :nbeta]
        Db     = Cb[:, :nbeta] @ Cb[:, :nbeta].T
    else:
        """
        SADNO guess, heavily inspired by the psi4 implementation
        """
        psi4.core.print_out("Creating SADNO guess\n\n")
        sad_basis_list = psi4.core.BasisSet.build(mol, "ORBITAL",
                options["BASIS"],puream=True,return_atomlist=True)
        sad_fitting_list = psi4.core.BasisSet.build(mol,"DF_BASIS_SAD",
                psi4.core.get_option("SCF", "DF_BASIS_SAD"),puream=True, return_atomlist=True)
        SAD = psi4.core.SADGuess.build_SAD(wfn.basisset(), sad_basis_list)
        SAD.set_atomic_fit_bases(sad_fitting_list)
        SAD.compute_guess()
        Da = SAD.Da().np

        Dhelp = -1.0 * np.copy(Da)
        Dhelp = A.T @ S.T @ Dhelp @ S @ A

        _,C1 = np.linalg.eigh(Dhelp)

        Ca = A @ C1
        Cb = np.copy(Ca)
        
        Cocca.np[:] = Ca[:, :nalpha]
        Coccb.np[:] = Cb[:, :nbeta]
     
        #This is the guess!
        Da  = Cocca.np @ Cocca.np.T
        Db  = Coccb.np @ Coccb.np.T

    """
    end read
    """
    printHeader("Molecule:",2)
    mol.print_out()
    printHeader("XC & JK-Info:",2)

    if (sup.is_x_lrc()):
      if (psi4.core.has_local_option_changed("SCF","SCF_TYPE")==False):
            psi4.core.set_local_option("SCF","SCF_TYPE","DISK_DF")
    else:
        if (psi4.core.has_local_option_changed("SCF","SCF_TYPE")==False):
            psi4.core.set_local_option("SCF","SCF_TYPE","MEM_DF")  

    jk = psi4.core.JK.build(wfn.basisset(),aux=aux,jk_type=psi4.core.get_option("SCF", "SCF_TYPE"))
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    jk.set_do_K(sup.is_x_hybrid())
    jk.set_do_wK(sup.is_x_lrc())
    jk.set_omega(sup.x_omega())
        
    jk.initialize()
    jk.C_left_add(Cocca)
    jk.C_left_add(Coccb)

    Da_m = psi4.core.Matrix(nbf,nbf)
    Db_m = psi4.core.Matrix(nbf,nbf)
    
    sup.print_out()
    psi4.core.print_out("\n\n")
    jk.print_header()
        
    diis = ACDIIS(max_vec=options["DIIS_LEN"],diismode=options["DIIS_MODE"])
    diisa_e = 1000.0
    diisb_e = 1000.0

    printHeader("Starting SCF:",2)    
    psi4.core.print_out("""{:>10} {:8.2E}
{:>10} {:8.2E}
{:>10} {:8.4f}
{:>10} {:8.2E}
{:>10} {:8d}
{:>10} {:8d}
{:>10} {:^11}""".format(
    "E_CONV:",options["E_CONV"],
    "D_CONV:",options["D_CONV"],
    "DAMP:",options["GAMMA"],
    "DIIS_EPS:",options["DIIS_EPS"],
    "MAXITER:", options["MAXITER"],
    "DIIS_LEN:",options["DIIS_LEN"],
    "DIIS_MODE:",options["DIIS_MODE"]))

    myTimer = Timer()

    psi4.core.print_out("\n\n{:^4} {:^14} {:^11} {:^11} {:^11} {:^11} {:^6} \n".format("# IT", "Escf", "dEscf","Derror","DIIS-E","MIX","Time"))
    psi4.core.print_out("="*80+"\n")
    diis_counter = 0

    for SCF_ITER in range(1, options["MAXITER"] + 1):
        myTimer.addStart("SCF")     
        jk.compute()
        
        """
        Build Fock
        """
        Da_m.np[:] = Da
        Db_m.np[:] = Db
        Vpot.set_D([Da_m,Db_m])
        Vpot.compute_V([Va,Vb])

        Ja = np.asarray(jk.J()[0])
        Jb = np.asarray(jk.J()[1])

        if SCF_ITER>1 :
            FaOld = np.copy(Fa)
            FbOld = np.copy(Fb)

        Fa = H + (Ja + Jb) + Va
        Fb = H + (Ja + Jb) + Vb
        if sup.is_x_hybrid():
            Fa -= sup.x_alpha()*np.asarray(jk.K()[0]) 
            Fb -= sup.x_alpha()*np.asarray(jk.K()[1])
        if sup.is_x_lrc(): 
            Fa -= sup.x_beta()*np.asarray(jk.wK()[0]) 
            Fb -= sup.x_beta()*np.asarray(jk.wK()[1])
        """
        END BUILD FOCK
        """

        """
        CALC E
        """
        one_electron_E  = np.sum(Da * H)
        one_electron_E += np.sum(Da * H)
        coulomb_E       = np.sum(Da * (Ja+Jb))
        coulomb_E      += np.sum(Db * (Ja+Jb))

        exchange_E  = 0.0
        if sup.is_x_hybrid():
            exchange_E -=  sup.x_alpha() * np.sum(Da * np.asarray(jk.K()[0]))
            exchange_E -=  sup.x_alpha() * np.sum(Db * np.asarray(jk.K()[1]))
        if sup.is_x_lrc():
            exchange_E -= sup.x_beta() * np.sum(Da * np.asarray(jk.wK()[0]))
            exchange_E -= sup.x_beta() * np.sum(Db * np.asarray(jk.wK()[1]))


        XC_E = Vpot.quadrature_values()["FUNCTIONAL"]


        SCF_E = 0.0
        SCF_E += Enuc
        SCF_E += one_electron_E
        SCF_E += 0.5 * coulomb_E
        SCF_E += 0.5 * exchange_E
        SCF_E += XC_E
        """
        END CALCE
        """

        """
        DIIS/MIXING
        """
        diisa_e = np.ravel(A.T@(Fa@Da@S - S@Da@Fa)@A)
        diisb_e = np.ravel(A.T@(Fb@Db@S - S@Db@Fb)@A)
        diis.add(Fa,Fb,Da,Db,np.concatenate((diisa_e,diisb_e)))


        if ("DIIS" in options["MIXMODE"]) and (SCF_ITER>1):
            # Extrapolate alpha & beta Fock matrices separately
            (Fa,Fb) = diis.extrapolate(DIISError)
            diis_counter += 1

            if (diis_counter >= 2*options["DIIS_LEN"]):
                diis.reset()
                diis_counter = 0
                psi4.core.print_out("Resetting DIIS\n")

        elif (options["MIXMODE"] == "DAMP") and (SCF_ITER>1):
            #...but use damping to obtain the new Fock matrices
            Fa = (1-options["GAMMA"]) * np.copy(Fa) + (options["GAMMA"]) * FaOld
            Fb = (1-options["GAMMA"]) * np.copy(Fb) + (options["GAMMA"]) * FbOld

        """
        END DIIS/MIXING
        """
       

        """
        DIAG F-tilde -> get D
        """
        DaOld = np.copy(Da)
        DbOld = np.copy(Db)

        Ca,epsa = diag_H(Fa, A)
        Cocca.np[:]  = Ca[:, :nalpha]
        Da      = Cocca.np @ Cocca.np.T


        Cb,epsb = diag_H(Fb, A)
        Coccb.np[:]  = Cb[:, :nbeta]
        Db      = Coccb.np @ Coccb.np.T
        """
        END DIAG F + BUILD D
        """

        DError = (np.sum((DaOld-Da)**2)**0.5 + np.sum((DbOld-Db)**2)**0.5)
        EError = (SCF_E - Eold)
        DIISError = (np.sum(diisa_e**2)**0.5 + np.sum(diisb_e**2)**0.5)
     
        """
        OUTPUT
        """
        myTimer.addEnd("SCF")
        psi4.core.print_out(" {:3d} {:14.8f} {:11.3E} {:11.3E} {:11.3E} {:^11} {:6.2f} {:2d} \n".format(SCF_ITER,
             SCF_E,
             EError,
             DError,
             DIISError,
             options["MIXMODE"],
             myTimer.getTime("SCF"),
             len(diis.Fa)))
                  
        psi4.core.flush_outfile()
        if (abs(DIISError) < options["DIIS_EPS"]):
            options["MIXMODE"] = options["DIIS_MODE"]
        else:
            options["MIXMODE"] = "DAMP"        
        
        if (abs(EError) < options["E_CONV"]) and (abs(DError)<options["D_CONV"]):
            break

        Eold = SCF_E

        if SCF_ITER == options["MAXITER"]:
            psi4.core.clean()
            occa = np.zeros(nbf,dtype=np.float)
            occb = np.zeros(nbf,dtype=np.float)
            occa[:nalpha] = 1.0
            occb[:nbeta]  = 1.0
            np.savez(options["PREFIX"]+'_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb)
            raise Exception("Maximum number of SCF cycles exceeded.")

    psi4.core.print_out("\n\nFINAL GS SCF ENERGY: {:12.8f} [Ha] \n\n".format(SCF_E))

    mw = psi4.core.MoldenWriter(wfn)
    occa = np.zeros(nbf,dtype=np.float)
    occb = np.zeros(nbf,dtype=np.float)

    occa[:nalpha] = 1.0
    occb[:nbeta]  = 1.0

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
    mw.write(options["PREFIX"]+'_gs.molden',uhf.Ca(),uhf.Cb(),uhf.epsilon_a(),uhf.epsilon_b(),OCCA,OCCB,True)
    psi4.core.print_out("\nMoldenfile written\n")
    
    np.savez(options["PREFIX"]+'_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb)
    psi4.core.print_out("Canoncical Orbitals written\n\n")                        

    psi4.core.set_variable('CURRENT ENERGY', SCF_E)
    psi4.core.set_variable('GS ENERGY', SCF_E)
    return uhf
