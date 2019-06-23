# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 01:57:59 2018

@author: luke
"""
import psi4
import numpy as np
from .kshelper import diag_H,DIIS_helper,Timer
import os.path
import time


def DFTGroundState(mol,func,**kwargs):
    """
    Perform unrestrictred Kohn-Sham
    """
    psi4.core.print_out("\nEntering DFT GS-module.\n")

    maxiter = 100
    E_conv  = 1.0E-8
    D_conv  = 1.0E-6

    prefix = kwargs["PREFIX"]

    wfn   = psi4.core.Wavefunction.build(mol,psi4.core.get_global_option('BASIS'))
    aux   = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", psi4.core.get_global_option('BASIS'))
    mints = psi4.core.MintsHelper(wfn.basisset())

    sup = psi4.procrouting.dft_funcs.build_superfunctional(func, False)[0]
    sup.set_deriv(2)
    sup.allocate()

    uhf   = psi4.core.UHF(wfn,sup)

    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = np.zeros((mints.nbf(),mints.nbf()))

    H = T+V

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

    gamma    =  float(psi4.core.get_local_option("PSIXAS","DAMP"))
    diis_eps =  float(psi4.core.get_local_option("PSIXAS","DIIS_EPS"))
    """
    Read or Core Guess
    """    
    Cocca       = psi4.core.Matrix(nbf, nalpha)
    Coccb       = psi4.core.Matrix(nbf, nbeta)
    if (os.path.isfile(prefix+"_gsorbs.npz")):
        psi4.core.print_out("Restarting Calculation")
        Ca = np.load(prefix+"_gsorbs.npz")["Ca"]
        Cb = np.load(prefix+"_gsorbs.npz")["Cb"]
        Cocca.np[:]  = Ca[:, :nalpha]
        Da     = Ca[:, :nalpha] @ Ca[:, :nalpha].T
        Coccb.np[:]  = Cb[:, :nbeta]
        Db     = Cb[:, :nbeta] @ Cb[:, :nbeta].T
    else:
        Ca,epsa     = diag_H(H,A)       
        Cocca.np[:] = Ca[:, :nalpha]
        Da          = Ca[:, :nalpha] @ Ca[:, :nalpha].T

        Cb,epsb     = diag_H(H,A)        
        Coccb.np[:] = Cb[:, :nbeta]
        Db          = Cb[:, :nbeta] @ Cb[:, :nbeta].T
    """
    end read
    """

    # Initialize the JK object
    jk = psi4.core.JK.build(wfn.basisset(),aux,"MEM_DF")
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    jk.initialize()
    jk.C_left_add(Cocca)
    jk.C_left_add(Coccb)

    Da_m = psi4.core.Matrix(nbf,nbf)
    Db_m = psi4.core.Matrix(nbf,nbf)

    psi4.core.print_out(sup.description())
    psi4.core.print_out(sup.citation())
    
    psi4.core.print_out("\n DAMP: {:4.2f} \n DIIS_EPS: {:4.2f} \n".format(gamma,diis_eps))
    psi4.core.print_out("\n\n{:^4} {:^14} {:^14} {:^14} {:^4} {:^6} \n".format("# IT", "Escf", "dEscf","Derror","MIX","Time"))
    psi4.core.print_out("="*80+"\n")

    diisa = DIIS_helper()
    diisb = DIIS_helper()

    myTimer = Timer()

    MIXMODE = "DAMP"
    for SCF_ITER in range(1, maxiter + 1):
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
        Ka = np.asarray(jk.K()[0])
        Kb = np.asarray(jk.K()[1])

        if SCF_ITER>1 :
            FaOld = np.copy(Fa)
            FbOld = np.copy(Fb)

        Fa = (H + (Ja + Jb) - Vpot.functional().x_alpha()*Ka + Va)
        Fb = (H + (Ja + Jb) - Vpot.functional().x_alpha()*Kb + Vb)
        """
        END BUILD FOCK
        """

        """
        DIIS/MIXING
        """
        if (MIXMODE == "DIIS") and (SCF_ITER>1):
            diisa_e = Fa.dot(Da).dot(S) - S.dot(Da).dot(Fa)
            diisa_e = (A.T).dot(diisa_e).dot(A)
            diisa.add(Fa, diisa_e)

            diisb_e = Fb.dot(Db).dot(S) - S.dot(Db).dot(Fb)
            diisb_e = (A.T).dot(diisb_e).dot(A)
            diisb.add(Fb, diisb_e)

            # Extrapolate alpha & beta Fock matrices separately
            Fa = diisa.extrapolate()
            Fb = diisb.extrapolate()
        elif (MIXMODE == "DAMP") and (SCF_ITER>1):
            Fa = (1-gamma) * np.copy(Fa) + (gamma) * FaOld
            Fb = (1-gamma) * np.copy(Fb) + (gamma) * FbOld

        """
        END DIIS/MIXING
        """

        

        """
        CALC E
        """
        one_electron_E  = np.sum(Da * H)
        one_electron_E += np.sum(Da * H)
        coulomb_E       = np.sum(Da * (Ja+Jb))
        coulomb_E      += np.sum(Db * (Ja+Jb))

        alpha       = Vpot.functional().x_alpha()
        exchange_E  = 0.0;
        exchange_E -= alpha * np.sum(Da * Ka)
        exchange_E -= alpha * np.sum(Db * Kb)

        XC_E = Vpot.quadrature_values()["FUNCTIONAL"];


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
        DIAG F + BUILD D
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

        """
        OUTPUT
        """

        myTimer.addEnd("SCF")
        psi4.core.print_out(" {:3d} {:14.8f} {:14.8f} {:14.8f} {:^4} {:6.2f}\n".format(SCF_ITER,
             SCF_E,
             (SCF_E - Eold),
             (np.mean(np.abs(DaOld-Da)) + np.sum(np.abs(DbOld-Db))),
             MIXMODE,
             myTimer.getTime("SCF")))
                  
        psi4.core.flush_outfile()
        if (abs(SCF_E - Eold) < diis_eps):
            MIXMODE = "DIIS"
        else:
            MIXMODE = "DAMP"        
        
        if (abs(SCF_E - Eold) < E_conv):
            break

        Eold = SCF_E

        if SCF_ITER == maxiter:
            clean()
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

    OCCA.print_out()
    OCCB.print_out()

    mw = psi4.core.MoldenWriter(uhf)
    mw.write(prefix+'_gs.molden',uhf.Ca(),uhf.Cb(),uhf.epsilon_a(),uhf.epsilon_b(),OCCA,OCCB,True)
    psi4.core.print_out("Moldenfile written\n")

    np.savez(prefix+'_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb)
    psi4.core.print_out("Canoncical Orbitals written")

    psi4.core.set_variable('CURRENT ENERGY', SCF_E)

    return uhf
