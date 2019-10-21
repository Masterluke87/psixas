# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 01:57:59 2018

@author: luke
"""
import psi4
import numpy as np
from .kshelper import diag_H,ACDIIS,Timer
import os.path
import time
import logging
import pdb

def DFTGroundState(mol,func,**kwargs):
    """
    Perform unrestrictred Kohn-Sham
    """
    psi4.core.print_out("\n\nEntering Ground State Kohn-Sham:\n"+32*"="+"\n\n")

    maxiter = int(psi4.core.get_local_option("PSIXAS","MAXITER"))
    E_conv  = float(psi4.core.get_local_option("PSIXAS","E_GS_CONV"))
    D_conv  = float(psi4.core.get_local_option("PSIXAS","D_GS_CONV"))

    prefix = kwargs["PREFIX"]

    basis = psi4.core.get_global_option('BASIS')
    wfn   = psi4.core.Wavefunction.build(mol,basis)
    aux   = psi4.core.BasisSet.build(mol, "DF_BASIS_SCF", "", "JKFIT", basis)

    sup = psi4.driver.dft.build_superfunctional(func, False)[0]
    psi4.core.be_quiet()
    mints = psi4.core.MintsHelper(wfn.basisset())
    
    sup.set_deriv(2)
    sup.allocate()

    uhf   = psi4.core.UHF(wfn,sup)
    psi4.core.reopen_outfile()
    
    S = np.asarray(mints.ao_overlap())
    T = np.asarray(mints.ao_kinetic())
    V = np.asarray(mints.ao_potential())
    H = np.zeros((mints.nbf(),mints.nbf()))
    Dip  = np.array([np.asarray(x) for x in mints.ao_dipole()])
    Quad = np.array([np.asarray(x) for x in mints.ao_quadrupole()])


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
        """
        SADNO guess, heavily inspired by the psi4 implementation
        """
        psi4.core.print_out("Creating SADNO guess\n\n")
        sad_basis_list = psi4.core.BasisSet.build(mol, "ORBITAL",
                basis,puream=True,return_atomlist=True)
        sad_fitting_list = psi4.core.BasisSet.build(mol,"DF_BASIS_SAD",
                psi4.core.get_option("SCF", "DF_BASIS_SAD"),puream=True, return_atomlist=True)
        SAD = psi4.core.SADGuess.build_SAD(wfn.basisset(), sad_basis_list)
        SAD.set_atomic_fit_bases(sad_fitting_list)
        SAD.compute_guess();
        Da = SAD.Da().np

        Dhelp = -1.0 * np.copy(Da)
        Dhelp = A.T @ S.T @ Dhelp @ S @ A

        ea,C1 = np.linalg.eigh(Dhelp)

        Ca = A @ C1
        Cb = np.copy(Ca)
        
        Cocca.np[:] = Ca[:, :nalpha]
        Coccb.np[:] = Cb[:, :nbeta]
     
        #This is the guess!
        Da  = Cocca.np @ Cocca.np.T
        Db  = Coccb.np @ Coccb.np.T
 
        """
        Ca,epsa     = diag_H(H,A)       
        Cocca.np[:] = Ca[:, :nalpha]
        Da          = Ca[:, :nalpha] @ Ca[:, :nalpha].T

        Cb,epsb     = diag_H(H,A)        
        Coccb.np[:] = Cb[:, :nbeta]
        Db          = Cb[:, :nbeta] @ Cb[:, :nbeta].T
        """
    """
    end read
    """

    # Initialize the JK object
    jk = psi4.core.JK.build(wfn.basisset(),aux=aux,jk_type=psi4.core.get_option("SCF", "SCF_TYPE"))
    glob_mem = psi4.core.get_memory()/8
    jk.set_memory(int(glob_mem*0.6))
    jk.initialize()
    jk.C_left_add(Cocca)
    jk.C_left_add(Coccb)

    Da_m = psi4.core.Matrix(nbf,nbf)
    Db_m = psi4.core.Matrix(nbf,nbf)

    mol.print_out()
    psi4.core.print_out(sup.description())
    psi4.core.print_out(sup.citation())
    psi4.core.print_out("\n\n")
    jk.print_header()
        
    diis_len = psi4.core.get_local_option("PSIXAS","DIIS_LEN")

    diis = ACDIIS(max_vec=diis_len)

    diisa_e = 1000.0
    diisb_e = 1000.0
    
    psi4.core.print_out("""
Starting SCF:
"""+13*"="+"""\n
{:>10} {:8.2E}
{:>10} {:8.2E}
{:>10} {:8.4f}
{:>10} {:8.2E}
{:>10} {:8d}
{:>10} {:8d}""".format(
    "E_CONV:",E_conv,
    "D_CONV:",D_conv,
    "DAMP:",gamma,
    "DIIS_EPS:",diis_eps,
    "MAXITER:",maxiter,
    "DIIS_LEN:",diis_len))

    myTimer = Timer()

    MIXMODE = "DAMP"
    psi4.core.print_out("\n\n{:^4} {:^14} {:^11} {:^11} {:^11} {:^4} {:^6} \n".format("# IT", "Escf", "dEscf","Derror","DIIS-E","MIX","Time"))
    psi4.core.print_out("="*80+"\n")
    diis_counter = 0
    dipole    = np.zeros(3)
    dipoleOld = np.zeros(3)
    dDipole   = np.zeros(3)
    quad      = np.zeros(6)
    quadOld   = np.zeros(6)
    dQuad     = np.zeros(6)



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
        DIIS/MIXING
        """
        diisa_e = A.T@(Fa@Da@S - S@Da@Fa)@A
        diisb_e = A.T@(Fb@Db@S - S@Db@Fb)@A
        
        diis.add(Fa,Fb,Da,Db,diisa_e+diisb_e)


        if (MIXMODE == "DIIS") and (SCF_ITER>1):
            # Extrapolate alpha & beta Fock matrices separately
            (Fa,Fb) = diis.extrapolate(DIISError)
            diis_counter += 1

            if (diis_counter >= 2*diis_len):
                diis.reset()
                diis_counter = 0
                psi4.core.print_out("\nResetting DIIS\n")

        elif (MIXMODE == "DAMP") and (SCF_ITER>1):
            #...but use damping to obtain the new Fock matrices
            Fa = (1-gamma) * np.copy(Fa) + (gamma) * FaOld
            Fb = (1-gamma) * np.copy(Fb) + (gamma) * FbOld

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

        DError = (np.sum((DaOld-Da)**2)**0.5 + np.sum((DbOld-Db)**2)**0.5)/2
        EError = (SCF_E - Eold)
        DIISError = (np.sum(diisa_e**2)**0.5 + np.sum(diisb_e**2)**0.5)/2

        dipoleOld = dipole.copy()
        quadOld = quad.copy()
        dipole[0] = -np.einsum('mn,mn',Da+Db,Dip[0])
        dipole[1] = -np.einsum('mn,mn',Da+Db,Dip[1])
        dipole[2] = -np.einsum('mn,mn',Da+Db,Dip[2])

        quad = np.einsum("nm,inm->i",Da+Db,Quad)

        dDipole = dipoleOld-dipole
        dQuad   = quadOld - quad 
        logging.info("dDipole {:4.2f} {:4.2f} {:4.2f} ".format(dDipole[0],dDipole[1],dDipole[2]))
        logging.info(("dQuad "+"{:4.2f} "*6).format(*[x for x in dQuad]))
        logging.info("Alpha: {} eV".format((epsa[nalpha]-epsa[nalpha-1])*27.211386)) 
        logging.info("Beta:  {} eV".format((epsb[nbeta]-epsb[nbeta-1])*27.211386)) 



        """
        OUTPUT
        """
        myTimer.addEnd("SCF")
        psi4.core.print_out(" {:3d} {:14.8f} {:11.3E} {:11.3E} {:11.3E} {:^4} {:6.2f} {:2d} \n".format(SCF_ITER,
             SCF_E,
             EError,
             DError,
             DIISError,
             MIXMODE,
             myTimer.getTime("SCF"),
             len(diis.Fa)))
                  
        psi4.core.flush_outfile()
        if (abs(DIISError) < diis_eps):
            MIXMODE = "DIIS"
        else:
            MIXMODE = "DAMP"        
        
        if (abs(EError) < E_conv) and (abs(DError)<D_conv):
            break

        Eold = SCF_E

        if SCF_ITER == maxiter:
            psi4.core.clean()
            occa = np.zeros(nbf,dtype=np.float)
            occb = np.zeros(nbf,dtype=np.float)
            occa[:nalpha] = 1.0
            occb[:nbeta]  = 1.0
            np.savez(prefix+'_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb,epsa=epsa,epsb=epsb)
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
    psi4.core.print_out("Canoncical Orbitals written\n\n")                        

    psi4.core.set_variable('CURRENT ENERGY', SCF_E)
    psi4.core.set_variable('GS ENERGY', SCF_E)


    return uhf
