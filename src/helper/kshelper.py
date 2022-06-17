# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 18:30:11 2018

@author: luke
"""

import numpy as np
import time
from scipy.optimize import minimize 
import logging
import psi4



def MoldenWriter(filename, wavefunction, CaIn, CbIn, eps_a, eps_b,occa,occb):
    basisset = wavefunction.basisset()
    mol = wavefunction.molecule()
    # Header and geometry (Atom, Atom #, Z, x, y, z)
    mol_string = '[Molden Format]\n[Atoms] (AU)\n'
    for atom in range(mol.natom()):
        mol_string += f"{mol.symbol(atom):2s}  {atom+1:2d}  {int(mol.Z(atom)):3d}   {mol.x(atom):20.10f} {mol.y(atom):20.10f} {mol.z(atom):20.10f}\n"

    # Dump basis set
    mol_string += '[GTO]\n'
    for atom in range(mol.natom()):
        mol_string += f"  {atom+1:d} 0\n"
        for rel_shell_idx in range(basisset.nshell_on_center(atom)):
            abs_shell_idx = basisset.shell_on_center(atom, rel_shell_idx)
            shell = basisset.shell(abs_shell_idx)
            mol_string += f" {shell.amchar:s}{shell.nprimitive:5d}  1.00\n"
            for prim in range(shell.nprimitive):
                mol_string += f"{shell.exp(prim):20.10f} {shell.original_coef(prim):20.10f}\n"
        mol_string += '\n'

    Ca = psi4.core.Matrix.from_array(CaIn)
    Cb = psi4.core.Matrix.from_array(CbIn)
    occupation_a = psi4.core.Vector.from_array(occa)
    occupation_b = psi4.core.Vector.from_array(occb)
    epsilon_a = psi4.core.Vector.from_array(eps_a)
    epsilon_b = psi4.core.Vector.from_array(eps_b)

    aotoso = wavefunction.aotoso()
    Ca_ao_mo = psi4.core.doublet(aotoso, psi4.core.Matrix.from_array(Ca), False, False).nph
    Cb_ao_mo = psi4.core.doublet(aotoso, psi4.core.Matrix.from_array(Cb), False, False).nph
    ao_overlap = wavefunction.mintshelper().ao_overlap().np
    # Convert from Psi4 internal normalization to the unit normalization expected by Molden
    ao_normalizer = ao_overlap.diagonal()**(-1 / 2)
    Ca_ao_mo = psi4.core.Matrix.from_array([(i.T / ao_normalizer).T for i in Ca_ao_mo])
    Cb_ao_mo = psi4.core.Matrix.from_array([(i.T / ao_normalizer).T for i in Cb_ao_mo])

    # Reorder AO x MO matrix to fit Molden conventions
    '''
    Reordering expected by Molden
    P: x, y, z
    5D: D 0, D+1, D-1, D+2, D-2
    6D: xx, yy, zz, xy, xz, yz
    7F: F 0, F+1, F-1, F+2, F-2, F+3, F-3
    10F: xxx, yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz
    9G: G 0, G+1, G-1, G+2, G-2, G+3, G-3, G+4, G-4
    15G: xxxx, yyyy, zzzz, xxxy, xxxz, yyyz, zzzx, zzzy, xxyy, xxzz, yyzz, xxyz, yyxz, zzxy
    
    Molden does not handle angular momenta higher than G
    '''
    molden_cartesian_order = [
        [2,0,1,0,0,0,0,0,0,0,0,0,0,0,0], # p
        [0,3,4,1,5,2,0,0,0,0,0,0,0,0,0], # d
        [0,4,5,3,9,6,1,8,7,2,0,0,0,0,0], # f
        [0,3,4,9,12,10,5,13,14,7,1,6,11,8,2] # g
    ]

    nirrep = wavefunction.nirrep()
    count = 0 # Keeps track of count for reordering
    temp_a = Ca_ao_mo.clone() # Placeholders for original AO x MO matrices
    temp_b = Cb_ao_mo.clone()

    for i in range(basisset.nshell()):
        am = basisset.shell(i).am
        if (am == 1 and basisset.has_puream()) or (am > 1 and am < 5 and basisset.shell(i).is_cartesian()):
            for j in range(basisset.shell(i).nfunction):
                for h in range(nirrep):        
                    for k in range(Ca_ao_mo.coldim()[h]):
                        Ca_ao_mo.set(h,count + molden_cartesian_order[am-1][j],k,temp_a.get(h,count+j,k))
                        Cb_ao_mo.set(h,count + molden_cartesian_order[am-1][j],k,temp_b.get(h,count+j,k))
        count += basisset.shell(i).nfunction
        
    # Dump MO information
    if basisset.has_puream():
        mol_string += '[5D]\n[7F]\n[9G]\n\n'
    ct = mol.point_group().char_table()
    mol_string += '[MO]\n'

    mo_dim = [Ca.shape[0]]
    
    # Alphas. If Alphas and Betas are the same, then only Alphas with double occupation will be written (see line marked "***")
    mos = []
    for h in range(nirrep):
        for n in range(mo_dim[h]):
            mos.append((epsilon_a.get(h, n), (h, n)))

    # Sort mos based on energy
    def mosSort(element):
        return element[0]
    mos.sort(key=mosSort)

    for i in range(len(mos)):
        h, n = mos[i][1]
        mol_string += f" Sym= A\n Ene= {epsilon_a.get(h, n):24.10e}\n Spin= Alpha\n"
        mol_string += f" Occup= {occupation_a.get(h, n):24.10e}\n"
        for so in range(wavefunction.nso()):
            mol_string += f"{so+1:3d} {Ca_ao_mo.get(h, so, n):24.10e}\n"

    # Betas
    mos = []
    if not wavefunction.same_a_b_orbs() or wavefunction.epsilon_a() != wavefunction.epsilon_b() or not wavefunction.same_a_b_dens():
        for h in range(nirrep):
            for n in range(mo_dim[h]):
                mos.append((epsilon_b.get(h, n), (h, n)))
        mos.sort(key=mosSort)
        for i in range(len(mos)):
            h, n = mos[i][1]
            mol_string += f" Sym= A\n Ene= {epsilon_b.get(h, n):24.10e}\n Spin= Beta\n " \
                          f"Occup= {occupation_b.get(h, n):24.10e}\n"
            for so in range(wavefunction.nso()):
                mol_string += f"{so+1:3d} {Cb_ao_mo.get(h, so, n):24.10e}\n"

    # Write Molden string to file
    with open(filename,'w') as fn:
        fn.write(mol_string)    
    
    

def diag_H(H, A):
    Hp = A.T @ H @A 
    e, C2 = np.linalg.eigh(Hp)
    C = A @ C2
    return (C,e)

class Timer(object):
    def __init__(self):
        self.entries = {}
    def addStart(self,entry):
        if (entry in self.entries):
            self.entries[entry]["Start"] = time.time()
        else:
            self.entries[entry] = { }
            self.entries[entry]["Start"] = time.time()

    def addEnd(self,entry):
        if (entry in self.entries):
            self.entries[entry]["End"] = time.time()
        else:
            self.entries[entry] = { }
            self.entries[entry]["End"] = time.time()
    
    def getTime(self,entry):
        return (self.entries[entry]["End"] - self.entries[entry]["Start"])

    def printAlltoFile(self,filename):
        S = "Timings:\n"
        for x in self.entries:
            S += "{:^10} | {:5.2f}s \n".format(x,self.getTime(x))
        S+="\n"
        open(filename,"a").write(S)

class ACDIIS(object):
    def __init__(self,max_vec=6,diismode="ADIIS+CDIIS"):
        self.Fa = [] 
        self.Fb = [] 
        self.Da = [] 
        self.Db = []
        self.error = [] 
        self.DijFij  = None
        self.c_adiis = None #Coefficient from ADIIS
        self.c_cdiis = None #Coefficient from CDIIS
        self.max_vec = max_vec
        self.mode = diismode
        if (self.mode != "CDIIS") and (self.mode !="ADIIS+CDIIS"):
            raise Exception("DIIS: Don't know this mode: {}".format(self.mode))    


    def reset(self):
        self.Fa = [] 
        self.Fb = [] 
        self.Da = [] 
        self.Db = []
        self.error = [] 
        self.DijFij  = None
        self.c_adiis = None #Coefficient from ADIIS
        self.c_cdiis = None #Coefficient from CDIIS

    def add(self,Fa,Fb,Da,Db,Error):
        self.Fa.append(np.copy(Fa))
        self.Fb.append(np.copy(Fb))
        self.Da.append(np.copy(Da))
        self.Db.append(np.copy(Db))
        self.error.append(Error.ravel().copy())
        
        diis_count = len(self.Fa)

        if diis_count > self.max_vec:
            # Remove oldest vector
            del self.Fa[0]
            del self.Da[0]
            del self.Fb[0]
            del self.Db[0]
            del self.error[0]
            diis_count -= 1

    def extrapolate(self,DIISError):
        """
        first update self.c_adiis and self.c_cdiis
        """
        

        if self.mode == "ADIIS+CDIIS":
            self.solve_adiis()
            self.solve_cdiis()

            logging.info("Coeffs from ADIIS: {}".format(self.c_adiis))
            logging.info("MAX ADIIS: {}".format(np.max(self.c_adiis)))
            logging.info("Coeffs from CDIIS: {}".format(self.c_cdiis))

            if (DIISError > 0.1):
                #Just do ADIIS
                Fa = np.zeros_like(self.Fa[-1])
                Fb = np.zeros_like(self.Fb[-1])
                cs = 0.0
                exp  = [[c,index] for index,c in enumerate(self.c_adiis) if abs(c)>1E-6]
                sexp = np.sum([x[0] for x in exp])
                c= 0.0
                for i in exp:
                    c = i[0]/sexp
                    Fa += c*self.Fa[i[1]]
                    Fb += c*self.Fb[i[1]]
                    cs += c
                logging.info("DOING ADIIS: sum of coeffs: {}".format(cs))
                return (Fa,Fb)

            elif (DIISError < 0.1) and (DIISError > 1E-4):
                #Linear combine
                Fa_diis  = np.zeros_like(self.Fa[-1])
                Fb_diis  = np.zeros_like(self.Fa[-1])
                Fa_adiis = np.zeros_like(self.Fa[-1])
                Fb_adiis = np.zeros_like(self.Fa[-1])
                Fa       = np.zeros_like(self.Fa[-1])
                Fb       = np.zeros_like(self.Fa[-1])
                cs = 0.0

                exp  = [[c,index] for index,c in enumerate(self.c_adiis) if abs(c)>1E-6]
                sexp = np.sum([x[0] for x in exp])
                c = 0.0
                for i in exp:
                    c = i[0]/sexp 
                    Fa_adiis += c*self.Fa[i[1]]
                    Fb_adiis += c*self.Fb[i[1]]
                    cs += c
                logging.info("DOING ADIIS: sum of coeffs: {}".format(cs))
                exp  = [[c,index] for index,c in enumerate(self.c_cdiis) if abs(c)>1E-6]
                sexp = np.sum([x[0] for x in exp])
                c= 0.0
                for i in exp:
                    c = i[0]/sexp
                    Fa_diis += c*self.Fa[i[1]]
                    Fb_diis += c*self.Fb[i[1]]
                    cs += c
                logging.info("DOING CDIIS: sum of coeffs: {}".format(cs))
                Fa = 10*DIISError*Fa_adiis + (1-10*DIISError)*Fa_diis
                Fb = 10*DIISError*Fb_adiis + (1-10*DIISError)*Fb_diis
                return (Fa,Fb)
            elif (DIISError < 1E-4):
                Fa = np.zeros_like(self.Fa[-1])
                Fb = np.zeros_like(self.Fb[-1])
                cs = 0.0
                exp  = [[c,index] for index,c in enumerate(self.c_cdiis) if abs(c)>1E-6]
                sexp = np.sum([x[0] for x in exp])
                c= 0.0
                for i in exp:
                    c = i[0]/sexp
                    Fa += c*self.Fa[i[1]]
                    Fb += c*self.Fb[i[1]]
                    cs += c
                logging.info("CDIIS: sum of coeffs: {}".format(cs))
                return (Fa,Fb)
            else:
                raise Exception("DIIS: Thats not possible.")    

        elif self.mode=="CDIIS":
            self.solve_cdiis()
            Fa = np.zeros_like(self.Fa[-1])
            Fb = np.zeros_like(self.Fb[-1])
            cs = 0.0
            exp  = [[c,index] for index,c in enumerate(self.c_cdiis) if abs(c)>1E-6]
            sexp = np.sum([x[0] for x in exp])
            c= 0.0
            for i in exp:
                c = i[0]/sexp
                Fa += c*self.Fa[i[1]]
                Fb += c*self.Fb[i[1]]
                cs += c
            logging.info("CDIIS: sum of coeffs: {}".format(cs))
            return (Fa,Fb)
        else:
            raise Exception("DIIS: Unknown mode!")

    def update_DiF(self):
        """
        Update the ADIIS intermediates
        """
        self.DiF = np.zeros(len(self.Fa))
        for c,(ia,ib) in enumerate(zip(self.Da,self.Db)):
            self.DiF[c] = (np.trace((ia-self.Da[-1]).T @ self.Fa[-1]))+(np.trace((ib-self.Db[-1]).T @ self.Fb[-1]))


    def update_DiFj(self):
        """
        Update the ADIIS intermediates
        """
        self.DiFj = np.zeros((len(self.Fa),len(self.Fa)))
        for ci,(ia,ib) in enumerate(zip(self.Da,self.Db)):
            for cj,(ja,jb) in enumerate(zip(self.Fa,self.Fb)):
                self.DiFj[ci,cj] = np.trace((ia-self.Da[-1]).T @ (ja-self.Fa[-1])) + np.trace((ib-self.Db[-1]).T @ (jb-self.Fb[-1]))

    def compute_energy(self,x):
        c = x*x/x.dot(x)
        E  = 2* (c.T @ self.DiF)
        E += c @ self.DiFj @ c
        return E

    def compute_grad(self,x):
        x2sum = x.dot(x)
        c = x*x/x.dot(x)
        dE = 2*self.DiF 
        dE+= np.einsum('i,ik->k',c,self.DiFj) 
        dE+= np.einsum('j,kj->k',c,self.DiFj) 
        cx = np.diag(x*x2sum) - np.einsum('k,n->kn', x**2, x)
        cx *=2/x2sum**2 
        return np.einsum('k,kn->n',dE,cx)

    def solve_adiis(self):
        self.update_DiF()
        self.update_DiFj()
        res = minimize(self.compute_energy,np.ones(len(self.Fa)),method="BFGS",jac=self.compute_grad,tol=1E-9)
        c = res.x*res.x/res.x.dot(res.x)
        logging.info("Energy for opt coeffs: {} || {} || {} || {}".format(res.fun,res.success,np.sum(c),c))
        self.c_adiis = c

    def solve_cdiis(self):
        diis_count = len(self.Fa)

        if diis_count == 0:
            raise Exception("DIIS: No previous vectors.")
        if diis_count == 1:
            self.c_cdiis = [1.0]
            return


        # Build error matrix B
        B = np.empty((diis_count + 1, diis_count + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0
        for num1, e1 in enumerate(self.error):
            B[num1, num1] = np.vdot(e1, e1)
            for num2, e2 in enumerate(self.error):
                if num2 >= num1: continue
                val = np.vdot(e1, e2)
                B[num1, num2] = B[num2, num1] = val

        # normalize
        B[abs(B) < 1.e-14] = 1.e-14
        B[:-1, :-1] /= np.abs(B[:-1, :-1]).max()
        # Build residual vector
        resid = np.zeros(diis_count + 1)
        resid[-1] = -1
        # Solve pulay equations
        ci = np.dot(np.linalg.pinv(B), resid)
        self.c_cdiis = ci[:-1].copy()


def printHeader(txt,order=1):
    if order==1:
        n = len(txt)+2
        psi4.core.print_out("\n\n"+"="*n+"\n")
        psi4.core.print_out("="+txt+"=\n")
        psi4.core.print_out("="*n+"\n")

    if order==2:
        n = len(txt)
        psi4.core.print_out("\n\n"+txt+"\n")
        psi4.core.print_out("="*n+"\n\n")
        

augBas ={"firstRow" : [[0,(1.2380000000,1.0)],
   [0,(0.8840000000,1.0)],
   [0,(0.6320000000,1.0)],
   [0,(0.4510000000,1.0)],
   [0,(0.3220000000,1.0)],
   [0,(0.2300000000,1.0)],
   [0,(0.1640000000,1.0)],
   [0,(0.1170000000,1.0)],
   [0,(0.0839000000,1.0)],
   [0,(0.0599000000,1.0)],
   [0,(0.0428000000,1.0)],
   [0,(0.0306000000,1.0)],
   [0,(0.0218000000,1.0)],
   [0,(0.0156000000,1.0)],
   [0,(0.0111000000,1.0)],
   [0,(0.0079000000,1.0)],
   [0,(0.0057000000,1.0)],
   [0,(0.0040000000,1.0)],
   [0,(0.0029000000,1.0)],
   [1,(1.2380000000,1.0)],
   [1,(0.8840000000,1.0)],
   [1,(0.6320000000,1.0)],
   [1,(0.4510000000,1.0)],
   [1,(0.3220000000,1.0)],
   [1,(0.2300000000,1.0)],
   [1,(0.1640000000,1.0)],
   [1,(0.1170000000,1.0)],
   [1,(0.0839000000,1.0)],
   [1,(0.0599000000,1.0)],
   [1,(0.0428000000,1.0)],
   [1,(0.0306000000,1.0)],
   [1,(0.0218000000,1.0)],
   [1,(0.0156000000,1.0)],
   [1,(0.0111000000,1.0)],
   [1,(0.0079000000,1.0)],
   [1,(0.0057000000,1.0)],
   [1,(0.0040000000,1.0)],
   [1,(0.0029000000,1.0)],
   [2,(1.2380000000,1.0)],
   [2,(0.8840000000,1.0)],
   [2,(0.6320000000,1.0)],
   [2,(0.4510000000,1.0)],
   [2,(0.3220000000,1.0)],
   [2,(0.2300000000,1.0)],
   [2,(0.1640000000,1.0)],
   [2,(0.1170000000,1.0)],
   [2,(0.0839000000,1.0)],
   [2,(0.0599000000,1.0)],
   [2,(0.0428000000,1.0)],
   [2,(0.0306000000,1.0)],
   [2,(0.0218000000,1.0)],
   [2,(0.0156000000,1.0)],
   [2,(0.0111000000,1.0)],
   [2,(0.0079000000,1.0)],
   [2,(0.0057000000,1.0)],
   [2,(0.0040000000,1.0)],
   [2,(0.0029000000,1.0)]],
   "secondRow" : [
    [0,(9.3340000000,1.0)],
    [0,(6.6660000000,1.0)],
    [0,(4.7610000000,1.0)],
    [0,(3.4000000000,1.0)],
    [0,(2.4280000000,1.0)],
    [0,(1.7340000000,1.0)],
    [0,(1.2380000000,1.0)],
    [0,(0.8840000000,1.0)],
    [0,(0.6320000000,1.0)],
    [0,(0.4510000000,1.0)],
    [0,(0.3220000000,1.0)],
    [0,(0.2300000000,1.0)],
    [0,(0.1640000000,1.0)],
    [0,(0.1170000000,1.0)],
    [0,(0.0839000000,1.0)],
    [0,(0.0599000000,1.0)],
    [0,(0.0428000000,1.0)],
    [0,(0.0306000000,1.0)],
    [0,(0.0218000000,1.0)],
    [0,(0.0156000000,1.0)],
    [0,(0.0111000000,1.0)],
    [0,(0.0079000000,1.0)],
    [0,(0.0057000000,1.0)],
    [0,(0.0040000000,1.0)],
    [0,(0.0029000000,1.0)],
    [1,(9.3340000000,1.0)],
    [1,(6.6660000000,1.0)],
    [1,(4.7610000000,1.0)],
    [1,(3.4000000000,1.0)],
    [1,(2.4280000000,1.0)],
    [1,(1.7340000000,1.0)],
    [1,(1.2380000000,1.0)],
    [1,(0.8840000000,1.0)],
    [1,(0.6320000000,1.0)],
    [1,(0.4510000000,1.0)],
    [1,(0.3220000000,1.0)],
    [1,(0.2300000000,1.0)],
    [1,(0.1640000000,1.0)],
    [1,(0.1170000000,1.0)],
    [1,(0.0839000000,1.0)],
    [1,(0.0599000000,1.0)],
    [1,(0.0428000000,1.0)],
    [1,(0.0306000000,1.0)],
    [1,(0.0218000000,1.0)],
    [1,(0.0156000000,1.0)],
    [1,(0.0111000000,1.0)],
    [1,(0.0079000000,1.0)],
    [1,(0.0057000000,1.0)],
    [1,(0.0040000000,1.0)],
    [1,(0.0029000000,1.0)],
    [2,(9.3340000000,1.0)],
    [2,(6.6660000000,1.0)],
    [2,(4.7610000000,1.0)],
    [2,(3.4000000000,1.0)],
    [2,(2.4280000000,1.0)],
    [2,(1.7340000000,1.0)],
    [2,(1.2380000000,1.0)],
    [2,(0.8840000000,1.0)],
    [2,(0.6320000000,1.0)],
    [2,(0.4510000000,1.0)],
    [2,(0.3220000000,1.0)],
    [2,(0.2300000000,1.0)],
    [2,(0.1640000000,1.0)],
    [2,(0.1170000000,1.0)],
    [2,(0.0839000000,1.0)],
    [2,(0.0599000000,1.0)],
    [2,(0.0428000000,1.0)],
    [2,(0.0306000000,1.0)],
    [2,(0.0218000000,1.0)],
    [2,(0.0156000000,1.0)],
    [2,(0.0111000000,1.0)],
    [2,(0.0079000000,1.0)],
    [2,(0.0057000000,1.0)],
    [2,(0.0040000000,1.0)],
    [2,(0.0029000000,1.0)]
    ]
    }






