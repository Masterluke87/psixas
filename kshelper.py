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






