# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 18:30:11 2018

@author: luke
"""

import numpy as np
import time
from scipy.optimize import minimize 
import logging




def diag_H(H, A):
    Hp = A.dot(H).dot(A)
    e, C2 = np.linalg.eigh(Hp)
    C = A.dot(C2)
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


class EDIIS_helper(object):
    def __init__(self,max_vec=6):
        self.Fa = []
        self.Fb = []
        self.Da = [] 
        self.Db = []
        self.E  = []
        self.DijFij  = None
        self.c       = None
        self.max_vec = max_vec

    def update_DijFij(self):
        self.DijFij = np.zeros((len(self.Fa),len(self.Fa)))
        for ci in range(len(self.Fa)):
            for cj in range(len(self.Fa)):
                self.DijFij[ci,cj] = 0.5 + np.trace((self.Fa[ci]-self.Fa[cj])@(self.Da[ci]-self.Da[cj]))+0.5*np.trace((self.Fb[ci]-self.Fb[cj])@ (self.Db[ci]-self.Db[cj])) 

    def compute_energy(self,c):
        x = c*c/ c.dot(c)
        E = x.dot(self.E) + (x@self.DijFij@x)
        return E


    def add(self,E,Fa,Fb,Da,Db):
        self.Fa.append(np.copy(Fa))
        self.Fb.append(np.copy(Fb))
        self.Da.append(np.copy(Da))
        self.Db.append(np.copy(Db))
        self.E.append(np.copy(E))
        diis_count = len(self.Fa)
        if diis_count > self.max_vec:
            # Remove oldest vector
            del self.Fa[0]
            del self.Da[0]
            del self.Fb[0]
            del self.Db[0]
            del self.E[0]
            diis_count -= 1

        self.update_DijFij()
        c = np.ones(len(self.Fa))
        c /= np.sum(c)
        E = self.compute_energy(c)
        #pdb.set_trace()
        print("Energy for same coeffs: {}".format(E))
        res = minimize(self.compute_energy,c,method="L-BFGS-B")
        c = res.x
        E = self.compute_energy(c)
        print("Energy for opt coeffs: {} || {} ".format(E,res.success))
        c = c*c/c.dot(c)
        print(c,sum(c))
        self.c = c

    def extrapolate(self):
        Fa = np.zeros_like(self.Fa[-1])
        Fb = np.zeros_like(self.Fb[-1])
        for c,i in enumerate(self.c):
            if i>1E-6:
                Fa += i*self.Fa[c]
                Fb += i*self.Fb[c]
        return (Fa,Fb)
   


class ADIIS_helper(object):
    def __init__(self,max_vec=6):
        self.Fa = []
        self.Fb = []
        self.Da = [] 
        self.Db = []
        self.DiF  = None 
        self.DiFj = None
        self.c    = None
        self.max_vec = max_vec


    def update_DiF(self):
        self.DiF = np.zeros(len(self.Fa))
        for c,(ia,ib) in enumerate(zip(self.Da,self.Db)):
            self.DiF[c] = (np.trace((ia-self.Da[-1]).T @ self.Fa[-1]))+(np.trace((ib-self.Db[-1]).T @ self.Fb[-1]))


    def update_DiFj(self):
        self.DiFj = np.zeros((len(self.Fa),len(self.Fa)))

        for ci,(ia,ib) in enumerate(zip(self.Da,self.Db)):
            for cj,(ja,jb) in enumerate(zip(self.Fa,self.Fb)):
                self.DiFj[ci,cj] = np.trace((ia-self.Da[-1]).T @ (ja-self.Fa[-1])) + np.trace((ib-self.Db[-1]).T @ (jb-self.Fb[-1]))

    def extrapolate(self):
        Fa = np.zeros_like(self.Fa[-1])
        Fb = np.zeros_like(self.Fb[-1])
        cs = 0.0
        for c,i in enumerate(self.c):
            if abs(c)>1E-6:
                Fa += i*self.Fa[c]
                Fb += i*self.Fb[c]
                cs += i
        logging.info("ADIIS: sum of coeffs: {}".format(cs))
        return (Fa,Fb)
        
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

    def test_grad(self,x):
        x0 = np.copy(x)
        dfx0 = np.zeros_like(x0)
        for i in range(len(x0)):
            x1 = x0.copy()
            x1[i] += 1E-4
            dfx0[i] = (self.compute_energy(x1)-self.compute_energy(x0))*1E+4
        logging.info('Grad: {}'.format(str(dfx0-self.compute_grad(x0))))
        
        
    def add(self,Fa,Fb,Da,Db):
        self.Fa.append(np.copy(Fa))
        self.Fb.append(np.copy(Fb))
        self.Da.append(np.copy(Da))
        self.Db.append(np.copy(Db))
        
        diis_count = len(self.Fa)
        if diis_count > self.max_vec:
            # Remove oldest vector
            del self.Fa[0]
            del self.Da[0]
            del self.Fb[0]
            del self.Db[0]
            diis_count -= 1

        self.update_DiF()
        self.update_DiFj()
        c = np.ones(len(self.Fa))
        c /= np.sum(c)
        E = self.compute_energy(c)
        print("Energy for same coeffs: {}".format(E))
        res = minimize(self.compute_energy,c,method="BFGS",jac=self.compute_grad,tol=1E-9)
        c = res.x*res.x/res.x.dot(res.x)
        print("Energy for opt coeffs: {} || {} || {} || {}".format(res.fun,res.success,np.sum(c),c))
        self.c = c


class ACDIIS(object):
    def __init__(self,max_vec=6):
        self.Fa = [] 
        self.Fb = [] 
        self.Da = [] 
        self.Db = []
        self.error = [] 
        self.DijFij  = None
        self.c_adiis = None #Coefficient from ADIIS
        self.c_cdiis = None #Coefficient from CDIIS
        self.max_vec = max_vec

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
        self.solve_adiis()
        self.solve_cdiis()

        logging.info("Coeffs from ADIIS: {}".format(self.c_adiis))
        logging.info("MAX ADIIS: {}".format(np.max(self.c_adiis)))
        logging.info("Coeffs from CDIIS: {}".format(self.c_cdiis))

        """
        If ADIIS has a very large coeff, we are doing CDIIS
        
        if np.max(self.c_adiis) > 0.95:
            Fa = np.zeros_like(self.Fa[-1])
            Fb = np.zeros_like(self.Fb[-1])
            cs = 0.0
            exp  = [[c,index] for index,c in enumerate(self.c_cdiis) if abs(c)>1E-6]
            sexp = np.sum([x[0] for x in exp])
            c = 0.0
            for i in exp:
                c = i[0]/sexp
                Fa += c*self.Fa[i[1]]
                Fb += c*self.Fb[i[1]]
                cs += c
            logging.info("CDIIS: sum of coeffs: {}".format(cs))
            return (Fa,Fb)
        """


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




    def update_DiF(self):
        self.DiF = np.zeros(len(self.Fa))
        for c,(ia,ib) in enumerate(zip(self.Da,self.Db)):
            self.DiF[c] = (np.trace((ia-self.Da[-1]).T @ self.Fa[-1]))+(np.trace((ib-self.Db[-1]).T @ self.Fb[-1]))


    def update_DiFj(self):
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





class DIIS_helper(object):
    """
    A helper class to compute DIIS extrapolations.
    Notes
    -----
    Equations taken from [Sherrill:1998], [Pulay:1980:393], & [Pulay:1969:197]
    Algorithms adapted from [Sherrill:1998] & [Pulay:1980:393]
    """

    def __init__(self, max_vec=6):
        """
        Intializes the DIIS class.
        Parameters
        ----------
        max_vec : int (default, 6)
            The maximum number of vectors to use. The oldest vector will be deleted.
        """
        self.error = []
        self.vector = []
        self.c = None
        self.max_vec = max_vec

    def add(self, state, error):
        """
        Adds a set of error and state vectors to the DIIS object.
        Parameters
        ----------
        state : array_like
            The state vector to add to the DIIS object.
        error : array_like
            The error vector to add to the DIIS object.
        Returns
        ------
        None
        """

        error = np.array(error)
        state = np.array(state)
        if len(self.error) > 1:
            if self.error[-1].shape[0] != error.size:
                raise Exception("Error vector size does not match previous vector.")
            if self.vector[-1].shape != state.shape:
                raise Exception("Vector shape does not match previous vector.")

        self.error.append(error.ravel().copy())
        self.vector.append(state.copy())
        
        # Limit size of DIIS vector
        diis_count = len(self.vector)
        if diis_count > self.max_vec:
            # Remove oldest vector
            del self.vector[0]
            del self.error[0]
            diis_count -= 1

    def extrapolate(self):
        """
        Performs the DIIS extrapolation for the objects state and error vectors.
        Parameters
        ----------
        None
        Returns
        ------
        ret : ndarray
            The extrapolated next state vector
        """

        # Limit size of DIIS vector
        diis_count = len(self.vector)

        if diis_count == 0:
            raise Exception("DIIS: No previous vectors.")
        if diis_count == 1:
            return self.vector[0]

        if diis_count > self.max_vec:
            # Remove oldest vector
            del self.vector[0]
            del self.error[0]
            diis_count -= 1

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
        # combination of previous fock matrices
        V  = np.zeros_like(self.vector[-1])
        cs = 0.0
        for num, c in enumerate(ci[:-1]):
            if abs(c)>=1E-6:
                V += c * self.vector[num]
                cs += c
        logging.info("DIIS: sum of coeffs: {}".format(cs))
        return V
