# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 18:30:11 2018

@author: luke
"""

import numpy as np
import time

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
        V = np.zeros_like(self.vector[-1])
        for num, c in enumerate(ci[:-1]):
            V += c * self.vector[num]

        return V
