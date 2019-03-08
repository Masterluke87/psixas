#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 10:14:48 2018

@author: luke
"""


from matplotlib import pylab as plt
import numpy as np
import pickle




def GaussBroad(X,Y,xmin,xmax,Npts,FWHM=1.5,FWHM2=1.5):
    """
    Function to broad a stick spectrum
    """
    XVals = np.array([X])[np.where(np.array([X])<=xmax)]
    YVals = np.array([Y])[np.where(np.array([X])<=xmax)]

    Xgrid = np.linspace(xmin,xmax,Npts)
    Ygrid = np.zeros(len(Xgrid))

    m =(FWHM2-FWHM)/(xmax-xmin)
    n =FWHM-m*xmin

    for i,j in zip(XVals,YVals):
        sigma = (m*i+n)/(2*np.sqrt(2*np.log(2)))
        tmpY  = 1/(np.sqrt(2*np.pi)*np.float(sigma))*np.exp(-(Xgrid-np.float(i))**2/(2*np.float(sigma)**2))
        if np.max(tmpY)>0:
            tmpY /=np.max(tmpY)
            tmpY *= j
            Ygrid += tmpY

    return Xgrid,Ygrid


spec = pickle.load(open("WATER_b.spectrum","rb"))
Ints = spec["En"]*27.211385*(spec["Dx"]**2+spec["Dy"]**2+spec["Dz"]**2)
X,Y  = GaussBroad(spec["En"]*27.211385,Ints,535,550,1000,0.8,0.8)

scale = 1/np.max(Y)

plt.plot(X,Y*scale)
plt.vlines(spec["En"]*27.211385,0,Ints*scale)
plt.yticks([])
plt.ylabel("Absorption (arb. units)",fontsize=16)
plt.xlabel("Photon Energy [eV] ",fontsize=16)
plt.xlim((535,550))
plt.ylim((0,1.2))
plt.show()




