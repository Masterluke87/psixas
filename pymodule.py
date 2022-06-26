#
# @BEGIN LICENSE
#
# psixas by Psi4 Developer, a plugin to:
#
# Psi4: an open-source quantum chemistry software package
#
# Copyright (c) 2007-2017 The Psi4 Developers.
#
# The copyrights for code used from other parties are included in
# the corresponding files.
#
# This file is part of Psi4.
#
# Psi4 is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3.
#
# Psi4 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with Psi4; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#
# @END LICENSE
#
import sys
import numpy as np
import psi4
import psi4.driver.p4util as p4util
from psi4.driver.procrouting import proc_util
from psixas.src.dft.ksgs import DFTGroundState
from psixas.src.dft.ksex import DFTExcitedState
from psixas.src.helper.spec import CalcSpec
from psixas.src.helper.kshelper import printHeader
import logging

logging.basicConfig(filename='additional.log',level=logging.CRITICAL,filemode='w', format='%(name)s -%(levelname)s - %(message)s')




def run_psixas(name, **kwargs):
    r"""Function encoding sequence of PSI module and plugin calls so that
    psixas can be called via :py:func:`~driver.energy`. For post-scf plugins.

    >>> energy('psixas')

    """
    lowername = name.lower()
    kwargs = p4util.kwargs_lower(kwargs)

    #print the banner
    printBanner()

    mol  = kwargs["molecule"]
    func = kwargs["functional"]

    tmp = psi4.core.get_local_option("PSIXAS","MODE")

    if (psi4.core.has_local_option_changed("SCF","SCF_TYPE")==False):
        psi4.core.set_local_option("SCF","SCF_TYPE","MEM_DF")
    mode = tmp.split("+")
    if not(all([x in ["GS","LOC","EX","SPEC"] for x in mode])):
        raise Exception("Wrong mode, possible values are GS, LOC, EX, SPEC.")

    if "GS" in mode:
        DFTGroundState(mol,func,PREFIX=psi4.core.get_local_option("PSIXAS","PREFIX"))

    if "LOC" in mode:
        loc_sub = np.array(psi4.core.get_local_option("PSIXAS","LOC_SUB"),dtype=np.int)
        printHeader("Entering Localization")
        psi4.core.be_quiet()
        wfn     = psi4.core.Wavefunction.build(mol,psi4.core.get_global_option('BASIS'))

        nbf = wfn.nso()
        sup = psi4.driver.dft.build_superfunctional(func, False)[0]
        sup.set_deriv(2)
        sup.allocate()

        uhf   = psi4.core.UHF(wfn,sup)

        prefix = psi4.core.get_local_option("PSIXAS","PREFIX")
        Ca = np.load(prefix+"_gsorbs.npz")["Ca"]
        Cb = np.load(prefix+"_gsorbs.npz")["Cb"]
        occa = np.load(prefix+"_gsorbs.npz")["occa"]
        occb = np.load(prefix+"_gsorbs.npz")["occb"]
        epsa = np.load(prefix+"_gsorbs.npz")["epsa"]
        epsb = np.load(prefix+"_gsorbs.npz")["epsb"]


        locCa = psi4.core.Matrix(wfn.nso(),len(loc_sub))
        locCb = psi4.core.Matrix(wfn.nso(),len(loc_sub))

        locCa.np[:] = np.copy(Ca[:,loc_sub])
        locCb.np[:] = np.copy(Cb[:,loc_sub])

        LocalA = psi4.core.Localizer.build("PIPEK_MEZEY", wfn.basisset(), locCa )
        LocalB = psi4.core.Localizer.build("PIPEK_MEZEY", wfn.basisset(), locCb )
        psi4.core.reopen_outfile()

        LocalA.localize()
        LocalB.localize()

        Ca[:,loc_sub] = LocalA.L
        Cb[:,loc_sub] = LocalB.L

        np.savez(prefix+'_gsorbs',Ca=Ca,Cb=Cb,occa=occa,occb=occb)
        psi4.core.print_out("Localized Orbitals written\n")

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
        mw.write(prefix+'_loc.molden',uhf.Ca(),uhf.Cb(),uhf.epsilon_a(),uhf.epsilon_b(),OCCA,OCCB,True)
        psi4.core.print_out("Moldenfile written\n")

    orbitals = []

    if ("EX" in mode):
        orbs   = psi4.core.get_local_option("PSIXAS","ORBS")
        occs   = psi4.core.get_local_option("PSIXAS","OCCS")
        freeze = psi4.core.get_local_option("PSIXAS","FREEZE")
        spin   = psi4.core.get_local_option("PSIXAS","SPIN")
        ovl    = psi4.core.get_local_option("PSIXAS","OVL")

        lens = [len(x) for x in [orbs,occs,freeze,spin,ovl]]
        if  len(list((set(lens))))>1:
            raise Exception("Input arrays have inconsistent length"+" ".join(str(lens)))
        for i in range(len(orbs)):
            orbitals.append({"orb" : orbs[i],"spin": spin[i].lower(),"occ" : occs[i], "frz" : freeze[i]=="T","DoOvl":ovl[i] == "T" })
        DFTExcitedState(mol,func,orbitals)

    if ("SPEC" in mode):
        CalcSpec(mol,func)







    #psixas_wfn = psi4.core.plugin('psixas.so', wfn)

    return 0 #psixas_wfn


# Integration with driver routines
psi4.driver.procedures['energy']['psixas'] = run_psixas


def printBanner():
    Banner = '''
                    ___     ___      ___    __  __     ___      ___   
                   | _ \   / __|    |_ _|   \ \/ /    /   \    / __|  
                   |  _/   \__ \     | |     >  <     | - |    \__ \  
                  _|_|_    |___/    |___|   /_/\_\    |_|_|    |___/  
                _| """ | _|"""""| _|"""""| _|"""""| _|"""""| _|"""""| 
                "`-0-0-' "`-0-0-' "`-0-0-' "`-0-0-' "`-0-0-' "`-0-0-' 
                
                         An X-Ray Absorption Plugin for PSI4
    '''
    psi4.core.print_out(Banner)
