/*
 * @BEGIN LICENSE
 *
 * psixas by Psi4 Developer, a plugin to:
 *
 * Psi4: an open-source quantum chemistry software package
 *
 * Copyright (c) 2007-2017 The Psi4 Developers.
 *
 * The copyrights for code used from other parties are included in
 * the corresponding files.
 *
 * This file is part of Psi4.
 *
 * Psi4 is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, version 3.
 *
 * Psi4 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License along
 * with Psi4; if not, write to the Free Software Foundation, Inc.,
 * 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * @END LICENSE
 */

#include "psi4/psi4-dec.h"
#include "psi4/libpsi4util/PsiOutStream.h"
#include "psi4/liboptions/liboptions.h"
#include "psi4/libmints/wavefunction.h"
#include "psi4/libpsio/psio.hpp"
#include "psi4/libpsio/psio.hpp"

namespace psi{ namespace psixas {

extern "C" PSI_API
int read_options(std::string name, Options& options)
{
    if (name == "PSIXAS"|| options.read_globals()) {
        /*- The amount of information printed to the output file -*/
        options.add_str("MODE","GS");
        options.add_str("PREFIX", "KS");
        options.add_double("DAMP", 0.8);
        options.add_double("DIIS_EPS", 0.1);
        options.add_int("DIIS_LEN", 6);
	options.add_double("VSHIFT",0.0);
	options.add_int("MAXITER",100);
        options.add_array("ORBS");
        options.add_array("OCCS");
        options.add_array("SPIN");
        options.add_array("FREEZE");
	options.add_array("OVL");
        options.add_array("LOC_SUB"); 
        
    }

    return true;
}

extern "C" PSI_API
SharedWavefunction psixas(SharedWavefunction ref_wfn, Options& options)
{
    int print = options.get_int("PRINT");

    /* Your code goes here */

    // Typically you would build a new wavefunction and populate it with data
    return ref_wfn;
}

}} // End namespaces

