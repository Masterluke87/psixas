Installation 
============

The most convenient way to get PSIXAS to run is to use Psi4 via 
conda or miniconda. Let's assume you created a new environment:

.. code-block:: bash

    conda create -n p4env psi4 psi4-dev -c psi4
    conda activate p4env

this will install the most recent stable version of Psi4.

Then, you can checkout the PSIXAS repository into the directory where all your Psi4
plugins are located:

.. code-block:: bash

    cd /path/to/psi4Plugins
    git clone https://github.com/Masterluke87/psixas
    cd psixas
    $(psi4 --plugin-compile)
    make

If you do *not* want to build the master branch version, 
you can checkout another branch, for example:

.. code-block:: bash

    git checkout PSIXAS-1.0

After the plugin is installed,
one needs to export the path that contains the plugin:

.. code-block:: bash

    export PYTHONPATH=/path/to/psi4Plugins

The installation is complete, and PSIXAS is ready to run!
