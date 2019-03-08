# PSIXAS
A PSI4 plugin to calculate X-ray absorption spectra (NEXAFS, PP-NEXAFS, XPS). The implemented methods are based on the Transition-Potential and Delta-Kohn-Sham approach. For details, please refer to the documentation (will be published soon).

Parts of the program were inspired by the [psi4numpy](https://github.com/psi4/psi4numpy) package

## Install
To install and use psi4xas, you will need PSI4 already installed on your computer. You can then checkout the repository:
``` bash
cd /path/to/psi4Plugins/
git clone https://github.com/Masterluke87/psixas/
```
then compile the Plugin:

``` bash
cd psixas
$(psi4 --plugin-compile)
make 
```
## Run the Plugin
To run th plugin you just have to set the PYTHONPATH variable:
``` bash
export PYTHONPATH=/path/to/psi4Plugins
```


## Example O-K edge of water
``` python

```
