GravothermalSIDM
================

GravothermalSIDM solves a set of gravothermal fluid equations to obtain the evolution of an isolated spherical halo comprised of self-interacting dark matter.

Author: [Kimberly Boddy](https://kboddy.github.io)

Contributors: Hiroya Nishikawa, [Sophia Gad-Nasr](https://github.com/SophiaNasr?tab=repositories), [Laura Sagunski](https://dmgw.space/)

[![GPLv3 license](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html#license-text)
[![Generic badge](https://img.shields.io/badge/arXiv-2204.06568-a30f0f.svg)](https://arxiv.org/abs/2204.06568)
[![Generic badge](https://img.shields.io/badge/arXiv-2312.09296-a30f0f.svg)](https://arxiv.org/abs/2312.09296)

Main features
-------------

GravothermalSIDM starts with a specified initial halo, divided into discrete radial bins, and tracks its gravothermal evolution.
The evolution proceeds by alternating between conducting heat across the radial bins over a short time and enforcing quasistatic equilibrium.
Snapshots of the halo properties are saved as the halo evolves.

Options for initial halo profiles:
* NFW
* Hernquist
* Isothermal

Options for the SIDM cross section:
* constant (velocity independent)
* power-law velocity dependence
* Yukawa scattering (Born approximation)
  * t- and u-channel
  * t-channel only

Getting started
---------------

Download the code from GitHub:
```console
git clone https://github.com/kboddy/GravothermalSIDM.git
```

Install the following Python package dependencies:
```console
pip install numpy scipy astropy h5py numba mpmath
```

As a simple example, run
```console
python runHaloEvolution.py
```

The Jupyter notebook `tutorial.ipynb` displays the documentation for the setup and run options.

Usage
-----

GravothermalSIDM is free software, licensed under [GPLv3](LICENSE) with standard [terms](TERMS).

If you use this software, please cite the following in your publications:
* [Universal Gravothermal Evolution of Isolated Self-Interacting Dark Matter Halos for Velocity-Dependent Cross Sections (arXiv: 2204.06568)](https://arxiv.org/abs/2204.06568)
* [On the Late-Time Evolution of Velocity-Dependent Self-Interacting Dark Matter Halos (arXiv: 2312.09296)](https://arxiv.org/abs/2312.09296)

Additionally, feel free to cite [Accelerated core collapse in tidally stripped self-interacting dark matter halos (arXiv: 1901.00499)](https://arxiv.org/abs/1901.00499), which developed the code on which GravothermalSIDM is based.

The numerical implementation of the gravothermal evolution is based on the detailed description provided in [Jason Pollack's senior undergraduate thesis](http://arks.princeton.edu/ark:/88435/dsp01t435gg18w).
