# SXAE

This Python script creates a finite element model of a fibre-reinforced composite based on ultra-low resolution X-ray scattering tensor tomography. It therefore creates a voxel-based finite element mesh based on the scattering data. Further, for each of the eight integration points of each element the 30 independent material stiffness matrix components are updated based on the sub-voxel fibre orientation and the sub-voxel fibre volume fraction. The original stiffness matrix is computed with the Mori-Tanaka homogenisation based on Eshelby tensor for an ellipsoidal inclusion, representing short carbon fibres. The 30 independent stiffness matrix components are then output in Fortran format to be used in an Abaqus Subroutine.

This code has been used for the listed publications:

R. M. Auenhammer, J. Kim, C. Oddy, L. P. Mikkelsen, F. Marone, M. Stampanoni, L. E. Asp, X-ray scattering tensor tomography based finite element modelling of fibre reinforced composites, npj Computational Materials 10 (2024). doi: 10.1038/s41524-024-01234-5
https://www.nature.com/articles/s41524-024-01234-5

Following packages must be installed:
python 3.11.8; numpy 1.26.4; scipy 1.12.0; matplotlib 3.8.0; numba 0.59.1; seaborn 0.12.2

The code can be also run on Code Ocean:
https://codeocean.com/capsule/8442429/tree/v4
