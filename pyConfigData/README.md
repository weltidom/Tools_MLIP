# pyDataHandler

Offers classes for handling ab-initio datasets in the form of [ANI](https://github.com/aiqm/ANI1x_datasets) HDF5, [MLIP](https://mlip.skoltech.ru) .cfg and extended xyz.

Trajectories (sets of atom configurations) are handled within a Pandas DataFrame. Each row describes one trajectory with list of types and list of lists for positions and forces.