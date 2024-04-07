from ase.io import read
import numpy as np
import pandas as pd
import os
import itertools

from mace.calculators import MACECalculator

directory='/Graphite_MLIP'
c_range=[ name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name)) ] # only select folders not files
models=['ANI-1ccx', 's66x8', 'ANI-1ccx+s66x8']

combinations=np.array(list(itertools.product(c_range,models)))

results=pd.DataFrame({
    'Calculation type': 'MACE',
    'Model': combinations[:,1],
    'c Factor': combinations[:,0],
    'c Parameter': 0,
    'Energy': 0
})

for i, row in results.iterrows():
    # paths to
    model = f"{row['Model']}.model" # needs to be a deployed model
    data = f"{row['c Factor']}/POSCAR"

    calculator = MACECalculator(model_path=model, device='cuda')
    atoms = read(data, format='vasp')


    atoms.set_calculator(calculator)
    e=atoms.get_total_energy()
    atoms.calc=None

    results.loc[i, 'Energy'] = e
    results.loc[i, 'c Parameter'] = atoms.get_cell()[2,2]

pd.to_pickle(results, 'graphite_MACE.pkl')