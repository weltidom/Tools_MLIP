from ase.io import read
import numpy as np
import pandas as pd
from itertools import product
from mace.calculators import MACECalculator

path_model='/home/st/st_us-031400/st_st179390/mace/ANI-1ccx_multi-gpu/ANI-1ccx.model' # path to MACE model (CUDA)
path_config='graphite.cif' # path to CIF config data
path_out='test.pkl' # where pkl output will be produced


calculator = MACECalculator(model_paths=path_model, device='cuda')
conf = read(path_config, '0', format='cif')
conf.set_calculator(calculator)

# produce combinations of all factors for a and c lattice parameters
factors = np.linspace(0.25,2,1000)
f_comb = np.array(list(product(factors, factors)))

data = pd.DataFrame({
    'Factor a': f_comb[:,0],
    'Factor c': f_comb[:,1],
    'Total energy': 0,
    'Parameter a': 0,
    'Parameter c': 0
})

cell_init=conf.get_cell()
for i, row in data.iterrows():
    conf.set_cell([
        cell_init[0]*row['Factor a'],
        cell_init[1]*row['Factor a'],
        cell_init[2]*row['Factor c']
        ],
    scale_atoms=True
    )

    data.loc[i, 'Parameter a']=conf.get_cell()[0,0]
    data.loc[i, 'Parameter c']=conf.get_cell()[2,2]
    data.loc[i, 'Total energy']=conf.get_total_energy()
    
data.to_pickle(path_out)
print(f'Grid search finished and data saved under {path_out}')