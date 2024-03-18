from ase.io import read, write
import numpy as np

from nequip.ase import NequIPCalculator


splits=np.asplits(0,10)

for split in splits:
    # paths to
    model = f'results/model_{split}.pth' # needs to be a deployed model
    data = f'~/traj/hea/valid/valid_{split}.xyz'
    output = f'eval/standard/test_{split}.xyz'


    calculator = NequIPCalculator.from_deployed_model(model_path=model)
    atoms_lst = read(data, index=':', format='extxyz')

    for atoms in atoms_lst: 
        atoms.set_calculator(calculator)
        e=atoms.get_total_energy()
        f=atoms.get_forces()
        atoms.calc=None
        atoms.info['NequIP energy'] = e
        atoms.arrays['NequIP forces'] = f

    write(output, images=atoms_lst, format='extxyz')
    print(f"Split {split} evaluation finished")