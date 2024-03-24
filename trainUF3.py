# UF2,3 potential

import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from ase.io import read, write

from uf3.data import io, composition
from uf3.representation import bspline
from uf3.representation import process
from uf3.regression import least_squares
from uf3.forcefield import calculator

# User parameters
element_list = ['Ta', 'V', 'Cr', 'W']
degree=3

chemical_system = composition.ChemicalSystem(element_list=element_list,degree=degree)

trailing_trim = 3
leading_trim = 0

n_cores=24

splits=np.arange(0,1)

# Initialize basis
bspline_config = bspline.BSplineBasis(chemical_system,
                                      leading_trim=leading_trim,
                                      trailing_trim=trailing_trim)

# specify correct keys used in .xyz file
data_coordinator = io.DataCoordinator(
    energy_key='Energy',
    force_key='forces',
)

for split in splits:
    # Paths to
    model_path = f'results/model_{split}.json' # path where resulting model is to be saved
    data_path = f'/home/st/st_us-031400/st_st179390/traj/hea/train/train_{split}.xyz' # training data path
    test_paths = {'standard': f'/home/st/st_us-031400/st_st179390/traj/hea/valid/valid_{split}.xyz',
                  '2500k':  f'/home/st/st_us-031400/st_st179390/traj/hea/valid_2500k/total_md_{split}.xyz',
                  '4comb': f'/home/st/st_us-031400/st_st179390/traj/hea/valid_4comp/4comp_{split}.xyz'
                  } # test data path
    filename = f'df_features_{split}.h5'
    table_template = "features_{}"
    
    # Load data
    data_coordinator.dataframe_from_trajectory(data_path, prefix='dft')
    df_data = data_coordinator.consolidate()

    print("Number of energies:", len(df_data))
    print("Number of forces:", int(np.sum(df_data["size"]) * 3))

    # Analyze cut-off
    # from uf3.data import analyze
    # from tqdm.auto import tqdm

    # analyzer = analyze.DataAnalyzer(chemical_system, 
    #                             r_cut=10.0,
    #                             bins=0.01)
    # atoms_key = data_coordinator.atoms_key
    # histogram_slice = np.random.choice(np.arange(len(df_data)),
    #                                 min(1000, len(df_data)),
    #                                 replace=False)
    # df_slice = df_data[atoms_key].iloc[histogram_slice]
    # analyzer.load_entries(df_slice)
    # analysis = analyzer.analyze()

    # Compute energy and force features
    representation = process.BasisFeaturizer(bspline_config)
    client = ProcessPoolExecutor(max_workers=n_cores)

    representation.batched_to_hdf(filename,
                              df_data,
                              client,
                              n_jobs = n_cores,
                              batch_size=50,
                              progress="bar",
                              table_template=table_template)
    
    # Fit model
    regularizer = bspline_config.get_regularization_matrix(ridge_1b=1e-6,
                                                       ridge_2b=0.0,
                                                       ridge_3b=1e-4,
                                                       curvature_2b=1e-8,
                                                       curvature_3b=1e-8)
    
    model = least_squares.WeightedLinearModel(bspline_config,
                                          regularizer=regularizer)

    # Fit with energies and forces

    # Needs subset so program actually works, cannot all be within a range (thus removing one index value in between)
    # df.index.unique(level=0).intersection(subset) is used in .fit_from_file, which must not return RangeIndex(...) but Int64Index([...], dtype='int64')
    subset=np.arange(0, len(df_data))
    subset=np.delete(subset,[4])

    model.fit_from_file(filename, 
                    df_data.index[subset],
                    weight=0.5, 
                    batch_size=2500,
                    energy_key="Energy", # match key used in .xyz file
                    progress="bar")
    
    # Evaluation of test set (adding onto test_path's dataset)
    calc = calculator.UFCalculator(model)
    for name, path in test_paths.items(): 
        atoms_lst = read(path, index=':', format='extxyz')

        for atoms in atoms_lst: 
            atoms.set_calculator(calc)
            e=atoms.get_total_energy()
            f=atoms.get_forces()
            atoms.calc=None
            atoms.info['UF3_energy'] = e
            atoms.arrays['UF3_forces'] = f

        write(f'results/{name}_{split}.xyz', images=atoms_lst, format='extxyz')
    model.to_json(model_path)
    print(f"Split {split} evaluation finished")

