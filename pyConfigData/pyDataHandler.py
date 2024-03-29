import warnings
try:
    import numpy as np
    import pandas as pd
    import ase.io as io
except ImportError:
    warnings.warn('NumPy, Pandas and ASE packages installations are required')

try:
    import dataloader as dl
except ImportError:
    # allow the ImportError to pass silently
    dl = None


class Cfg:
    def __init__(self, file_name:str, path:str):
        self.name = file_name
        self.path = path

    def read(self):
        '''Read file and save lines in list self.lines.'''
        lines = []
        with open(self.path, 'r') as file:
            for line in file:
                lines.append(line.rstrip())
        self.lines = lines
        self.no_configs = self.lines.count('BEGIN_CFG')
        
    def parse(self):
        '''Parse previously read self.lines to DataFrame self.data.'''
        config = 0
        data_lst = []
        
        for i, line in enumerate(self.lines):
            if 'BEGIN_CFG' in line:
                config+=1
                data = pd.DataFrame({
                    'Name': self.name,
                    'Lattice': [[]],
                    'Configuration': config,
                    'Config. size': 0,
                    'Energy': 0,
                    'Atom': [[]],
                    'Position': [[]],
                    'Force': [[]]
                    })
            elif 'Size' in line:
                size = int(self.lines[i+1]) # number of atoms in config
                data['Config. size'] = size
            elif ('SuperCell' in line) or ('Supercell' in line):
                data['Lattice']=[[
                    list(filter(lambda x: len(x) > 0,self.lines[i+1].split(' '))),
                    list(filter(lambda x: len(x) > 0,self.lines[i+2].split(' '))),
                    list(filter(lambda x: len(x) > 0,self.lines[i+3].split(' ')))
                    ]]
            elif 'AtomData' in line:
                atoms, pos, force = [], [], []
                for entry in range(1,size+1):
                    values = list(filter(lambda x: len(x) > 0,self.lines[i+entry].strip('\t').split(' '))) # clean up string, turn into list, remove empty entries
                    atoms.append(int(values[1]))
                    pos.append(values[2:5])
                    force.append(values[5:8])
                data['Atom']=[atoms]
                data['Position']=[pos]
                data['Force']=[force]
            elif 'Energy' in line:
                data['Energy'] = float(self.lines[i+1].strip('\t'))
            elif 'END_CFG' in line:
                data_lst.append(data)

        self.data = pd.concat(data_lst)

    def write(self, df:pd.DataFrame):
        '''Write data to .cfg file for use with MLIP.'''
        cols = df.columns
        with open(f'{self.path}.xyz', 'w') as f:
            for index, row in df.iterrows():
                f.write('BEGIN_CFG\n')
                f.write(f'Size\n\t {row["Config. size"]}\n')
                if 'Lattice' in cols:
                    lat = row['Lattice']
                    f.write('SuperCell')
                    f.write(f'\t \t {lat[0][0]:>10}{lat[0][1]:>10}{lat[0][2]:>10}\n')
                    f.write(f'\t \t {lat[1][0]:>10}{lat[1][1]:>10}{lat[1][2]:>10}\n')
                    f.write(f'\t \t {lat[2][0]:>10}{lat[2][1]:>10}{lat[2][2]:>10}\n')
                f.write(f'AtomData: {"id":>10}{"type":>10}{"cartes_x":>10}{"cartes_y":>10}{"cartes_z":>10}')
                if 'Force' in cols:
                    f.write(f'{"fx":>10}{"fy":>10}{"fz":>10}\n')
                    id=1
                    for atom, pos, forces in zip(row['Atom'], row['Position'],row['Force']):
                        f.write(f'{id:>5}{atom:>5}{pos[0]:>10}{pos[1]:>10}{pos[2]:>10}{forces[0]:>10}{forces[1]:>10}{forces[2]:>10}\n')
                        id+=1
                else:
                    f.write('\n')
                    id=1
                    for atom, pos in zip(row['Atom'], row['Position']):
                        f.write(f'{id:>5}{atom:>5}{pos[0]:>10}{pos[1]:>10}{pos[2]:>10}\n')
                        id+=1
                f.write(f'Energy\n \t \t \t {row["Energy"]}\n')
                if 'Stress' in cols:
                    f.write(f'PlusStress {"xx":>10}{"yy":>10}{"zz":>10}{"yz":>10}{"xz":>10}{"xy":>10}\n')
                    f.write(f'{"":>10}{row["Stress"][0]}{row["Stress"][1]}{row["Stress"][2]}{row["Stress"][3]}{row["Stress"][4]}{row["Stress"][5]}')
                f.write('END_CFG\n \n')

class Hdf5:
    '''Class for handling HDF-5 data format provided by ANI-1. Only energy and position implemented yet (coupled cluster data set).'''
    def __init__(self, path:str, name:str, data_keys:list):
        self.name = name
        self.path = path
        self.keys = data_keys # keys indicate which type of data set to extract, e.g. 'ccsd(t)_cbs.energy' for coupled cluster data

    def parse(self):
        '''Parse H5 file into self.data.'''
        if dl is None:
            warnings.warn('Install dataloader.py to parse H5 file from https://github.com/aiqm/ANI1x_datasets/blob/master/dataloader.py')

        df=[]
        config=1

        for data in dl.iter_data_buckets(self.path,keys=self.keys):
            conform=1
            self.properties=list(data.keys()) # contained properties e.g. energy, position, forces 
            e_desc = next(x for x in self.properties if '.energy' in x)

            for energy, position in zip(data[e_desc],data['coordinates']):
                entry = pd.DataFrame({
                                'Configuration': config,
                                'Conformation': conform,
                                'Config. size': len(data['atomic_numbers']),
                                'Energy': energy,
                                'Atom': [data['atomic_numbers']],
                                'Position': [position]
                                })
                df.append(entry)
                conform+=1
            config+=1

        self.data=pd.concat(df)

    def convert_ha_ev(self):
        '''Convert energy values from Hartree to eV.'''
        self.data['Energy'] = self.data['Energy']*27.211386245988 # convert Ha to eV

class Xyz:
    '''Class for handling extended-xyz data format utilized by MACE.'''
    def __init__(self, folder:str, name:str, assignment:dict={}):
        self.name = name
        self.assign = assignment # dictionary with number to atom species assignment
        self.path = f'{folder}/{name}'

    @staticmethod
    def unpack(lst):
        '''Unpack list (2 levels) and turn into string.'''
        unpacked = ''
        for entry in lst:
            unpacked = f'{unpacked} {" ".join(entry)}'
        return unpacked[1:]

    def type_to_symbol(self, no):
        '''Method for turning atom type numbers to atomic symbols.'''
        if len(self.assign)<1:
            return no
        for type, symbol in self.assign.items():
            if type==no:
                species=symbol
        return species
    
    def read(self, desc:str='', inference:bool=False, energy_key:str='', force_key:str=''):
        '''Read and parse data from extended XYZ file. Resulting dataframe saved under self.data.'''
        atoms_lst = io.read(
            self.path,
            index=':',
            format='extxyz'
        )
        
        data_lst = []
        for i, atom in enumerate(atoms_lst):
            entry_dict = atom.todict()
            entry = pd.DataFrame({
                    'Type': desc,
                    'Name': ''.join(np.unique(atom.get_chemical_symbols())),
                    #'Lattice': [entry_dict['cell']],
                    'Configuration': i+1,
                    'Config. size': len(atom.get_positions()),
                    #'Energy': entry_dict['info']['energy'],
                    #'Energy_Inference': entry_dict['info'][energy_key],
                    'Atom': [atom.get_chemical_symbols()],
                    'Position': [entry_dict['positions']],
                    #'Force': [entry_dict['forces']],
                    #'Force_Inference': [entry_dict[force_key]],
                    })
            if 'energy' in entry_dict['info']:
                entry['Energy']=entry_dict['info']['energy']
            elif 'Energy' in entry_dict['info']:
                entry['Energy']=entry_dict['info']['Energy']

            if inference:
                entry['Energy_Inference'] = entry_dict['info'][energy_key]
                if len(force_key)>0:
                    entry['Force_Inference'] = [entry_dict[force_key]]
            if ('T' in entry_dict['pbc']):
                entry['Lattice'] = [entry_dict['cell']]
            if 'forces' in entry_dict:
                entry['Force']=[entry_dict['forces']]

            data_lst.append(entry)
            
        self.data = pd.concat(data_lst)

    def write(self, data:pd.DataFrame):
        '''Write configuration data to extended XYZ file for use in MACE. File will be saved under self.path.xyz'''
        with open(f'{self.path}.xyz', 'w') as f:
            for index, subset in data.iterrows():
                f.write(f"{subset['Config. size']}\n")

                # determine whether lattice parameters are contained within dataset
                if 'Lattice' in subset:
                    #lat=self.unpack(subset['Lattice'].to_list()[0])
                    lat=self.unpack(subset['Lattice'])
                    lat=f'Lattice=\"{lat}\"'
                    pbc=f'pbc=\"T T T\"'
                else:
                    lat=''
                    pbc=f'pbc=\"F F F\"'

                if 'Force' in data.columns:
                    f.write(f'energy={float(subset["Energy"])} {lat} Properties=species:S:1:pos:R:3:force:R:3 {pbc}\n')
                    for species, position, force in zip(subset['Atom'], subset['Position'], subset['Force']):
                        if not species in self.assign:
                            print(f'Type {species} not contained within self.assign.')
                        f.write(f'{self.type_to_symbol(species)}{position[0]:>12}{position[1]:>12}{position[2]:>12}{force[0]:>12}{force[1]:>12}{force[2]:>12}\n')
                else:
                    f.write(f'Energy={float(subset["Energy"])} {lat} Properties=species:S:1:pos:R:3 {pbc}\n')
                    for species, position in zip(subset['Atom'], subset['Position']):
                        if not species in self.assign:
                            print(f'Type {species} not contained within self.assign.')
                        f.write(f'{self.type_to_symbol(species)}{position[0]:>24}{position[1]:>24}{position[2]:>24}\n')

        print(f'XYZ file saved under {self.path}.xyz')