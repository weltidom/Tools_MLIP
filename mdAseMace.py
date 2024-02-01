from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
import numpy as np
import time

from mace.calculators import MACECalculator

calculator = MACECalculator(model_path='/home/st/st_us-031400/st_st179390/mace/ta-v-cr-w/inference/mace_split_0.model', device='cpu')
init_conf = read('', '0')
init_conf.set_calculator(calculator)

dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
def write_frame():
        dyn.atoms.write('md_3bpa.xyz', append=True)
dyn.attach(write_frame, interval=50)
dyn.run(100)
print("MD finished!")