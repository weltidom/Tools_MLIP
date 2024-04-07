from ase import units
from ase.md.langevin import Langevin
from ase.io import read, write
from ase.md import MDLogger


from uf3.forcefield import calculator
from uf3.regression import least_squares

# Load model
model = least_squares.WeightedLinearModel.from_json('/home/st/st_us-031400/st_st179390/uf3/hea/results/model_0.json')


# Initialize calculator
calc = calculator.UFCalculator(model)
init_conf = read('/home/st/st_us-031400/st_st179390/mace/ta-v-cr-w/inference_ase/4comp_0k_432atoms.xyz', '0')
init_conf.set_calculator(calc)

# Run MD
steps=10000
dyn = Langevin(init_conf, 0.5*units.fs, temperature_K=310, friction=5e-3)
# def write_frame():
#         dyn.atoms.write('mdTimeResults.xyz', append=True)
# dyn.attach(write_frame, interval=50)

dyn.attach(MDLogger(dyn, init_conf, 'md.log', header=False, stress=False,
           peratom=True, mode="a"), interval=50)
dyn.run(steps)

print(f"MD with {steps} steps finished")