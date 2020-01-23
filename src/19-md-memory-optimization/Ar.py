import time
import material, md

unitcell = {
    'lattice' : [
        [5.385, 0.000, 0.000],
        [0.000, 5.385, 0.000],
        [0.000, 0.000, 5.385],
        ],
    'fractional' : [
        ['Ar', 40, [0.0, 0.0, 0.0]],
        ['Ar', 40, [0.0, 0.5, 0.5]],
        ['Ar', 40, [0.5, 0.0, 0.5]],
        ['Ar', 40, [0.5, 0.5, 0.0]],
        ],
    }
Ar = material.Material(unitcell, [4,4,4])

input_infor = {
    'temperature' : 60,
    'cutoff' : 10.0,
    'time_step' : 5.0,
    'lj' : {
        'epsilon' : 1.032e-2,
        'sigma' : 3.405,
        }
    }
test = md.MolecularDynamics(Ar, input_infor)

start = time.time()
test.equilibration(10000)
test.production(10000, 100)