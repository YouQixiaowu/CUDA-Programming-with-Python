import math, time
import numpy as np
import GPU, material

class MolecularDynamics(object):
    """


    """
    def __init__(self, material, input_infor:dict):

        # 从材料类中拷贝相应数据
        atomic_number = material.atomic_number
        kB = 8.617343e-5
        TIME_UNIT_CONVERSION = 1.018051e+1
        MN = 256
        cutoff = input_infor['cutoff']
        self.initial_temperature = input_infor['temperature']
        self.time_step = input_infor['time_step']/TIME_UNIT_CONVERSION
        
        epsilon = input_infor['lj']['epsilon']
        sigma = input_infor['lj']['sigma']
        e24s6 = 24.0 * epsilon * sigma**6
        e48s12 = 48.0 * epsilon * sigma**12
        e4s6 = 4.0 * epsilon * sigma**6
        e4s12 = 4.0 * epsilon * sigma**12

        velocity = np.random.uniform(-1.0, 1.0, size=(3, atomic_number))
        velocity = velocity - np.tile((velocity).sum(axis=1)/atomic_number, (atomic_number, 1)).T
        temperature = (material.atomic_mass*velocity*velocity).sum() / (3 * kB * atomic_number)
        velocity = velocity * math.sqrt(self.initial_temperature / temperature)
        
        box = [material.lattice[0,0], material.lattice[1,1], material.lattice[2,2], 
            material.lattice[0,0]*0.5, material.lattice[1,1]*0.5, material.lattice[2,2]*0.5]
        lj = [cutoff*cutoff, e24s6, e48s12, e4s6, e4s12]

        self.gpu = GPU.kernel(atomic_number, MN, double=False)
        self.gpu.upload(
            material.atomic_mass, 
            material.coordinate, 
            velocity, 
            lj, 
            box)

        self.gpu.find_neighbor(cutoff)

    def equilibration(self, Ne):
        self.gpu.find_force()
        start = time.time()
        for step in range(Ne):
            self.gpu.integrate(self.time_step, 1)
            self.gpu.find_force()
            self.gpu.integrate(self.time_step, 2)
            self.gpu.scale_velocity(self.initial_temperature)
        end = time.time()
        print("time used for equilibration = ", end - start," s")

    def production(self, Np, Ns=None):
        if Ns==None:
            Ns = Np
        start = time.time()
        with open('energy.txt', 'w') as f:
            for step in range(Np):
                self.gpu.integrate(self.time_step, 1)
                self.gpu.find_force()
                self.gpu.integrate(self.time_step, 2)
                if step % Ns == 0:
                    f.write('{}\t{}\n'.format(self.gpu.sum_ke(), self.gpu.sum_pe()))
        end = time.time()
        print("time used for production = ", end - start," s")

