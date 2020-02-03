import math, time, sys, os
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
from pycuda.compiler import DynamicSourceModule

class Kernel(object):
    """
    atomic_number : Atomic number.
    neighbor_number : Maximum number of adjacent lists (Default: 200).
    kernel : Kernel function file path (Default: 'This file path'/kernel.cu).
    double : Whether to use double precision floating-point numbers (Default: False).
    """
    def __init__(self, atomic_number, neighbor_number=200, kernel=None, double=False):

        if not isinstance(atomic_number, int):
            raise ValueError(self.__doc__)

        if not isinstance(neighbor_number, int):
            raise ValueError(self.__doc__)

        if double:
            self.real_py = 'float64' 
            self.real_cpp = 'double'
        else:
            self.real_py = 'float32'
            self.real_cpp = 'float'

        if kernel == None:
            kernel = os.path.join(sys.path[0], 'kernel.cu')
        if os.path.exists(kernel):
            with open(os.path.join(sys.path[0], kernel), 'r') as f:
                kernel_code = f.read()
        else:
            raise ValueError(self.__doc__)

        self.MN = np.int32(neighbor_number)
        self.atomic_number = np.int32(atomic_number)
        self.block = (128,1,1)
        self.grid = ((atomic_number-1)//128+1, 1)

        kernel_program = DynamicSourceModule(kernel_code.replace('real', self.real_cpp))
        self._find_neighbor  = kernel_program.get_function('gpu_find_neighbor')
        self._find_force     = kernel_program.get_function('gpu_find_force')
        self._integrate      = kernel_program.get_function('gpu_integrate')
        self._scale_velocity = kernel_program.get_function('gpu_scale_velocity')

        self.neighbor_number = gpuarray.empty(atomic_number, dtype=np.int32)
        self.neighbor_index  = gpuarray.empty(atomic_number*neighbor_number, dtype=np.int32)
        self.force_x         = gpuarray.empty(atomic_number, dtype=self.real_py)
        self.force_y         = gpuarray.empty(atomic_number, dtype=self.real_py)
        self.force_z         = gpuarray.empty(atomic_number, dtype=self.real_py)
        self.pe              = gpuarray.empty(atomic_number, dtype=self.real_py)
        self.ke              = gpuarray.empty(atomic_number, dtype=self.real_py)
    
    def upload(self, atomic_mass, coordinate, velocity, lj, box):
        self.lj           = gpuarray.to_gpu(np.array(lj, dtype=self.real_py))
        self.box          = gpuarray.to_gpu(np.array(box, dtype=self.real_py))
        self.atomic_mass  = gpuarray.to_gpu(np.array(atomic_mass, dtype=self.real_py))
        self.coordinate_x = gpuarray.to_gpu(np.array(coordinate[0], dtype=self.real_py))
        self.coordinate_y = gpuarray.to_gpu(np.array(coordinate[1], dtype=self.real_py))
        self.coordinate_z = gpuarray.to_gpu(np.array(coordinate[2], dtype=self.real_py))
        self.velocity_x   = gpuarray.to_gpu(np.array(velocity[0], dtype=self.real_py))
        self.velocity_y   = gpuarray.to_gpu(np.array(velocity[1], dtype=self.real_py))
        self.velocity_z   = gpuarray.to_gpu(np.array(velocity[2], dtype=self.real_py))


    def sum_ke(self):
        return gpuarray.sum(self.ke)

    def sum_pe(self):
        return gpuarray.sum(self.pe)

    def integrate(self, time_step, flag):
        self._integrate(
            self.atomic_number, 
            np.__dict__[self.real_py](time_step),
            np.__dict__[self.real_py](time_step*0.5),
            self.atomic_mass,
            self.coordinate_x,
            self.coordinate_y,
            self.coordinate_z,
            self.velocity_x,
            self.velocity_y,
            self.velocity_z,
            self.force_x,
            self.force_y,
            self.force_z,
            self.ke,
            np.int32(flag),
            grid=self.grid, 
            block=self.block,
            )
        
    def find_neighbor(self, cutoff):
        self._find_neighbor(
            self.atomic_number, 
            self.MN,
            self.neighbor_number,
            self.neighbor_index,
            self.box,
            self.coordinate_x,
            self.coordinate_y,
            self.coordinate_z,
            np.__dict__[self.real_py](cutoff*cutoff),
            grid=self.grid, 
            block=self.block,
            )
        
    def find_force(self):
        self._find_force(
            self.lj, 
            self.atomic_number,  
            self.neighbor_number, 
            self.neighbor_index, 
            self.box,
            self.coordinate_x,
            self.coordinate_y,
            self.coordinate_z,
            self.force_x,
            self.force_y,
            self.force_z,
            self.pe,
            grid=self.grid, 
            block=self.block,
            )
        
    def scale_velocity(self, target_temperature):
        kB = 8.617343e-5
        temperature = gpuarray.sum(self.ke).get() / (1.5 * kB * self.atomic_number)
        tk = np.__dict__[self.real_py](math.sqrt(target_temperature/temperature))
        self.velocity_x = self.velocity_x*tk
        self.velocity_y = self.velocity_y*tk
        self.velocity_z = self.velocity_z*tk


