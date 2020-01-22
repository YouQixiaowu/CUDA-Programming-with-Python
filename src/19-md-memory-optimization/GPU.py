import math, time, sys, os
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import DynamicSourceModule


class kernel(object):
    """
    kernel doc
    """
    def __init__(self, number, MN, kernel=None, double=False):

        if not isinstance(number, int):
            raise ValueError(self.__doc__)

        if not isinstance(MN, int):
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

        self.MN = np.int32(MN)
        self.atomic_number = np.int32(number)
        self.block = (128,1,1)
        self.grid = ((number-1)//128+1, 1)
        size_real = np.array(0, dtype=self.real_py).itemsize
        size_int = np.array(0, dtype=np.int32).itemsize

        kernel_program = DynamicSourceModule(kernel_code.replace('real', self.real_cpp))
        self._find_neighbor  = kernel_program.get_function('gpu_find_neighbor')
        self._find_force     = kernel_program.get_function('gpu_find_force')
        self._integrate      = kernel_program.get_function('gpu_integrate')
        self._g_sum          = kernel_program.get_function('gpu_sum')
        self._scale_velocity = kernel_program.get_function('gpu_scale_velocity')

        self.g_sum           = drv.mem_alloc(size_real)
        self.lj              = drv.mem_alloc(size_real*5)
        self.box             = drv.mem_alloc(size_real*6)
        self.neighbor_number = drv.mem_alloc(size_int*number)
        self.neighbor_index  = drv.mem_alloc(size_int*number*MN)
        self.atomic_mass     = drv.mem_alloc(size_real*number)
        self.coordinate_x    = drv.mem_alloc(size_real*number)
        self.coordinate_y    = drv.mem_alloc(size_real*number)
        self.coordinate_z    = drv.mem_alloc(size_real*number)
        self.velocity_x      = drv.mem_alloc(size_real*number)
        self.velocity_y      = drv.mem_alloc(size_real*number)
        self.velocity_z      = drv.mem_alloc(size_real*number)
        self.force_x         = drv.mem_alloc(size_real*number)
        self.force_y         = drv.mem_alloc(size_real*number)
        self.force_z         = drv.mem_alloc(size_real*number)
        self.pe              = drv.mem_alloc(size_real*number)
        self.ke              = drv.mem_alloc(size_real*number)
    
    def upload(self, atomic_mass, coordinate, velocity, lj, box):
        drv.memcpy_htod(self.atomic_mass,  np.array(atomic_mass, dtype=self.real_py))
        drv.memcpy_htod(self.coordinate_x, np.array(coordinate[0], dtype=self.real_py))
        drv.memcpy_htod(self.coordinate_y, np.array(coordinate[1], dtype=self.real_py))
        drv.memcpy_htod(self.coordinate_z, np.array(coordinate[2], dtype=self.real_py))
        drv.memcpy_htod(self.velocity_x,   np.array(velocity[0], dtype=self.real_py))
        drv.memcpy_htod(self.velocity_y,   np.array(velocity[1], dtype=self.real_py))
        drv.memcpy_htod(self.velocity_z,   np.array(velocity[2], dtype=self.real_py))
        drv.memcpy_htod(self.lj,           np.array(lj, dtype=self.real_py))
        drv.memcpy_htod(self.box,          np.array(box, dtype=self.real_py))

    def _sum(self, GPU_Array):
        M = (self.atomic_number - 1) // 25600 + 1
        self.h_sum = np.array(0.0, dtype=self.real_py)
        drv.memcpy_htod(self.g_sum, self.h_sum)
        self._g_sum(
            self.atomic_number,
            np.int32(M),
            GPU_Array,
            self.g_sum,
            grid=(int((self.atomic_number-1)//(128*M)+1), 1, 1), 
            block=(128,1,1),
            )
        drv.memcpy_dtoh(self.h_sum, self.g_sum)
        
        return self.h_sum

    def sum_ke(self):
        return self._sum(self.ke)

    def sum_pe(self):
        return self._sum(self.pe)

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
        temperature = self._sum(self.ke) / (1.5 * kB * self.atomic_number)
        self._scale_velocity(
            self.atomic_number,
            np.__dict__[self.real_py](math.sqrt(target_temperature/temperature)),
            self.velocity_x,
            self.velocity_y,
            self.velocity_z,
            grid=self.grid, 
            block=self.block,
            )
        
