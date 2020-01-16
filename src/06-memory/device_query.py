import pycuda.driver as cuda
import pycuda.autoinit

print("%d device(s) found." % cuda.Device.count() )  #打印发现的设备数量

for device_id in range(cuda.Device.count()):                
    dev=cuda.Device(device_id)        #索引第 device_id 个GPU   
    prop = dev.get_attributes()

    print("Device id:                                 %d" %
        device_id)
    print("Device name:                               %s" %
        dev.name())
    print("Compute capability:                        %d.%d" %
        dev.compute_capability())
    print("Amount of global memory:                   %g GB" %
        (dev.total_memory() / (1024 * 1024 * 1024)))
    print("Amount of constant memory:                 %g KB" %
        (prop[cuda.device_attribute.TOTAL_CONSTANT_MEMORY]  / 1024.0))
    print("Maximum grid size:                         %d %d %d" %
        (prop[cuda.device_attribute.MAX_GRID_DIM_X], 
        prop[cuda.device_attribute.MAX_GRID_DIM_Y], 
        prop[cuda.device_attribute.MAX_GRID_DIM_Z]))
    print("Maximum block size:                        %d %d %d" %
        (prop[cuda.device_attribute.MAX_BLOCK_DIM_X], 
        prop[cuda.device_attribute.MAX_BLOCK_DIM_Y], 
        prop[cuda.device_attribute.MAX_BLOCK_DIM_Z]))
    print("Number of SMs:                             %d" %
        prop[cuda.device_attribute.MULTIPROCESSOR_COUNT])
    print("Maximum amount of shared memory per block: %g KB" %
        (prop[cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK] / 1024.0))
    print("Maximum amount of shared memory per SM:    %g KB" %
        (prop[cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR] / 1024.0))
    print("Maximum number of registers per block:     %d K" %
        (prop[cuda.device_attribute.MAX_REGISTERS_PER_BLOCK] / 1024))
    print("Maximum number of registers SM:            %d K" %
        (prop[cuda.device_attribute.MAX_REGISTERS_PER_MULTIPROCESSOR] / 1024))
    print("Maximum number of threads per block:       %d" %
        prop[cuda.device_attribute.MAX_THREADS_PER_BLOCK])
    print("Maximum number of threads per SM:          %d" %
        prop[cuda.device_attribute.MAX_THREADS_PER_MULTIPROCESSOR])
