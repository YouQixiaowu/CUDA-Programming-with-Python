import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

mod = SourceModule(
    '#define NUM {0}\n'.format(5)+r"""
__global__ void function(void)
{
    int NUMBER = 10;
    printf("num=%d\tnumber=%d\n", NUM, NUMBER);  
}
""")
function = mod.get_function("function")
function(block=(1,1,1))
