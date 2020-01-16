
printf '\n add_cpu.py (single precision) \n'
python add_cpu.py -float

printf '\n add_cpu.py (double precision) \n'
python add_cpu.py -double

printf '\n add_gpu.py (single precision) \n'
python add_gpu.py -float

printf '\n add_gpu.py (double precision) \n'
python add_gpu.py -double

printf '\n add_gpu_memcpy.cu (single precision) \n'
python add_gpu_memcpy.py -float

printf '\n add_gpu_memcpy.cu (double precision) \n'
python add_gpu_memcpy.py -double

printf '\n arithmetic_cpu.cu (single precision) \n'
python arithmetic_cpu.py -float

printf '\n arithmetic_cpu.cu (double precision) \n'
python arithmetic_cpu.py -double 

printf '\n arithmetic_gpu.cu (single precision) \n'
python arithmetic_gpu.py -float 1000000

printf '\n arithmetic_gpu.cu (double precision) \n'
python arithmetic_gpu.py -double 1000000

