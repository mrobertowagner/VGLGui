import pyopencl as cl

# Cria uma plataforma OpenCL
platforms = cl.get_platforms()
gpu_devices = platforms[0].get_devices(device_type=cl.device_type.GPU)

# Cria o contexto OpenCL usando o primeiro dispositivo GPU
context = cl.Context(devices=gpu_devices)
