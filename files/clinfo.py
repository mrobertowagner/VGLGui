import pyopencl as cl
import os

if __name__ == "__main__":
  print("==========")
  print("Libraries used by pyopencl")
  print("")
  result = os.popen("ldd /usr/local/lib/python3.6/dist-packages/pyopencl/_cl.cpython-36m-x86_64-linux-gnu.so").read()
  print(result)

  result = os.popen("ldd /usr/lib/python3/dist-packages/pyopencl/_cffi.abi3.so").read()
  print(result)
  
  print("==========")
  print("Platforms and devices detected by pyopencl")
  print("")
  platforms = cl.get_platforms()
  i_p = 0
  for p in platforms:
    print("platform[%d]: %s" % (i_p, str(p)))
    i_p = i_p + 1
    devices = p.get_devices()
    i_d = 0
    for d in devices:
      print("  |--device[%d]: %s" % (i_d, str(d)))
      print("      |--type: %s" % cl.device_type.to_string(d.type))
      i_d = i_d + 1
