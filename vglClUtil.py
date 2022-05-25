
"""
    ************************************************************************
    ***                                                                  ***
    ***                     Wrapper code for CL_UTIL                     ***
    ***                                                                  ***
    *** Author: ddantas                                                  ***
    *** 28/10/2021                                                       ***
    ***                                                                  ***
    ************************************************************************
"""
#!/usr/bin/python3 python3

# OPENCL LIBRARY
import pyopencl as cl

# VGL LIBRARYS
import vgl_lib as vl

#TO WORK WITH MAIN
import numpy as np


"""
    /** Test if images are equal.
    
  */    
"""
def vglClEqual1(img_input, img_output):

    vl.vglCheckContext(img_input, vl.VGL_CL_CONTEXT())
    vl.vglCheckContext(img_output, vl.VGL_CL_CONTEXT())

    _program = vl.get_ocl_context().get_compiled_kernel("CL_UTIL/vglClEqual.cl", "vglClEqual")
    _kernel = _program.vglClEqual

    _kernel.set_arg(0, img_input.get_oclPtr())
    _kernel.set_arg(1, img_output.get_oclPtr())

    # THIS IS A BLOCKING COMMAND. IT EXECUTES THE KERNEL.
    cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, _kernel, img_input.get_oclPtr().shape, None)

    vl.vglSetContext(img_output, vl.VGL_CL_CONTEXT())

    return 1


def vglClEqual(img_input1, img_input2):

    vl.vglCheckContext(img_input1, vl.VGL_CL_CONTEXT())
    vl.vglCheckContext(img_input2, vl.VGL_CL_CONTEXT())

    _program = vl.get_ocl_context().get_compiled_kernel("CL_UTIL/vglClEqual.cl", "vglClEqual")
    _kernel = _program.vglClEqual

    #ocl = vl.vglClImage.get_ocl()
    #mobj_equal = np.uint32(7)
    mobj_equal = np.array((1,), dtype=np.uint32)
    mobj_equal[0] = 1
    mf = cl.mem_flags
    mobj_ptr = cl.Buffer(vl.get_ocl().context, mf.READ_WRITE, 4)
    cl.enqueue_copy(vl.get_ocl().commandQueue, mobj_ptr, mobj_equal, is_blocking=True)


    _kernel.set_arg(0, img_input1.get_oclPtr())
    _kernel.set_arg(1, img_input2.get_oclPtr())
    _kernel.set_arg(2, mobj_ptr)

    # THIS IS A BLOCKING COMMAND. IT EXECUTES THE KERNEL.
    cl.enqueue_nd_range_kernel(vl.get_ocl().commandQueue, _kernel, img_input1.get_oclPtr().shape, None)

    import time
    #time.sleep(1)

    print("mobj_equal  = %d" % mobj_equal)
    cl.enqueue_copy(vl.get_ocl().commandQueue, mobj_equal, mobj_ptr, is_blocking=True)
    print("mobj_equal  = %d" % mobj_equal)
    return mobj_equal
        

