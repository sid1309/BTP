#!/usr/bin/env python3
# trt_engine_sanity_fixed.py
# Fixed TensorRT sanity check: handles trt.nptype and batch bindings.
# Usage: python3 trt_engine_sanity_fixed.py /path/to/yolov5s_idd.engine

import sys
import os
import time
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # initializes CUDA driver
except Exception as e:
    print("Failed to import tensorrt / pycuda:", e)
    sys.exit(2)

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        buf = f.read()
        engine = runtime.deserialize_cuda_engine(buf)
        return engine


def shape_bytes(shape, dtype):
    """Return number of bytes for a numpy dtype and shape tuple."""
    return int(np.prod(shape)) * np.dtype(dtype).itemsize


def main(argv):
    if len(argv) != 2:
        print("Usage: python3 trt_engine_sanity_fixed.py <engine_path>")
        return 1

    engine_path = argv[1]
    if not os.path.exists(engine_path):
        print("Engine file not found:", engine_path)
        return 2

    print("Loading engine:", engine_path)
    engine = load_engine(engine_path)
    if engine is None:
        print("Failed to deserialize engine.")
        return 3

    print("Engine loaded OK.")
    # print binding info
    for b in range(engine.num_bindings):
        name = engine.get_binding_name(b)
        is_input = engine.binding_is_input(b)
        dtype = engine.get_binding_dtype(b)
        shape = engine.get_binding_shape(b)
        # convert dtype to numpy dtype using trt.nptype
        try:
            np_dtype = trt.nptype(dtype)
        except Exception:
            np_dtype = np.float32
        print("Binding %d: name=%s  input=%s  dtype=%s  shape=%s  numpy_dtype=%s" %
              (b, name, str(is_input), str(dtype), str(shape), str(np_dtype)))

    # Create execution context
    ctx = engine.create_execution_context()
    if ctx is None:
        print("Failed to create execution context")
        return 4

    # Find input and output binding indices
    input_bindings = [i for i in range(engine.num_bindings) if engine.binding_is_input(i)]
    output_bindings = [i for i in range(engine.num_bindings) if not engine.binding_is_input(i)]

    if len(input_bindings) == 0:
        print("No input binding found.")
        return 5

    # Use first input binding
    ib = input_bindings[0]
    # Get static binding shape from engine
    binding_shape = tuple(engine.get_binding_shape(ib))
    print("Engine input binding shape (as stored in engine):", binding_shape)

    # If engine has a fixed batch dimension (e.g., first dim >1), respect it.
    # If it contains -1 or 0, try to set a reasonable shape.
    use_shape = list(binding_shape)
    dynamic_batch = False
    if any((d <= 0 for d in use_shape)):
        # choose reasonable default
        use_shape = [1, 3, 640, 640]
        print("Engine had dynamic/unknown dims; forcing input shape to", use_shape)
        try:
            ctx.set_binding_shape(ib, tuple(use_shape))
        except Exception as e:
            print("Could not set binding shape:", e)
    else:
        # For fixed-shape engine, the context binding shape is same as engine's
        try:
            ctx.set_binding_shape(ib, tuple(use_shape))
        except Exception:
            pass

    # Convert dtype
    try:
        np_dtype_in = trt.nptype(engine.get_binding_dtype(ib))
    except Exception:
        np_dtype_in = np.float32

    print("Using input dtype:", np_dtype_in)

    # Prepare device/host buffers and bindings list
    bindings = [None] * engine.num_bindings
    stream = cuda.Stream()

    # Prepare input host buffer: random values
    h_input = np.random.random(size=tuple(use_shape)).astype(np_dtype_in)
    d_input = cuda.mem_alloc(h_input.nbytes)
    bindings[ib] = int(d_input)
    cuda.memcpy_htod_async(d_input, h_input, stream)

    # Prepare outputs based on context binding shapes (which reflect set shapes)
    output_arrays = {}
    for ob_idx in output_bindings:
        try:
            out_shape = tuple(ctx.get_binding_shape(ob_idx))
        except Exception:
            out_shape = tuple(engine.get_binding_shape(ob_idx))
        # convert dtype
        try:
            np_dtype_out = trt.nptype(engine.get_binding_dtype(ob_idx))
        except Exception:
            np_dtype_out = np.float32
        host_out = np.empty(out_shape, dtype=np_dtype_out)
        device_out = cuda.mem_alloc(host_out.nbytes)
        bindings[ob_idx] = int(device_out)
        output_arrays[ob_idx] = (host_out, device_out)
        print("Alloc output binding", ob_idx, "shape", out_shape, "dtype", np_dtype_out, "bytes", host_out.nbytes)

    # Execute (async) and measure
    t0 = time.time()
    try:
        ctx.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        stream.synchronize()
    except Exception as e:
        print("Execution failed:", repr(e))
        return 6
    t1 = time.time()
    elapsed = t1 - t0
    print("Inference took %.4f s  (~%.2f FPS)" % (elapsed, (1.0 / elapsed) if elapsed > 0 else 0.0))

    # Copy outputs back and report stats
    for ob_idx, (host_out, dev_out) in output_arrays.items():
        cuda.memcpy_dtoh_async(host_out, dev_out, stream)
    stream.synchronize()

    for ob_idx, (host_out, _) in output_arrays.items():
        flat = host_out.ravel()
        # safe stats
        try:
            mn = float(np.nanmin(flat))
            mx = float(np.nanmax(flat))
            mean = float(np.nanmean(flat))
            nonzero = int((flat != 0).sum())
        except Exception:
            mn, mx, mean, nonzero = None, None, None, None
        print("Output binding", ob_idx, "-> shape:", host_out.shape, " dtype:", host_out.dtype)
        print(" stats: min %s  max %s  mean %s  nonzero_count %s / %d" %
              (str(mn), str(mx), str(mean), str(nonzero), flat.size))

    print("Sanity test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

