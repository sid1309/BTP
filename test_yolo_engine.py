# test_yolo_engine.py
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import time

ENGINE_PATH = "/home/nvida/btp_project/yolov5s_idd.engine"

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
with open(ENGINE_PATH, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())

context = engine.create_execution_context()
input_idx = engine.get_binding_index("images")
output_idx = engine.get_binding_index("output0")

# Create dummy input (640x640 RGB)
inp = np.random.rand(1, 3, 640, 640).astype(np.float16)
d_input = cuda.mem_alloc(inp.nbytes)
output_shape = (1, 25200, 11)
out = np.empty(output_shape, dtype=np.float16)
d_output = cuda.mem_alloc(out.nbytes)

# Copy + infer
cuda.memcpy_htod(d_input, inp)
start = time.time()
context.execute_v2([int(d_input), int(d_output)])
cuda.memcpy_dtoh(out, d_output)
print("Inference done in", time.time() - start, "s")
print("Output sample:", out.flatten()[:20])

