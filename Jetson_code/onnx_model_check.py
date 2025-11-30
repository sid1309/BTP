# onnx_model_check.py
import onnx
import sys
import numpy as np

if len(sys.argv) != 2:
    print("Usage: python3 onnx_model_check.py /path/to/model.onnx")
    sys.exit(1)

path = sys.argv[1]
m = onnx.load(path)
print("Loaded ONNX:", path)
print("Opset version:", m.opset_import[0].version if m.opset_import else "n/a")
print("Producer:", m.producer_name, m.producer_version)

# count parameters (sum of all initializer element counts)
param_count = 0
for init in m.graph.initializer:
    dims = [d for d in init.dims]
    param_count += int(np.prod(dims))
print(f"Estimated param count (weights only): {param_count:,}")

# Heuristic to guess YOLO variant (common sizes)
if param_count < 12_000_000:
    guess = "yolov5s (likely)"
elif param_count < 30_000_000:
    guess = "yolov5m (likely)"
elif param_count < 60_000_000:
    guess = "yolov5l (likely)"
else:
    guess = "yolov5x (likely)"
print("Variant guess:", guess)

# print model inputs/outputs and suspicious reshape/view nodes
print("\n=== INPUTS ===")
for inp in m.graph.input:
    name = inp.name
    dims = []
    if inp.type and inp.type.tensor_type and inp.type.tensor_type.shape:
        for d in inp.type.tensor_type.shape.dim:
            dims.append(getattr(d,"dim_value", None))
    print(" IN:", name, dims)

print("\n=== OUTPUTS ===")
for out in m.graph.output:
    dims = []
    if out.type and out.type.tensor_type and out.type.tensor_type.shape:
        for d in out.type.tensor_type.shape.dim:
            dims.append(getattr(d,"dim_value", None))
    print(" OUT:", out.name, dims)

# list node ops that might cause trouble
bad_ops = []
for i, node in enumerate(m.graph.node):
    if node.op_type in ("Reshape", "View", "aten::view", "Slice", "Unsqueeze") or "view" in node.name.lower():
        bad_ops.append((i, node.op_type, node.name, list(node.input), list(node.output)))
if bad_ops:
    print("\n=== POSSIBLE PROBLEM NODES ===")
    for i, op, name, ins, outs in bad_ops[:50]:
        print(f" #{i} {op} name={name} inputs={ins} outputs={outs}")
else:
    print("\nNo suspicious reshape/view nodes found.")

print("\nDone.")

