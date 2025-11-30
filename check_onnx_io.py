import onnx

m = onnx.load("/home/nvida/btp_project/best_dynamic.onnx")

print("\n=== ONNX INPUTS ===")
for inp in m.graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dims.append(d.dim_value if d.dim_value > 0 else None)
    print("IN:", inp.name, dims)

print("\n=== ONNX OUTPUTS ===")
for out in m.graph.output:
    print("OUT:", out.name)

