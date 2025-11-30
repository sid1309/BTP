import onnx

onnx_path = "/home/nvida/btp_project/best_static.onnx"

# Load ONNX file
m = onnx.load(onnx_path)
opset = m.opset_import[0].version if m.opset_import else "unknown"

print("ONNX path:", onnx_path)
print("Opset version:", opset)

# ------------------------------
# INPUTS
# ------------------------------
print("\n=== INPUTS ===")
for inp in m.graph.input:
    dims = []
    for d in inp.type.tensor_type.shape.dim:
        dv = getattr(d, "dim_value", None)
        if dv == 0:
            dv = None
        dims.append(dv)
    print(" IN:", inp.name, dims)

# ------------------------------
# OUTPUTS
# ------------------------------
print("\n=== OUTPUTS ===")
for out in m.graph.output:
    dims = []
    for d in out.type.tensor_type.shape.dim:
        dv = getattr(d, "dim_value", None)
        if dv == 0:
            dv = None
        dims.append(dv)
    print(" OUT:", out.name, dims)

# ------------------------------
# Problematic Nodes (Reshape / View)
# ------------------------------
print("\n=== POSSIBLE BAD NODES (Reshape / View / aten.view) ===")
for i, node in enumerate(m.graph.node):
    if node.op_type.lower() in ("reshape", "view") or "view" in node.op_type.lower():
        print(f" #{i:03d}  {node.op_type}  name={node.name}")
        print("     inputs =", list(node.input))
        print("     outputs =", list(node.output))
        for attr in node.attribute:
            print("       attr:", attr.name)

print("\nFinished ONNX check.")

