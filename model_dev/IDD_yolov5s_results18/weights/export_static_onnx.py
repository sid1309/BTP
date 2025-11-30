import torch

model_path = "/home/admin/Documents/BTP/yolov5/runs/train/IDD_yolov5s_results18/weights/best.pt"
out_path = "/home/admin/Documents/BTP/yolov5/runs/train/IDD_yolov5s_results18/weights/best_static.onnx"

print("Loading model WITHOUT AutoShape...")
m = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, autoshape=False)

# Get the core model (DetectMultiBackend → model.model)
core = m.model
core.eval().to('cuda')

dummy = torch.randn(4, 3, 640, 640).cuda()  # static BS=4

torch.onnx.export(
    core,
    dummy,
    out_path,
    opset_version=18,
    input_names=['images'],
    output_names=['output0'],
    dynamic_axes=None,        # IMPORTANT: STATIC ONNX
    do_constant_folding=True
)

print("Saved:", out_path)
