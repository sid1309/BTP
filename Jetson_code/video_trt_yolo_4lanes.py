#!/usr/bin/env python3
"""
video_trt_yolo_4lanes.py - countdown centered

This keeps all previous logic unchanged but centers the "Next change in: XXs"
countdown in the middle of the mosaic (both horizontally and vertically).
"""
import argparse
import time
import numpy as np
import cv2
import sys

# TRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except Exception as e:
    print("ERROR importing TensorRT/pycuda:", e)
    raise

# ---------- helpers ----------
def now_perf(): return time.perf_counter()

def box_iou(a, b):
    a = np.asarray(a, float).reshape(4,)
    b = np.asarray(b, float)
    if b.ndim == 1:
        b = b.reshape(1,4)
    ax1,ay1,ax2,ay2 = a
    bx1,by1,bx2,by2 = b[:,0],b[:,1],b[:,2],b[:,3]
    ix1 = np.maximum(ax1,bx1); iy1 = np.maximum(ay1,by1)
    ix2 = np.minimum(ax2,bx2); iy2 = np.minimum(ay2,by2)
    iw = np.maximum(0.0, ix2-ix1); ih = np.maximum(0.0, iy2-iy1)
    inter = iw*ih
    area_a = max(0.0,(ax2-ax1))*max(0.0,(ay2-ay1))
    area_b = (bx2-bx1)*(by2-by1)
    union = area_a + area_b - inter
    return inter/(union + 1e-6)

def nms(boxes, conf=0.18, iou=0.45):
    if boxes is None or len(boxes) == 0:
        return np.zeros((0,6))
    boxes = boxes[boxes[:,4] >= conf]
    if len(boxes) == 0:
        return np.zeros((0,6))
    order = np.argsort(-boxes[:,4])
    boxes = boxes[order]
    keep = []
    while len(boxes) > 0:
        best = boxes[0]
        keep.append(best)
        if len(boxes) == 1:
            break
        rest = boxes[1:]
        ious = box_iou(best[:4], rest[:,:4])
        same_class = rest[:,5] == best[5]
        suppress = (ious > iou) & same_class
        boxes = rest[~suppress]
    return np.stack(keep, 0)

# class weights for congestion scoring
CLS_W = {0:1.0, 1:1.0, 2:3.0, 3:0.5, 4:3.0}
def congestion_from_kept(kept):
    s = 0.0
    for b in kept:
        cls = int(b[5])
        s += CLS_W.get(cls, 1.0)
    return s

# ---------- TensorRT wrapper ----------
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TrtEngine:
    def __init__(self, engine_path):
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.inputs=[]; self.outputs=[]
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            shape = tuple(self.engine.get_binding_shape(idx))
            if self.engine.binding_is_input(idx):
                self.inputs.append({'idx':idx,'name':name,'dtype':dtype,'shape':shape})
            else:
                self.outputs.append({'idx':idx,'name':name,'dtype':dtype,'shape':shape})
        self.hbuf=[]; self.dbuf=[]
        self.bindings=[None]*self.engine.num_bindings
        for ent in (self.inputs + self.outputs):
            host = np.empty(ent['shape'], dtype=ent['dtype'])
            dev = cuda.mem_alloc(host.nbytes)
            self.hbuf.append(host)
            self.dbuf.append(dev)
            self.bindings[ent['idx']] = int(dev)

    def infer(self, batch):
        # copy input
        self.hbuf[0][:] = batch
        cuda.memcpy_htod(self.dbuf[0], self.hbuf[0].ravel())
        # execute
        self.ctx.execute_v2(self.bindings)
        # collect outputs
        outs=[]
        off = len(self.inputs)
        for i,outmeta in enumerate(self.outputs):
            host = self.hbuf[off + i]
            cuda.memcpy_dtoh(host, self.dbuf[off + i])
            outs.append(host.copy())
        return outs

# ---------- decode ----------
def decode_output(output0, conf_threshold=0.18):
    batch = output0.shape[0]
    results=[]
    for b in range(batch):
        preds = output0[b]
        # expecting shape (N,11) as before
        if preds.shape[1] != 11:
            results.append(np.zeros((0,6))); continue
        cx = preds[:,0]; cy = preds[:,1]; w = preds[:,2]; h = preds[:,3]
        conf = preds[:,4]
        cls_scores = preds[:,5:11]
        cls_id = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(len(cls_id)), cls_id]
        final_conf = conf * cls_conf
        mask = final_conf >= conf_threshold
        if mask.sum() == 0:
            results.append(np.zeros((0,6))); continue
        x1 = cx - 0.5*w; y1 = cy - 0.5*h
        x2 = cx + 0.5*w; y2 = cy + 0.5*h
        boxes = np.stack([x1,y1,x2,y2,final_conf,cls_id], axis=1)
        results.append(boxes[mask])
    return results

# ---------- main ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--engine', required=True)
    p.add_argument('--videos', nargs=4, required=True)
    p.add_argument('--detect_interval', type=float, default=3.0)
    p.add_argument('--signal_interval', type=float, default=15.0)
    p.add_argument('--conf', type=float, default=0.18)
    p.add_argument('--tile', type=int, default=420,    # increased default tile size
                   help='tile size (square) for each lane; default 420')
    args = p.parse_args()

    print(f"Detection every {args.detect_interval}s | Signal change interval: {args.signal_interval}s")
    start = time.perf_counter()
    trt_engine = TrtEngine(args.engine)
    in_shape = trt_engine.inputs[0]['shape'] if trt_engine.inputs else None
    print("Engine input shape:", in_shape)

    caps = [cv2.VideoCapture(v) for v in args.videos]
    for i,c in enumerate(caps):
        if not c.isOpened():
            print("Cannot open", args.videos[i]); return

    # state
    last_green = [None]*4   # None -> never green (treat as start for wait calc)
    current_green = None
    last_kept = [np.zeros((0,6)) for _ in range(4)]
    cong = [0.0]*4

    detect_interval = args.detect_interval
    signal_interval = args.signal_interval

    next_det = start    # run detection at t=0
    sig_idx = 0

    cv2.namedWindow("4lanes", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("4lanes", args.tile * 2 + 40, args.tile * 2 + 80)

    def run_detection(now, frames):
        imgs=[]
        for f in frames:
            im = cv2.resize(f, (640,640))
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
            imgs.append(im)
        batch = np.transpose(np.stack(imgs),(0,3,1,2)).astype(np.float32)
        outs = trt_engine.infer(batch)
        output0 = outs[-1]
        dec = decode_output(output0, conf_threshold=args.conf)
        for i in range(4):
            k = nms(dec[i], conf=args.conf)
            last_kept[i] = k
            cong[i] = congestion_from_kept(k)
        # schedule next_det using start-grid to avoid drift
        k = int(np.floor((now - start) / detect_interval)) + 1
        return start + k * detect_interval

    def format_state_header(t_sig):
        return f"\n===== SIGNAL (decision time) @ {int(round(t_sig-start))}s ====="

    def compute_waits_and_eff(t_sig):
        waits = []
        for i in range(4):
            ref = last_green[i] if last_green[i] is not None else start
            w = t_sig - ref
            if w < 0: w = 0.0
            waits.append(w)
        eff = [0.7*cong[i] + 0.3*waits[i] for i in range(4)]
        return waits, eff

    print("Initial detection + immediate first decision at 0s.")
    try:
        while True:
            now = time.perf_counter()
            # read frames
            frames = []
            for c in caps:
                ok, f = c.read()
                if not ok:
                    print("Video ended or read failed -> exiting.")
                    raise SystemExit
                frames.append(f)

            # run detection when scheduled
            if now >= next_det - 1e-6:
                next_det = run_detection(now, frames)
                print(f"[detect] ran at {now-start:.2f}s  cong={cong}")

            # check if it's time for a signal decision at exact multiples from start
            t_sig = start + sig_idx * signal_interval
            if now + 1e-6 >= t_sig:
                # ensure detection is fresh before decision
                next_det = run_detection(now, frames)

                # compute pre-update waits and effective (this is what decision is based on)
                waits_pre, eff_pre = compute_waits_and_eff(t_sig)

                # choose winner based on eff_pre only; tie-break by wait_pre
                eff_arr = np.array(eff_pre)
                max_eff = eff_arr.max()
                candidates = np.where(np.isclose(eff_arr, max_eff))[0]
                if len(candidates) == 1:
                    winner = int(candidates[0])
                else:
                    waits_arr = np.array(waits_pre)
                    winner = int(candidates[np.argmax(waits_arr[candidates])])

                # Print the pre-update table — but mark chosen lane as GREEN visually.
                print(format_state_header(t_sig))
                for i in range(4):
                    status = "GREEN" if i == winner else "RED"
                    print(f"Lane {i+1}: {status:5s} | cong={cong[i]:5.2f} | wait={waits_pre[i]:5.2f} | effective={eff_pre[i]:5.2f}")
                print("="*40)

                # update state: winner's last_green becomes exactly t_sig
                last_green[winner] = t_sig
                current_green = winner

                # Print concise post-update note
                print(f"*** DECISION: lane {winner+1} GIVEN GREEN (based on pre-update effective). ***")
                waits_post, eff_post = compute_waits_and_eff(t_sig)
                for i in range(4):
                    print(f"(post) Lane {i+1}: wait={waits_post[i]:5.2f} | effective={eff_post[i]:5.2f}")
                print("="*40)

                # advance to next signal tick
                sig_idx += 1

            # Visualization: draw last-kept boxes on tiles (thin rectangles)
            tiles=[]
            for i, f in enumerate(frames):
                d = cv2.resize(f, (args.tile, args.tile))
                k = last_kept[i]
                sx = args.tile / 640.0
                sy = args.tile / 640.0
                for b in k:
                    x1,y1,x2,y2 = b[:4]
                    # unbold rectangle (thin)
                    cv2.rectangle(d, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)), (0,255,0), 1)

                # prepare bold lane label: draw a black thicker outline then colored text
                is_green = (i == current_green)
                # effective display value (same formula as before)
                wait_vis = (time.perf_counter() - (last_green[i] if last_green[i] is not None else start)) if not is_green else 0.0
                eff_vis = 0.7*cong[i] + 0.3*wait_vis
                label_text = f"Lane {i+1}: {'GREEN' if is_green else 'RED'} ({eff_vis:.2f})"
                tx, ty = 6, 28
                # outline
                cv2.putText(d, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 4, cv2.LINE_AA)
                # colored text on top
                # *** change here: green lane uses same yellow as countdown (0,255,255) ***
                color = (0,255,255) if is_green else (0, 0, 255)
                cv2.putText(d, label_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

                # show small status bottom-left
                last_det_secs = int(max(0.0, now - (next_det - detect_interval))) if next_det is not None else 0
                cv2.putText(d, f"last_det:{last_det_secs}s", (6, args.tile - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                tiles.append(d)

            # mosaic (2x2)
            top = np.hstack((tiles[0], tiles[1]))
            bot = np.hstack((tiles[2], tiles[3]))
            fin = np.vstack((top, bot))

            # draw countdown to next signal centered in mosaic
            next_sig = start + sig_idx * signal_interval
            remaining = max(0.0, next_sig - time.perf_counter())
            rem_text = f"Next change in: {int(np.ceil(remaining)):02d}s"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.0
            thickness_outline = 4
            thickness_text = 2
            (text_w, text_h), baseline = cv2.getTextSize(rem_text, font, font_scale, thickness_text)
            h_fin, w_fin = fin.shape[:2]
            text_x = (w_fin - text_w) // 2
            text_y = (h_fin // 2) - 10

            # outline and yellow text
            cv2.putText(fin, rem_text, (text_x, text_y), font, font_scale,
                        (0,0,0), thickness_outline, cv2.LINE_AA)
            cv2.putText(fin, rem_text, (text_x, text_y), font, font_scale,
                        (0,255,255), thickness_text, cv2.LINE_AA)

            cv2.imshow("4lanes", fin)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except SystemExit:
        pass
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        for c in caps: c.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

