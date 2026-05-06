"""
Live Face Swap inference service for GenFlow.

Run this on the same GPU pod that hosts ComfyUI, or any machine with a CUDA GPU.

Setup:
    pip install fastapi uvicorn[standard] python-multipart insightface onnxruntime-gpu opencv-python-headless numpy pillow

Run:
    python live_face_swap_service.py --port 8189

Then in GenFlow Settings, set "Live Face Swap URL" to:
    https://<your-runpod-host>-8189.proxy.runpod.net

This serves two endpoints:
    POST /set_source       — multipart "image" → caches the source face embedding
    POST /swap             — multipart "image" → returns JPEG bytes with face swapped
    GET  /healthz
"""
import argparse
import io
import os
import time
import urllib.request
from typing import Optional

import cv2
import insightface
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from insightface.app import FaceAnalysis

app = FastAPI(title="GenFlow Live Face Swap")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- model loading (once at startup) -----------------------------------------
print("[live-swap] loading InsightFace buffalo_l detector...")
analyzer = FaceAnalysis(name="buffalo_l")
# 320x320 detector cuts detection time roughly in half versus 640x640.
# Plenty of resolution for webcam-sized frames.
analyzer.prepare(ctx_id=0, det_size=(320, 320))

# inswapper_128 — the library's auto-downloader looks for a .zip that no longer
# exists. Fetch the raw .onnx directly into the expected cache location.
_INSWAP_DIR = os.path.expanduser("~/.insightface/models")
_INSWAP_PATH = os.path.join(_INSWAP_DIR, "inswapper_128.onnx")
_INSWAP_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/inswapper_128.onnx"
os.makedirs(_INSWAP_DIR, exist_ok=True)
if not os.path.exists(_INSWAP_PATH) or os.path.getsize(_INSWAP_PATH) < 100_000_000:
    print(f"[live-swap] downloading inswapper_128.onnx (~554MB) → {_INSWAP_PATH}")
    urllib.request.urlretrieve(_INSWAP_URL, _INSWAP_PATH)
    print("[live-swap] download complete")

print("[live-swap] loading inswapper_128.onnx ...")
swapper = insightface.model_zoo.get_model(_INSWAP_PATH)

source_face = None  # cached after /set_source


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(500, "JPEG encode failed")
    return buf.tobytes()


@app.get("/healthz")
def healthz():
    return {"ok": True, "source_loaded": source_face is not None}


@app.post("/set_source")
async def set_source(image: UploadFile = File(...)):
    global source_face
    img = _decode(await image.read())
    faces = analyzer.get(img)
    if not faces:
        raise HTTPException(400, "No face detected in source image")
    # use the largest face
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
    source_face = faces[0]
    return {"ok": True, "bbox": [int(v) for v in source_face.bbox.tolist()]}


def _expand_swap_chin(orig: np.ndarray, swapped: np.ndarray, face) -> np.ndarray:
    """Fast beard-cover: paint chin ellipse with cheek-skin color, tiny feather.

    Operates only on a small crop around the face for speed. Flat skin fill
    (no shading preservation) keeps things crisp rather than blurred.
    """
    try:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        kps = face.kps
        l_eye, r_eye, nose, _l_mouth, _r_mouth = kps

        H, W = swapped.shape[:2]

        # Work on a crop around the face bbox (2x faster than full frame)
        pad_x = int((x2 - x1) * 0.2)
        pad_y = int((y2 - y1) * 0.3)
        cx0 = max(0, x1 - pad_x); cy0 = max(0, y1 - pad_y)
        cx1 = min(W, x2 + pad_x); cy1 = min(H, y2 + pad_y)
        if cx1 - cx0 < 20 or cy1 - cy0 < 20:
            return swapped
        crop = swapped[cy0:cy1, cx0:cx1].copy()
        ch, cw = crop.shape[:2]

        # Coordinates relative to crop
        def rel(pt):
            return int(pt[0] - cx0), int(pt[1] - cy0)
        l_eye_c = rel(l_eye); r_eye_c = rel(r_eye); nose_c = rel(nose)

        # 1. Sample clean skin from both cheeks (one patch per side, 6x6)
        def sample(pt, half=5):
            px, py = pt
            x0 = max(0, px - half); xx = min(cw, px + half)
            y0 = max(0, py - half); yy = min(ch, py + half)
            p = crop[y0:yy, x0:xx]
            if p.size == 0: return None
            return np.median(p.reshape(-1, 3), axis=0)
        samples = [s for s in (
            sample(((l_eye_c[0] + nose_c[0]) // 2, (l_eye_c[1] + nose_c[1]) // 2 + 4)),
            sample(((r_eye_c[0] + nose_c[0]) // 2, (r_eye_c[1] + nose_c[1]) // 2 + 4)),
        ) if s is not None]
        if not samples:
            return swapped
        skin = np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)

        # 2. Chin ellipse mask (just below the nose down past the chin)
        e_cx = cw // 2
        e_cy = int((nose_c[1] + ch) * 0.55)
        e_w = int(cw * 0.32)
        e_h = int((ch - nose_c[1]) * 0.55)
        if e_w < 5 or e_h < 5:
            return swapped
        mask = np.zeros((ch, cw), dtype=np.uint8)
        cv2.ellipse(mask, (e_cx, e_cy), (e_w, e_h), 0, 0, 360, 255, -1)

        # Tight feather (15x15 is much faster than 41x41 and keeps edges crisp)
        mask_f = cv2.GaussianBlur(mask, (15, 15), 0).astype(np.float32) / 255.0
        mask_f = mask_f[:, :, None]

        # 3. Flat skin fill, no shading trick → no blur look
        fill = np.full_like(crop, skin)
        blended = (fill.astype(np.float32) * mask_f + crop.astype(np.float32) * (1 - mask_f)).astype(np.uint8)

        out = swapped.copy()
        out[cy0:cy1, cx0:cx1] = blended
        return out
    except Exception as e:
        print(f"[live-swap] chin-expand failed: {e}")
        return swapped


@app.post("/swap")
async def swap(image: UploadFile = File(...), cover_beard: int = 0):
    if source_face is None:
        raise HTTPException(400, "Source face not set. POST /set_source first.")
    t0 = time.time()
    img = _decode(await image.read())

    faces = analyzer.get(img)
    if not faces:
        # No face in this frame — return original so the stream stays smooth.
        return Response(content=_encode_jpeg(img), media_type="image/jpeg",
                        headers={"X-Latency-Ms": f"{int((time.time() - t0) * 1000)}", "X-Faces": "0"})

    res = img.copy()
    for face in faces:
        swapped = swapper.get(res, face, source_face, paste_back=True)
        if cover_beard:
            res = _expand_swap_chin(res, swapped, face)
        else:
            res = swapped

    out = _encode_jpeg(res)
    return Response(content=out, media_type="image/jpeg",
                    headers={"X-Latency-Ms": f"{int((time.time() - t0) * 1000)}", "X-Faces": str(len(faces))})


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8189)
    args = p.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
