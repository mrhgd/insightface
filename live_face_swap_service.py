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
# 512x512 is a good quality/speed tradeoff for webcam frames —
# detects smaller faces & landmarks more accurately than 320 without doubling time.
analyzer.prepare(ctx_id=0, det_size=(512, 512))

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

# Detection caching: detection (~30-60ms) is the most expensive step. Re-run it
# every Nth frame and reuse the bboxes/embeddings in between. Faces don't move
# dramatically between adjacent frames at 10+ fps, so this is nearly invisible.
_DET_EVERY = 3
_frame_count = 0
_cached_faces: list = []


def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Could not decode image")
    return img


def _encode_jpeg(img: np.ndarray, quality: int = 92) -> bytes:
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


@app.post("/swap")
async def swap(image: UploadFile = File(...)):
    global _frame_count, _cached_faces
    if source_face is None:
        raise HTTPException(400, "Source face not set. POST /set_source first.")
    t0 = time.time()
    img = _decode(await image.read())

    # Run heavy detection only every N frames; reuse cached faces in between.
    do_detect = (_frame_count % _DET_EVERY == 0) or not _cached_faces
    _frame_count += 1
    if do_detect:
        faces = analyzer.get(img)
        if faces:
            _cached_faces = faces
        else:
            # Lost the face — invalidate cache and return original this frame.
            _cached_faces = []
            return Response(content=_encode_jpeg(img), media_type="image/jpeg",
                            headers={"X-Latency-Ms": f"{int((time.time() - t0) * 1000)}", "X-Faces": "0"})
    else:
        faces = _cached_faces

    res = img.copy()
    for face in faces:
        res = swapper.get(res, face, source_face, paste_back=True)

    out = _encode_jpeg(res)
    return Response(content=out, media_type="image/jpeg",
                    headers={
                        "X-Latency-Ms": f"{int((time.time() - t0) * 1000)}",
                        "X-Faces": str(len(faces)),
                        "X-Detected": "1" if do_detect else "0",
                    })


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8189)
    args = p.parse_args()
    # log_level=warning + access_log=False trims a few ms per request
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False)
