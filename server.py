"""
==========================================================================
  EHSS Safety Hazard Detection — Local API Server
==========================================================================
  Server FastAPI lokal yang langsung pakai model best.pt + SAHI.
  
  CARA JALANKAN:
    1. pip install -r requirements.txt
    2. python server.py
    3. Buka http://localhost:8000/docs  ← Swagger UI untuk testing
    
  ENDPOINTS:
    POST /detect          → Upload gambar, dapat JSON hasil deteksi
    POST /detect-sahi     → Sama tapi pakai SAHI (lebih akurat)
    GET  /health          → Cek apakah server running
    GET  /docs            → Swagger UI (auto-generated)
==========================================================================
"""

import os
import io
import time
import base64
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# ============================================================
# CONFIG
# ============================================================
MODEL_PATH = "best.pt"
CONFIDENCE_THRESHOLD = 0.25
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# SAHI settings
SAHI_SLICE_HEIGHT = 320
SAHI_SLICE_WIDTH = 320
SAHI_OVERLAP_RATIO = 0.3

# Class info
CLASS_NAMES = [
    "person", "helmet", "safety_vest", "wet_floor",
    "blocked_walkway", "exposed_cable", "chemical_spill"
]

CLASS_COLORS = {
    "person": "#FF6B6B",
    "helmet": "#4ECDC4",
    "safety_vest": "#FFE66D",
    "wet_floor": "#45B7D1",
    "blocked_walkway": "#F7934C",
    "exposed_cable": "#A855F7",
    "chemical_spill": "#EF4444",
}

# ============================================================
# LOAD MODELS (1x saat startup)
# ============================================================
print("=" * 60)
print("  EHSS Safety Hazard Detection — Local Server")
print("=" * 60)
print(f"  Model   : {MODEL_PATH}")
print(f"  Device  : {DEVICE}")

if not Path(MODEL_PATH).exists():
    print(f"\n❌ ERROR: Model file '{MODEL_PATH}' tidak ditemukan!")
    print(f"   Pastikan file ada di folder: {Path('.').resolve()}")
    exit(1)

# Load YOLO model
print("\n📦 Loading YOLO model...")
yolo_model = YOLO(MODEL_PATH)
print("   ✅ YOLO model loaded!")

# Load SAHI model
sahi_model = None
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    print("📦 Loading SAHI model...")
    sahi_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=MODEL_PATH,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        device=DEVICE,
    )
    print("   ✅ SAHI model loaded!")
    SAHI_AVAILABLE = True
except ImportError:
    print("   ⚠️  SAHI not installed. /detect-sahi endpoint will be unavailable.")
    print("      Install: pip install sahi>=0.11.0")
    SAHI_AVAILABLE = False

print(f"\n  🚀 Server ready! Open http://localhost:8000/docs\n")
print("=" * 60)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="EHSS Safety Hazard Detection API",
    description=(
        "API untuk deteksi hazard keselamatan kerja menggunakan YOLOv11 + SAHI.\n\n"
        "**7 Kelas yang dideteksi:**\n"
        "person, helmet, safety_vest, wet_floor, blocked_walkway, exposed_cable, chemical_spill\n\n"
        "**Endpoints:**\n"
        "- `POST /detect` — Deteksi standar (cepat)\n"
        "- `POST /detect-sahi` — Deteksi dengan SAHI (lebih akurat untuk objek kecil)\n"
        "- `POST /detect-visual` — Deteksi + return gambar dengan bounding box\n"
    ),
    version="1.0.0",
)

# CORS — supaya frontend web bisa akses API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Ganti dengan domain frontend kamu di production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/docs")


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def format_detections(boxes, model):
    """Format YOLO results ke list of dict."""
    detections = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_name": model.names[cls_id],
            "class_id": cls_id,
            "confidence": round(conf, 4),
            "bbox": {
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
                "width": round(x2 - x1, 2),
                "height": round(y2 - y1, 2),
            }
        })
    return detections


def format_sahi_detections(sahi_result):
    """Format SAHI results ke list of dict."""
    detections = []
    for pred in sahi_result.object_prediction_list:
        bbox = pred.bbox.to_xyxy()
        detections.append({
            "class_name": pred.category.name,
            "class_id": pred.category.id,
            "confidence": round(pred.score.value, 4),
            "bbox": {
                "x1": round(bbox[0], 2),
                "y1": round(bbox[1], 2),
                "x2": round(bbox[2], 2),
                "y2": round(bbox[3], 2),
                "width": round(bbox[2] - bbox[0], 2),
                "height": round(bbox[3] - bbox[1], 2),
            }
        })
    return detections


def make_summary(detections):
    """Hitung summary per kelas."""
    summary = {}
    for d in detections:
        name = d["class_name"]
        summary[name] = summary.get(name, 0) + 1
    return summary


def draw_boxes_on_image(img: Image.Image, detections: list) -> Image.Image:
    """Gambar bounding box pada image."""
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for det in detections:
        color = CLASS_COLORS.get(det["class_name"], "#FFFFFF")
        b = det["bbox"]
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]

        # Box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        # Label
        label = f"{det['class_name']} {det['confidence']:.0%}"
        text_bbox = draw.textbbox((x1, y1 - 20), label, font=font)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1, y1 - 20), label, fill="black", font=font)

    return img


async def save_upload_temp(image: UploadFile) -> str:
    """Simpan uploaded file ke temp path."""
    content = await image.read()
    suffix = Path(image.filename or "img.jpg").suffix or ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        return tmp.name


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
async def health_check():
    """Cek apakah server running dan model loaded."""
    return {
        "status": "ok",
        "model": MODEL_PATH,
        "device": DEVICE,
        "sahi_available": SAHI_AVAILABLE,
        "classes": CLASS_NAMES,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
    }


@app.post("/detect")
async def detect(
    image: UploadFile = File(..., description="Upload gambar (JPG/PNG)"),
    confidence: float = Query(
        default=CONFIDENCE_THRESHOLD,
        ge=0.01, le=1.0,
        description="Minimum confidence threshold (0.01 - 1.0)"
    ),
):
    """
    🔍 Deteksi standar (tanpa SAHI) — **Cepat**
    
    Upload gambar, dapat JSON hasil deteksi.
    Cocok untuk gambar dengan objek berukuran normal.
    """
    tmp_path = await save_upload_temp(image)

    try:
        start = time.time()
        results = yolo_model.predict(source=tmp_path, conf=confidence, verbose=False)
        elapsed = time.time() - start

        detections = []
        if results and len(results) > 0:
            detections = format_detections(results[0].boxes, yolo_model)

        return {
            "success": True,
            "mode": "standard",
            "total_detections": len(detections),
            "inference_time_ms": round(elapsed * 1000, 1),
            "confidence_threshold": confidence,
            "summary": make_summary(detections),
            "detections": detections,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/detect-sahi")
async def detect_sahi(
    image: UploadFile = File(..., description="Upload gambar (JPG/PNG)"),
    confidence: float = Query(
        default=CONFIDENCE_THRESHOLD,
        ge=0.01, le=1.0,
        description="Minimum confidence threshold"
    ),
    slice_size: int = Query(
        default=SAHI_SLICE_HEIGHT,
        ge=128, le=1024,
        description="Ukuran slice SAHI (pixel)"
    ),
):
    """
    🔍 Deteksi dengan SAHI — **Lebih akurat** untuk objek kecil
    
    Gambar dipotong jadi slice-slice kecil, tiap slice di-detect, 
    hasilnya digabung. Lebih lambat tapi akurasi jauh lebih tinggi
    terutama untuk exposed_cable dan objek kecil lainnya.
    """
    if not SAHI_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SAHI not installed. Run: pip install sahi>=0.11.0"
        )

    tmp_path = await save_upload_temp(image)

    try:
        sahi_model.confidence_threshold = confidence

        start = time.time()
        result = get_sliced_prediction(
            tmp_path,
            sahi_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=SAHI_OVERLAP_RATIO,
            overlap_width_ratio=SAHI_OVERLAP_RATIO,
            verbose=0,
        )
        elapsed = time.time() - start

        detections = format_sahi_detections(result)

        return {
            "success": True,
            "mode": "sahi",
            "total_detections": len(detections),
            "inference_time_ms": round(elapsed * 1000, 1),
            "confidence_threshold": confidence,
            "slice_size": slice_size,
            "summary": make_summary(detections),
            "detections": detections,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/detect-visual")
async def detect_visual(
    image: UploadFile = File(..., description="Upload gambar (JPG/PNG)"),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.01, le=1.0),
    use_sahi: bool = Query(default=False, description="Pakai SAHI?"),
):
    """
    🖼️ Deteksi + return gambar dengan bounding box
    
    Return gambar PNG dengan bounding box tergambar.
    Berguna untuk preview/testing langsung di browser.
    """
    tmp_path = await save_upload_temp(image)

    try:
        # Detect
        if use_sahi and SAHI_AVAILABLE:
            sahi_model.confidence_threshold = confidence
            result = get_sliced_prediction(
                tmp_path, sahi_model,
                slice_height=SAHI_SLICE_HEIGHT,
                slice_width=SAHI_SLICE_WIDTH,
                overlap_height_ratio=SAHI_OVERLAP_RATIO,
                overlap_width_ratio=SAHI_OVERLAP_RATIO,
                verbose=0,
            )
            detections = format_sahi_detections(result)
        else:
            results = yolo_model.predict(source=tmp_path, conf=confidence, verbose=False)
            detections = format_detections(results[0].boxes, yolo_model) if results else []

        # Draw boxes
        img = Image.open(tmp_path).convert("RGB")
        img = draw_boxes_on_image(img, detections)

        # Return as PNG
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "X-Detections-Count": str(len(detections)),
                "X-Detection-Mode": "sahi" if (use_sahi and SAHI_AVAILABLE) else "standard",
            }
        )
    finally:
        os.unlink(tmp_path)


@app.post("/detect-full")
async def detect_full(
    image: UploadFile = File(..., description="Upload gambar (JPG/PNG)"),
    confidence: float = Query(default=CONFIDENCE_THRESHOLD, ge=0.01, le=1.0),
    use_sahi: bool = Query(default=False, description="Pakai SAHI?"),
):
    """
    🔍🖼️ Deteksi lengkap — JSON + gambar base64
    
    Return JSON hasil deteksi + gambar dengan bounding box dalam base64.
    Ini endpoint paling lengkap, cocok untuk frontend yang perlu 
    data JSON sekaligus preview gambar.
    """
    tmp_path = await save_upload_temp(image)

    try:
        start = time.time()

        if use_sahi and SAHI_AVAILABLE:
            sahi_model.confidence_threshold = confidence
            result = get_sliced_prediction(
                tmp_path, sahi_model,
                slice_height=SAHI_SLICE_HEIGHT,
                slice_width=SAHI_SLICE_WIDTH,
                overlap_height_ratio=SAHI_OVERLAP_RATIO,
                overlap_width_ratio=SAHI_OVERLAP_RATIO,
                verbose=0,
            )
            detections = format_sahi_detections(result)
            mode = "sahi"
        else:
            results = yolo_model.predict(source=tmp_path, conf=confidence, verbose=False)
            detections = format_detections(results[0].boxes, yolo_model) if results else []
            mode = "standard"

        elapsed = time.time() - start

        # Draw boxes
        img = Image.open(tmp_path).convert("RGB")
        img = draw_boxes_on_image(img, detections)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "mode": mode,
            "total_detections": len(detections),
            "inference_time_ms": round(elapsed * 1000, 1),
            "summary": make_summary(detections),
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{img_base64}",
        }
    finally:
        os.unlink(tmp_path)


# ============================================================
# RUN SERVER
# ============================================================
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Railway provides PORT
    print(f"\n🌐 Starting server at http://localhost:{port}")
    print(f"📖 Swagger UI at http://localhost:{port}/docs\n")
    uvicorn.run(app, host="0.0.0.0", port=port)
