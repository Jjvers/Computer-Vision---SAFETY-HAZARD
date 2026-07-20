"""
==========================================================================
  EHSS Safety Hazard Detection — SAHI Inference Module
==========================================================================
  Modul ini adalah ENGINE deteksi yang bisa di-import ke web app manapun.
  
  CARA PAKAI di web app kamu:
  
    from sahi_detector import SafetyDetector
    
    detector = SafetyDetector("best (1).pt")
    results = detector.detect("gambar.jpg")
    
  Mendukung:
    - Deteksi gambar tunggal (file path / PIL Image / base64)
    - Deteksi batch (folder / list of images)
    - SAHI slicing untuk akurasi tinggi pada objek kecil
    - Mode cepat tanpa SAHI untuk real-time
    - Export hasil dengan bounding box visual
    
  INSTALL:
    pip install ultralytics>=8.3.0 sahi>=0.11.0 Pillow
==========================================================================
"""

import os
import io
import base64
import time
from pathlib import Path
from typing import Union, List, Optional, Dict, Any

# PIL for image handling
from PIL import Image


class DetectionResult:
    """Hasil deteksi satu objek."""

    def __init__(self, class_name: str, class_id: int, confidence: float,
                 bbox: List[float], bbox_format: str = "xyxy"):
        self.class_name = class_name
        self.class_id = class_id
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.bbox_format = bbox_format

    def to_dict(self) -> Dict[str, Any]:
        """Convert ke dictionary — mudah untuk JSON response di web app."""
        return {
            "class_name": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": [round(v, 2) for v in self.bbox],
        }

    def __repr__(self):
        return (f"Detection({self.class_name}, conf={self.confidence:.2f}, "
                f"bbox={[round(v, 1) for v in self.bbox]})")


class ImageDetectionResult:
    """Hasil deteksi untuk satu gambar (bisa berisi banyak objek)."""

    def __init__(self, image_path: str, detections: List[DetectionResult],
                 inference_time: float, used_sahi: bool):
        self.image_path = image_path
        self.detections = detections
        self.inference_time = inference_time  # dalam detik
        self.used_sahi = used_sahi

    @property
    def count(self) -> int:
        return len(self.detections)

    def get_by_class(self, class_name: str) -> List[DetectionResult]:
        """Filter deteksi berdasarkan nama kelas."""
        return [d for d in self.detections if d.class_name == class_name]

    def summary(self) -> Dict[str, int]:
        """Hitung jumlah tiap kelas yang terdeteksi."""
        counts = {}
        for d in self.detections:
            counts[d.class_name] = counts.get(d.class_name, 0) + 1
        return counts

    def to_dict(self) -> Dict[str, Any]:
        """Convert ke dictionary — langsung bisa jadi JSON response."""
        return {
            "image": self.image_path,
            "total_detections": self.count,
            "inference_time_ms": round(self.inference_time * 1000, 1),
            "used_sahi": self.used_sahi,
            "summary": self.summary(),
            "detections": [d.to_dict() for d in self.detections],
        }

    def __repr__(self):
        return (f"ImageResult({self.image_path}, "
                f"{self.count} detections, "
                f"{self.inference_time*1000:.0f}ms, "
                f"sahi={'yes' if self.used_sahi else 'no'})")


class SafetyDetector:
    """
    EHSS Safety Hazard Detector dengan SAHI.
    
    Ini adalah class utama yang kamu import ke web app.
    
    Contoh penggunaan:
    
        # === Inisialisasi (1x saja, biasanya di startup app) ===
        detector = SafetyDetector(
            model_path="best (1).pt",
            device="cuda:0",           # atau "cpu"
            confidence=0.25,
        )
        
        # === Deteksi gambar (panggil tiap request) ===
        
        # Dari file path:
        result = detector.detect("foto.jpg")
        
        # Dari base64 (upload via web):
        result = detector.detect_base64(base64_string)
        
        # Dari PIL Image:
        result = detector.detect_pil(pil_image)
        
        # Batch (banyak gambar sekaligus):
        results = detector.detect_batch(["img1.jpg", "img2.jpg"])
        
        # === Ambil hasilnya ===
        print(result.to_dict())   # → langsung jadi JSON
        print(result.summary())   # → {"person": 3, "helmet": 2, ...}
    """

    # 8 kelas safety hazard
    CLASS_NAMES = [
        "person", "trolley", "phone", "apron", 
        "safety_glasses", "safety_gloves", "safety_boots", "safety_helmet"
    ]

    def __init__(
        self,
        model_path: str = "best (1).pt",
        device: str = "auto",
        confidence: float = 0.25,
        # SAHI parameters
        slice_height: int = 320,
        slice_width: int = 320,
        overlap_ratio: float = 0.3,
        use_sahi: bool = True,
    ):
        """
        Inisialisasi detector.
        
        Args:
            model_path: Path ke file .pt model YOLO
            device: "cuda:0", "cpu", atau "auto" (otomatis pilih GPU/CPU)
            confidence: Minimum confidence threshold (0.0 - 1.0)
            slice_height: Tinggi potongan SAHI (pixel)
            slice_width: Lebar potongan SAHI (pixel)
            overlap_ratio: Overlap antar potongan (0.0 - 1.0)
            use_sahi: True = pakai SAHI (akurat), False = inference biasa (cepat)
        """
        self.model_path = model_path
        self.confidence = confidence
        self.slice_height = slice_height
        self.slice_width = slice_width
        self.overlap_ratio = overlap_ratio
        self.use_sahi = use_sahi

        # Auto-detect device
        if device == "auto":
            import torch
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🔧 Initializing SafetyDetector...")
        print(f"   Model  : {model_path}")
        print(f"   Device : {self.device}")
        print(f"   SAHI   : {'ON' if use_sahi else 'OFF'}")

        # Validate model file exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"❌ Model file tidak ditemukan: {model_path}\n"
                f"   Pastikan file best.pt ada di lokasi yang benar."
            )

        # Load SAHI detection model (ini wraps YOLO internally)
        if self.use_sahi:
            try:
                from sahi import AutoDetectionModel
                self.sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="ultralytics",
                    model_path=str(model_path),
                    confidence_threshold=confidence,
                    device=self.device,
                )
                print(f"   ✅ SAHI model loaded successfully!")
            except ImportError:
                print(f"   ⚠️  SAHI not installed! Falling back to standard YOLO.")
                print(f"      Install: pip install sahi>=0.11.0")
                self.use_sahi = False
                self.sahi_model = None

        # Also load standard YOLO model (for fallback and fast mode)
        from ultralytics import YOLO
        self.yolo_model = YOLO(str(model_path))
        
        if not self.use_sahi:
            print(f"   ✅ YOLO model loaded (standard mode).")

        print(f"   ✅ SafetyDetector ready!\n")

    # ================================================================
    # CORE DETECTION METHODS
    # ================================================================

    def detect(
        self,
        image: Union[str, Path],
        use_sahi: Optional[bool] = None,
        confidence: Optional[float] = None,
    ) -> ImageDetectionResult:
        """
        Deteksi objek pada satu gambar.
        
        Args:
            image: Path ke file gambar
            use_sahi: Override SAHI on/off untuk request ini
            confidence: Override confidence threshold untuk request ini
            
        Returns:
            ImageDetectionResult dengan list deteksi
        """
        image_path = str(image)
        sahi_mode = use_sahi if use_sahi is not None else self.use_sahi
        conf = confidence if confidence is not None else self.confidence

        start_time = time.time()

        if sahi_mode and self.sahi_model is not None:
            detections = self._detect_sahi(image_path, conf)
        else:
            detections = self._detect_standard(image_path, conf)

        elapsed = time.time() - start_time

        return ImageDetectionResult(
            image_path=image_path,
            detections=detections,
            inference_time=elapsed,
            used_sahi=sahi_mode,
        )

    def detect_pil(
        self,
        pil_image: Image.Image,
        use_sahi: Optional[bool] = None,
        confidence: Optional[float] = None,
    ) -> ImageDetectionResult:
        """
        Deteksi dari PIL Image object.
        Berguna kalau gambar sudah di-load di memory (e.g., dari upload web).
        """
        # Save PIL to temp path, detect, cleanup
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path, format="JPEG", quality=95)

        try:
            result = self.detect(tmp_path, use_sahi=use_sahi, confidence=confidence)
            result.image_path = "uploaded_image"
            return result
        finally:
            os.unlink(tmp_path)

    def detect_base64(
        self,
        base64_string: str,
        use_sahi: Optional[bool] = None,
        confidence: Optional[float] = None,
    ) -> ImageDetectionResult:
        """
        Deteksi dari base64 encoded image string.
        Berguna untuk web API yang menerima gambar via JSON body.
        
        Contoh di Flask/FastAPI:
            @app.post("/detect")
            def detect(data: dict):
                result = detector.detect_base64(data["image_base64"])
                return result.to_dict()
        """
        # Strip data URL prefix if present (e.g., "data:image/jpeg;base64,...")
        if "," in base64_string:
            base64_string = base64_string.split(",", 1)[1]

        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        return self.detect_pil(pil_image, use_sahi=use_sahi, confidence=confidence)

    def detect_bytes(
        self,
        image_bytes: bytes,
        use_sahi: Optional[bool] = None,
        confidence: Optional[float] = None,
    ) -> ImageDetectionResult:
        """
        Deteksi dari raw bytes.
        Berguna untuk Flask: request.files['image'].read()
        """
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self.detect_pil(pil_image, use_sahi=use_sahi, confidence=confidence)

    def detect_batch(
        self,
        images: List[Union[str, Path]],
        use_sahi: Optional[bool] = None,
        confidence: Optional[float] = None,
    ) -> List[ImageDetectionResult]:
        """
        Deteksi batch (banyak gambar sekaligus).
        
        Args:
            images: List of image paths
            
        Returns:
            List of ImageDetectionResult
        """
        results = []
        total = len(images)
        for idx, img in enumerate(images):
            result = self.detect(img, use_sahi=use_sahi, confidence=confidence)
            results.append(result)
            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"   [{idx+1}/{total}] processed")
        return results

    # ================================================================
    # VISUALIZATION
    # ================================================================

    def detect_and_visualize(
        self,
        image: Union[str, Path],
        output_path: Optional[str] = None,
        use_sahi: Optional[bool] = None,
        confidence: Optional[float] = None,
    ) -> ImageDetectionResult:
        """
        Deteksi + simpan gambar dengan bounding box.
        
        Args:
            image: Path ke gambar input
            output_path: Path output (default: input_detected.jpg)
            
        Returns:
            ImageDetectionResult (gambar output di-save ke disk)
        """
        result = self.detect(image, use_sahi=use_sahi, confidence=confidence)

        # Draw bounding boxes
        img = Image.open(str(image)).convert("RGB")
        self._draw_boxes(img, result.detections)

        if output_path is None:
            p = Path(image)
            output_path = str(p.parent / f"{p.stem}_detected{p.suffix}")

        img.save(output_path, quality=95)
        print(f"   💾 Saved: {output_path}")
        return result

    def get_annotated_image_base64(
        self,
        image: Union[str, Path],
        use_sahi: Optional[bool] = None,
        confidence: Optional[float] = None,
    ) -> tuple:
        """
        Deteksi + return gambar dengan bounding box sebagai base64.
        Berguna untuk mengirim hasil visual langsung ke frontend web.
        
        Returns:
            (result, base64_string)
        """
        result = self.detect(image, use_sahi=use_sahi, confidence=confidence)

        img = Image.open(str(image)).convert("RGB")
        self._draw_boxes(img, result.detections)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=90)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        return result, f"data:image/jpeg;base64,{b64}"

    # ================================================================
    # INTERNAL METHODS
    # ================================================================

    def _detect_sahi(self, image_path: str, conf: float) -> List[DetectionResult]:
        """Inference menggunakan SAHI (sliced prediction)."""
        from sahi.predict import get_sliced_prediction

        # Update confidence if different from init
        self.sahi_model.confidence_threshold = conf

        sahi_result = get_sliced_prediction(
            image_path,
            self.sahi_model,
            slice_height=self.slice_height,
            slice_width=self.slice_width,
            overlap_height_ratio=self.overlap_ratio,
            overlap_width_ratio=self.overlap_ratio,
            verbose=0,
        )

        detections = []
        for pred in sahi_result.object_prediction_list:
            bbox = pred.bbox.to_xyxy()  # [x1, y1, x2, y2]
            detections.append(DetectionResult(
                class_name=pred.category.name,
                class_id=pred.category.id,
                confidence=pred.score.value,
                bbox=bbox,
            ))

        return detections

    def _detect_standard(self, image_path: str, conf: float) -> List[DetectionResult]:
        """Inference YOLO standar (tanpa SAHI, lebih cepat)."""
        results = self.yolo_model.predict(
            source=image_path,
            conf=conf,
            verbose=False,
        )

        detections = []
        if results and len(results) > 0:
            for box in results[0].boxes:
                cls_id = int(box.cls[0])
                detections.append(DetectionResult(
                    class_name=self.yolo_model.names[cls_id],
                    class_id=cls_id,
                    confidence=float(box.conf[0]),
                    bbox=box.xyxy[0].tolist(),
                ))

        return detections

    def _draw_boxes(self, img: Image.Image, detections: List[DetectionResult]):
        """Draw bounding boxes pada PIL Image."""
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)

        # Color per class
        colors = {
            "person": "#FF6B6B",
            "trolley": "#F7934C",
            "phone": "#A855F7",
            "apron": "#45B7D1",
            "safety_glasses": "#4ECDC4",
            "safety_gloves": "#FFE66D",
            "safety_boots": "#FF9F1C",
            "safety_helmet": "#2EC4B6"
        }

        try:
            font = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for det in detections:
            color = colors.get(det.class_name, "#FFFFFF")
            x1, y1, x2, y2 = det.bbox

            # Draw box
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

            # Draw label background
            label = f"{det.class_name} {det.confidence:.0%}"
            text_bbox = draw.textbbox((x1, y1 - 18), label, font=font)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 18), label, fill="black", font=font)


# ================================================================
# CONTOH INTEGRASI KE BERBAGAI WEB FRAMEWORKS
# ================================================================
#
# ============ FLASK ============
#
#   from flask import Flask, request, jsonify
#   from sahi_detector import SafetyDetector
#
#   app = Flask(__name__)
#   detector = SafetyDetector("best (1).pt")   # <-- load 1x saat startup
#
#   @app.route("/detect", methods=["POST"])
#   def detect():
#       if "image" in request.files:
#           # Upload file
#           img_bytes = request.files["image"].read()
#           result = detector.detect_bytes(img_bytes)
#       elif request.json and "image_base64" in request.json:
#           # Base64 JSON
#           result = detector.detect_base64(request.json["image_base64"])
#       else:
#           return jsonify({"error": "No image provided"}), 400
#
#       return jsonify(result.to_dict())
#
#
# ============ FASTAPI ============
#
#   from fastapi import FastAPI, UploadFile, File
#   from sahi_detector import SafetyDetector
#
#   app = FastAPI()
#   detector = SafetyDetector("best (1).pt")
#
#   @app.post("/detect")
#   async def detect(image: UploadFile = File(...)):
#       img_bytes = await image.read()
#       result = detector.detect_bytes(img_bytes)
#       return result.to_dict()
#
#
# ============ DJANGO ============
#
#   # views.py
#   from django.http import JsonResponse
#   from sahi_detector import SafetyDetector
#
#   detector = SafetyDetector("best (1).pt")
#
#   def detect_view(request):
#       if request.method == "POST" and request.FILES.get("image"):
#           img_bytes = request.FILES["image"].read()
#           result = detector.detect_bytes(img_bytes)
#           return JsonResponse(result.to_dict())
#
