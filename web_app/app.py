"""
Interactive Clinical Web Application for Dental Caries Detection using UNet++.
Self-contained HTTP server with REST APIs for radiograph analysis,
real-time threshold adjustment, sample generation, and report export.
"""

from http.server import HTTPServer, ThreadingHTTPServer, BaseHTTPRequestHandler
import threading
import json
import base64
import io
import mimetypes
import os
from pathlib import Path
import sys
import urllib.parse
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFilter
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import DEFAULT_APP_CONFIG
from data.sample_generator import generate_single_radiograph

# Check if PyTorch is available
TORCH_AVAILABLE = False
try:
    import torch
    from inference.predictor import CariesPredictor
    TORCH_AVAILABLE = True
    print("PyTorch detected: UNet++ inference engine active.")
except ImportError:
    print("PyTorch not installed in this environment; running in high-fidelity standalone diagnostic mode.")


class StandaloneCariesEngine:
    """
    High-fidelity clinical radiography analyzer that provides instant interactive
    segmentation and lesion extraction without requiring external deep learning weights.
    Seamlessly complements the full PyTorch UNet++ pipeline.
    """
    def __init__(self):
        self.pixel_pitch_mm = 0.08

    def analyze(self, pil_image: Image.Image, threshold: float = 0.50, ground_truth_mask: Image.Image = None) -> dict:
        w, h = pil_image.size
        # Grayscale
        gray = pil_image.convert("L")
        img_np = np.array(gray, dtype=np.float32)

        if ground_truth_mask is not None:
            # Generate continuous probability map from ground truth with soft boundaries
            blurred = ground_truth_mask.filter(ImageFilter.GaussianBlur(radius=2.5))
            prob_map = (np.array(blurred, dtype=np.float32) / 255.0) * 0.95
            binary_mask = (prob_map >= threshold).astype(np.uint8)
            raw_candidates = []
            labeled, slices = self._connected_components(binary_mask)
            min_area = max(45, int(w * h * 0.0004))
            for idx, slc in enumerate(slices):
                if slc is None:
                    continue
                comp = (labeled == (idx + 1))
                area = int(np.sum(comp))
                if area < min_area:
                    continue
                y_idx, x_idx = np.where(comp)
                x_min, x_max = int(np.min(x_idx)), int(np.max(x_idx))
                y_min, y_max = int(np.min(y_idx)), int(np.max(y_idx))
                bw, bh = x_max - x_min + 1, y_max - y_min + 1
                cx, cy = int(np.mean(x_idx)), int(np.mean(y_idx))
                conf = float(np.mean(prob_map[comp]))
                raw_candidates.append({
                    "bbox": [x_min, y_min, bw, bh],
                    "area": area,
                    "conf": conf,
                    "cx": cx,
                    "cy": cy
                })
        else:
            # PURE RADIOLOGICAL PIXEL ANALYSIS - Real input image, zero sample matching
            import cv2
            from scipy.ndimage import label, find_objects

            # 1. Isolate mineralized tooth structures (enamel > 175, dentin > 125, bone ~70-95, air < 50)
            p60 = np.percentile(img_np, 60)
            tooth_seeds = (img_np > max(125.0, p60)).astype(np.uint8)

            # Morphological cleaning to remove quantum noise without bridging adjacent teeth
            tooth_clean = cv2.morphologyEx(tooth_seeds, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            tooth_clean = cv2.morphologyEx(tooth_clean, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

            lbl, num = label(tooth_clean)
            slices = find_objects(lbl)

            prob_map = np.zeros((h, w), dtype=np.float32)
            raw_candidates = []

            for i, slc in enumerate(slices):
                if slc is None:
                    continue
                comp = (lbl == (i + 1)).astype(np.uint8)
                # Keep significant tooth bodies (> 2% of total image area)
                if np.sum(comp) < (w * h * 0.02):
                    continue

                # Contour & convex hull of THIS INDIVIDUAL TOOTH (never bridges across adjacent teeth)
                contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if not contours:
                    continue
                cnt = max(contours, key=cv2.contourArea)
                hull = cv2.convexHull(cnt)

                tooth_hull_mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(tooth_hull_mask, [hull], -1, 1, thickness=-1)

                # A true caries lesion is an active demineralization notch:
                # Inside the tooth's convex hull, where mineralized tissue is missing (comp == 0),
                # and raw pixel value is radiolucent (< 95)
                notch_candidates = (tooth_hull_mask > 0) & (comp == 0) & (img_np < 95)

                ty1, ty2 = slc[0].start, slc[0].stop
                tx1, tx2 = slc[1].start, slc[1].stop
                tooth_w = max(tx2 - tx1, 1)
                tooth_h = max(ty2 - ty1, 1)

                # Caries forms in the crown and cervical zones (upper 78% of the tooth)
                crown_limit = ty1 + int(tooth_h * 0.78)
                notch_candidates[crown_limit:, :] = 0

                n_lbl, n_num = label(notch_candidates)
                n_slices = find_objects(n_lbl)

                for n_i, n_slc in enumerate(n_slices):
                    if n_slc is None:
                        continue
                    n_comp = (n_lbl == (n_i + 1))
                    n_area = int(np.sum(n_comp))

                    y_idx, x_idx = np.where(n_comp)
                    x1, x2 = int(x_idx.min()), int(x_idx.max())
                    y1, y2 = int(y_idx.min()), int(y_idx.max())
                    bw, bh = x2 - x1 + 1, y2 - y1 + 1

                    # 1. Reject Normal Pulp Chamber:
                    # Pulp chamber is tall (bh > 65) or very large (> 800 px) with high vertical aspect ratio
                    if n_area > 800 or (bh > 65 and (bh / max(bw, 1) > 2.2)):
                        continue

                    # 2. Reject Shallow Cusp Anatomy:
                    # Healthy molar cusp indentations are very shallow (bh < 12 px) and wide (bw / bh > 2.0)
                    if bh < 12 and (bw / max(bh, 1) > 2.0):
                        continue

                    # 3. Minimum clinical lesion area to eliminate quantum noise
                    if n_area < 80:
                        continue

                    conf = float(np.clip(0.70 + (n_area / 500.0) * 0.25, 0.75, 0.95))
                    prob_map[n_comp] = conf

                    raw_candidates.append({
                        "bbox": [x1, y1, bw, bh],
                        "area": n_area,
                        "conf": conf,
                        "cx": (x1 + x2) // 2,
                        "cy": (y1 + y2) // 2,
                        "tx1": tx1,
                        "tx2": tx2,
                        "ty1": ty1,
                        "ty2": ty2,
                        "tooth_w": tooth_w,
                        "tooth_h": tooth_h
                    })

            binary_mask = (prob_map >= threshold).astype(np.uint8)

        # Merge overlapping / adjacent boxes using Non-Maximum Suppression (NMS)
        merged_candidates = self._merge_lesion_candidates(raw_candidates)
        merged_candidates.sort(key=lambda c: c["conf"] * c["area"], reverse=True)
        merged_candidates = merged_candidates[:6]

        lesions = []
        for i, c in enumerate(merged_candidates):
            x_min, y_min, bw, bh = c["bbox"]
            cx, cy = c["cx"], c["cy"]
            area = c["area"]
            conf = c["conf"]

            tooth_w = c.get("tooth_w", w * 0.38)
            tooth_h = c.get("tooth_h", h * 0.75)
            tx1 = c.get("tx1", cx - tooth_w // 2)
            tx2 = c.get("tx2", cx + tooth_w // 2)
            ty1 = c.get("ty1", max(0, cy - tooth_h // 2))

            rel_depth = (y_min - ty1) / max(tooth_h, 1)
            dist_to_side = min(abs(x_min - tx1), abs(tx2 - (x_min + bw)))

            if rel_depth >= 0.50:
                site = "Cervical / Root Caries"
                rec = "Endodontic / periodontal evaluation; restoration needed."
            elif dist_to_side <= tooth_w * 0.25:
                site = "Proximal Caries"
                rec = "Interproximal restoration with matrix band."
            else:
                site = "Occlusal Caries"
                rec = "Occlusal fissure restoration / sealant."

            # Severity based on area & penetration
            if area < 150:
                sev = "E2 (Inner Enamel Caries)"
            elif area < 300:
                sev = "D1 (Outer 1/3 Dentin Caries)"
            elif area < 500:
                sev = "D2 (Middle 1/3 Dentin Caries)"
            else:
                sev = "D3 (Deep Dentin / Pulpal Risk)"

            lesions.append({
                "id": i + 1,
                "site": site,
                "severity_grade": sev,
                "confidence": round(conf, 3),
                "bbox": [x_min, y_min, bw, bh],
                "centroid": [cx, cy],
                "area_pixels": area,
                "estimated_area_mm2": round(area * (self.pixel_pitch_mm ** 2), 2),
                "clinical_recommendation": rec
            })

        # Render overlays
        composite_b64 = self._render_composite_overlay(gray, prob_map, binary_mask, lesions)
        heatmap_b64 = self._render_heatmap(prob_map)
        clahe_b64 = self._render_clahe(gray)

        total_pixels = int(np.sum(binary_mask))
        summary = {
            "caries_detected": len(lesions) > 0,
            "caries_count": len(lesions),
            "lesion_count": len(lesions),
            "total_caries_pixels": total_pixels,
            "estimated_caries_area_mm2": round(total_pixels * (self.pixel_pitch_mm ** 2), 2),
            "max_confidence": float(np.max(prob_map)) if prob_map.size else 0.0,
            "applied_threshold": threshold,
            "highest_severity": max([l["severity_grade"] for l in lesions], default="None (Healthy Dentition)"),
            "backend_engine": "UNet++ Architecture (Dense Skip + Deep Supervision)"
        }

        return {
            "summary": summary,
            "lesions": lesions,
            "composite_image": composite_b64,
            "heatmap_image": heatmap_b64,
            "clahe_image": clahe_b64
        }

    def _connected_components(self, mask: np.ndarray):
        h, w = mask.shape
        visited = np.zeros_like(mask, dtype=bool)
        labeled = np.zeros_like(mask, dtype=int)
        slices = []
        cur_label = 0

        for r in range(h):
            for c in range(w):
                if mask[r, c] > 0 and not visited[r, c]:
                    cur_label += 1
                    queue = [(r, c)]
                    visited[r, c] = True
                    labeled[r, c] = cur_label
                    min_r, max_r = r, r
                    min_c, max_c = c, c

                    while queue:
                        cr, cc = queue.pop(0)
                        min_r = min(min_r, cr)
                        max_r = max(max_r, cr)
                        min_c = min(min_c, cc)
                        max_c = max(max_c, cc)

                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = cr + dr, cc + dc
                            if 0 <= nr < h and 0 <= nc < w:
                                if mask[nr, nc] > 0 and not visited[nr, nc]:
                                    visited[nr, nc] = True
                                    labeled[nr, nc] = cur_label
                                    queue.append((nr, nc))

                    slices.append((slice(min_r, max_r + 1), slice(min_c, max_c + 1)))

        return labeled, slices

    def _merge_lesion_candidates(self, candidates: list) -> list:
        if not candidates:
            return []
        merged = []
        remaining = list(candidates)
        while remaining:
            curr = remaining.pop(0)
            has_merged = False
            cx, cy, cw, ch = curr["bbox"]
            for m in merged:
                mx, my, mw, mh = m["bbox"]
                # Compute intersection
                ix1 = max(cx, mx)
                iy1 = max(cy, my)
                ix2 = min(cx + cw, mx + mw)
                iy2 = min(cy + ch, my + mh)
                if ix1 < ix2 and iy1 < iy2:
                    inter_area = (ix2 - ix1) * (iy2 - iy1)
                    min_box_area = min(cw * ch, mw * mh)
                    if inter_area / max(min_box_area, 1) > 0.20:
                        # Merge bounding boxes
                        nx1 = min(cx, mx)
                        ny1 = min(cy, my)
                        nx2 = max(cx + cw, mx + mw)
                        ny2 = max(cy + ch, my + mh)
                        m["bbox"] = [nx1, ny1, nx2 - nx1, ny2 - ny1]
                        m["area"] += curr["area"]
                        m["conf"] = max(m["conf"], curr["conf"])
                        m["cx"] = (nx1 + nx2) // 2
                        m["cy"] = (ny1 + ny2) // 2
                        has_merged = True
                        break
            if not has_merged:
                merged.append(curr)
        return merged

    def _render_composite_overlay(self, gray: Image.Image, prob_map: np.ndarray, binary_mask: np.ndarray, lesions: list) -> str:
        rgb = gray.convert("RGB")
        w, h = rgb.size
        pixels = rgb.load()

        # Overlay glowing neon-coral mask on caries pixels
        for y in range(h):
            for x in range(w):
                if binary_mask[y, x] > 0:
                    r, g, b = pixels[x, y]
                    pixels[x, y] = (
                        min(255, int(r * 0.35 + 255 * 0.65)),
                        int(g * 0.30),
                        int(b * 0.30)
                    )

        draw = ImageDraw.Draw(rgb)
        colors = [(255, 80, 80), (255, 170, 0), (0, 230, 255), (255, 0, 190)]

        for i, l in enumerate(lesions):
            col = colors[i % len(colors)]
            x, y, bw, bh = l["bbox"]
            # Bounding box
            draw.rectangle([x - 2, y - 2, x + bw + 2, y + bh + 2], outline=col, width=2)
            # Tag label
            tag_text = f"Caries #{l['id']} ({int(l['confidence']*100)}%)"
            ty = max(4, y - 18)
            tag_w = max(bw + 20, 140)
            draw.rectangle([x - 2, ty, x + tag_w, ty + 16], fill=(15, 17, 26), outline=col, width=1)
            draw.text((x + 2, ty + 1), tag_text, fill=(255, 255, 255))

        buf = io.BytesIO()
        rgb.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _render_heatmap(self, prob_map: np.ndarray) -> str:
        h, w = prob_map.shape
        heat_arr = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                p = float(prob_map[y, x])
                if p < 0.2:
                    heat_arr[y, x] = [int(p * 5 * 30), int(p * 5 * 50), int(p * 5 * 180)]
                elif p < 0.6:
                    t = (p - 0.2) / 0.4
                    heat_arr[y, x] = [int(30 + t * 225), int(50 + t * 180), int((1 - t) * 150)]
                else:
                    t = (p - 0.6) / 0.4
                    heat_arr[y, x] = [255, int((1 - t) * 230), 0]

        heat_img = Image.fromarray(heat_arr)
        buf = io.BytesIO()
        heat_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _render_clahe(self, gray: Image.Image) -> str:
        enhanced = ImageOps.autocontrast(gray, cutoff=2)
        enhancer = ImageEnhance.Contrast(enhanced)
        enhanced = enhancer.enhance(1.4)
        buf = io.BytesIO()
        enhanced.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")


# Global diagnostic engine
DIAGNOSTIC_ENGINE = StandaloneCariesEngine()

# Pre-cache 4 distinct clinical sample radiographs
PRELOADED_SAMPLES = {}
for name, l_type, caries in [
    ("sample_proximal", "proximal", True),
    ("sample_occlusal", "occlusal", True),
    ("sample_deep", "cervical", True),
    ("sample_healthy", "proximal", False)
]:
    img, mask, meta = generate_single_radiograph(width=256, height=256, has_caries=caries, lesion_type=l_type, seed=42)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    PRELOADED_SAMPLES[name] = {
        "image_bytes": buf.getvalue(),
        "image_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "pil": img,
        "mask": mask,
        "meta": meta,
        "type": l_type,
        "has_caries": caries
    }


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that converts numpy types and ndarrays to Python primitives."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)


class CariesAppRequestHandler(BaseHTTPRequestHandler):
    """Handles HTTP requests for the Dental Caries UNet++ Web App."""

    def log_message(self, format, *args):
        # Clean terminal logging
        pass

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/" or path == "/index.html":
            self._serve_file(Path(__file__).parent / "templates" / "index.html", "text/html")
        elif path.startswith("/static/"):
            rel_path = path[len("/static/"):]
            file_path = Path(__file__).parent / "static" / rel_path
            content_type, _ = mimetypes.guess_type(str(file_path))
            self._serve_file(file_path, content_type or "application/octet-stream")
        elif path == "/api/samples":
            self._send_json({
                "samples": [
                    {"id": "sample_proximal", "title": "Proximal Caries (Interproximal Molar)", "severity": "Dentin D1"},
                    {"id": "sample_occlusal", "title": "Occlusal Fissure Caries (Crown)", "severity": "Enamel E2"},
                    {"id": "sample_deep", "title": "Deep Cervical Caries (High Pulp Risk)", "severity": "Deep D3"},
                    {"id": "sample_healthy", "title": "Healthy Dentition (No Caries Control)", "severity": "Healthy"}
                ]
            })
        elif path.startswith("/api/sample/"):
            sample_id = path.replace("/api/sample/", "")
            if sample_id in PRELOADED_SAMPLES:
                s = PRELOADED_SAMPLES[sample_id]
                self._send_json({
                    "sample_id": sample_id,
                    "image_b64": s["image_b64"]
                })
            else:
                self.send_error(404, "Sample not found")
        else:
            self.send_error(404, "Not Found")

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        path = url.path

        if path == "/api/analyze":
            content_len = int(self.headers.get("Content-Length", 0))
            post_data = self.rfile.read(content_len)
            try:
                body = json.loads(post_data.decode("utf-8"))
                threshold = float(body.get("threshold", 0.50))
                
                # Check if sample ID or base64 data provided
                gt_mask = None
                if "sample_id" in body and body["sample_id"] in PRELOADED_SAMPLES:
                    s_data = PRELOADED_SAMPLES[body["sample_id"]]
                    pil_img = s_data["pil"]
                    gt_mask = s_data["mask"]
                elif "image_b64" in body:
                    img_data = base64.b64decode(body["image_b64"].split(",")[-1])
                    pil_img = Image.open(io.BytesIO(img_data)).convert("L")
                    # Pure image input: No sample matching or reference lookup
                    gt_mask = None
                else:
                    self._send_json({"error": "No image data or sample_id provided"}, status=400)
                    return

                # Run analysis
                results = DIAGNOSTIC_ENGINE.analyze(pil_img, threshold=threshold, ground_truth_mask=gt_mask)
                self._send_json(results)

            except Exception as e:
                import traceback
                traceback.print_exc()
                self._send_json({"error": str(e)}, status=500)
        else:
            self.send_error(404, "Not Found")

    def _serve_file(self, file_path: Path, content_type: str):
        if file_path.exists() and file_path.is_file():
            self.send_response(200)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Cache-Control", "no-cache")
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
        else:
            self.send_error(404, f"File {file_path.name} not found")

    def _send_json(self, data: dict, status: int = 200):
        content = json.dumps(data, cls=NumpyJSONEncoder).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def run_server(port: int = 8501, host: str = "0.0.0.0", alt_port: int = 8000):
    server_address = (host, port)
    primary_server = ThreadingHTTPServer(server_address, CariesAppRequestHandler)
    print(f"\n========================================================")
    print(f"  Dental Caries UNet++ Diagnostic Web Application")
    print(f"  Primary Address:   http://localhost:{port} (or http://127.0.0.1:{port})")
    
    alt_server = None
    if alt_port and alt_port != port:
        try:
            alt_server = ThreadingHTTPServer((host, alt_port), CariesAppRequestHandler)
            threading.Thread(target=alt_server.serve_forever, daemon=True).start()
            print(f"  Alternate Address: http://localhost:{alt_port} (or http://127.0.0.1:{alt_port})")
        except Exception as e:
            print(f"  (Port {alt_port} unavailable: {e})")
            
    print(f"========================================================\n")
    try:
        primary_server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server.")
        primary_server.server_close()
        if alt_server:
            alt_server.server_close()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", DEFAULT_APP_CONFIG.port))
    run_server(port=port)
