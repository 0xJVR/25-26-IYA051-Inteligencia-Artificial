import os
import json

# Paths
ROOT = os.path.abspath(os.path.dirname(__file__)) if "__file__" in globals() else os.getcwd()
CFG_PATH = os.path.join(ROOT, "config.json")
TPL_DIR = os.path.join(ROOT, "templates")
RANK_DIR = os.path.join(TPL_DIR, "ranks")
SUIT_DIR = os.path.join(TPL_DIR, "suits")
OUT_DIR = os.path.join(ROOT, "out")

# Ensure directories exist
for d in [RANK_DIR, SUIT_DIR, OUT_DIR]:
    os.makedirs(d, exist_ok=True)

# Default Configuration
DEFAULT_CFG = {
    "camera_index": 0,
    "card": { "aspect_ratio": 1.52, "canonical_width": 400 },
    "segmentation": {
        "green_hsv_low": [35, 40, 40], "green_hsv_high": [85, 255, 255],
        "white_v_min": 200, "morph_kernel": 5, "open_iters": 2, "close_iters": 2
    },
    "proposals": {
        "min_contour_area": 1000, "min_w": 40, "min_h": 60,
        "card_ar_range": [1.15, 1.65], "watershed_distance_rel": 0.35
    },
    "roi": {
        "tl": [0.02, 0.05, 0.16, 0.26], "br": [0.80, 0.72, 0.20, 0.28],
        "rank_split": 0.55, "expand_px": 0
    },
    "matching": {
        "rank_ncc_min": 0.60, "suit_ncc_min": 0.70,
        "orb_min": 0.10, "nms_iou": 0.50
    },
    "performance": { "parallel_quads": True, "workers": 8 },
    "overlay": {
        "show": True,
        "font_path": "/usr/share/fonts/truetype/noto/NotoEmoji-Regular.ttf",
        "font_size": 36, "padding": 10, "bg_alpha": 0.65
    }
}

# Global Config Object
CFG = {}

def load_config(path=CFG_PATH):
    """Load config.json, writing defaults if missing."""
    global CFG
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_CFG, f, indent=2)
        CFG.update(DEFAULT_CFG)
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        def merge(dflt, got):
            if isinstance(dflt, dict) and isinstance(got, dict):
                out = dict(dflt)
                for k, v in got.items():
                    out[k] = merge(dflt.get(k, v), v)
                return out
            return got

        CFG.update(merge(DEFAULT_CFG, cfg))
    except Exception as e:
        print(f"Failed to read config.json, using defaults. Error: {e}")
        CFG.update(DEFAULT_CFG)

def get_card_dims():
    """Calculate canonical width/height based on current config."""
    ar = float(CFG["card"]["aspect_ratio"])
    w = int(CFG["card"]["canonical_width"])
    h = int(round(w * ar))
    return w, h

# Load immediately on import
load_config()
