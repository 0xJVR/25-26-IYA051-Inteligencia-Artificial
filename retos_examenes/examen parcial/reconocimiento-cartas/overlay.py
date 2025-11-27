import os
import cv2 as cv
import numpy as np
from config import CFG
from utils import center_of_quad

try:
    from PIL import Image, ImageDraw, ImageFont
    _PIL_OK = True
except Exception:
    _PIL_OK = False

_BASES = {'S': 0x1F0A0, 'H': 0x1F0B0, 'D': 0x1F0C0, 'C': 0x1F0D0}

def _rank_offset(rank):
    if rank == 'A': return 1
    if rank == 'J': return 11
    if rank == 'Q': return 13
    if rank == 'K': return 14
    try: return int(rank)
    except: return None

def card_to_unicode(rank, suit):
    base = _BASES.get(suit)
    off = _rank_offset(rank)
    if base is None or off is None: return None
    return chr(base + off)

def overlay_texts_from_dets(dets):
    if not dets: return "", ""
    ordered = sorted(dets, key=lambda d: d["quad"][:, 0].mean())
    unic, ascii_cards = [], []
    for d in ordered:
        r, s = d["rank"], d["suit"]
        ch = card_to_unicode(r, s)
        if ch: unic.append(ch)
        ascii_cards.append(f"{r}{s}")
    return " ".join(unic), " ".join(ascii_cards)

def draw_bottom_overlay(img_bgr, dets):
    if not CFG.get("overlay", {}).get("show", True): return img_bgr
    uni, asc = overlay_texts_from_dets(dets)
    text = uni if uni else asc
    if not text: return img_bgr

    H, W = img_bgr.shape[:2]
    pad = int(CFG["overlay"]["padding"])
    fs = int(CFG["overlay"]["font_size"])
    bar_h = fs + pad * 2

    if _PIL_OK:
        try:
            im = Image.fromarray(cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)).convert("RGBA")
            draw = ImageDraw.Draw(im, "RGBA")
            draw.rectangle([(0, H - bar_h), (W, H)], fill=(0, 0, 0, int(255 * float(CFG["overlay"]["bg_alpha"]))))
            font = None
            fp = CFG["overlay"].get("font_path", "")
            if fp and os.path.isfile(fp):
                font = ImageFont.truetype(fp, fs)
            else:
                for trial in ["DejaVuSans.ttf", "/usr/share/fonts/truetype/noto/NotoColorEmoji.ttf"]:
                    try:
                        font = ImageFont.truetype(trial, fs)
                        break
                    except: pass
            if not font: font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y = (W - tw) // 2, H - pad - th
            draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))
            return cv.cvtColor(np.array(im.convert("RGB")), cv.COLOR_RGB2BGR)
        except Exception: pass

    # Fallback OpenCV
    display = asc if asc else text
    overlay = img_bgr.copy()
    cv.rectangle(overlay, (0, H - bar_h), (W, H), (0, 0, 0), -1)
    out = cv.addWeighted(overlay, float(CFG["overlay"]["bg_alpha"]), img_bgr, 1 - float(CFG["overlay"]["bg_alpha"]), 0)
    (tw, th), _ = cv.getTextSize(display, cv.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv.putText(out, display, ((W - tw) // 2, H - pad - 4), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
    return out

def draw_quad(img, quad, color=(60, 220, 60), thickness=2):
    pts = quad.astype(int).reshape(-1, 1, 2)
    cv.polylines(img, [pts], True, color, thickness)

def draw_label(img, text, org, bg=(0, 0, 0), fg=(255, 255, 255)):
    (tw, th), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x, y = int(org[0]), int(org[1])
    cv.rectangle(img, (x, y - th - 8), (x + tw + 6, y + 4), bg, -1)
    cv.putText(img, text, (x + 3, y - 2), cv.FONT_HERSHEY_SIMPLEX, 0.7, fg, 2, cv.LINE_AA)

def prompt_overlay(frame, text):
    f = frame.copy()
    h, w = f.shape[:2]
    cv.rectangle(f, (0, 0), (w, 60), (0, 0, 0), -1)
    cv.putText(f, text, (12, 40), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
    return f
