import os
import cv2 as cv
import numpy as np
from config import CFG
from core import warp_card
from utils import quad_to_bbox, iou_bboxes

def roi_from_frac(img, frac_rect, expand_px=0):
    h, w = img.shape[:2]
    x = int(frac_rect[0] * w)
    y = int(frac_rect[1] * h)
    ww = int(frac_rect[2] * w)
    hh = int(frac_rect[3] * h)
    x -= int(expand_px); y -= int(expand_px)
    ww += int(2 * expand_px); hh += int(2 * expand_px)
    x = max(0, x); y = max(0, y)
    x2 = min(w, x + ww); y2 = min(h, y + hh)
    if x2 <= x or y2 <= y: return img[0:1, 0:1].copy()
    return img[y:y2, x:x2].copy()

def prep_roi_bin(roi_bgr):
    g = cv.cvtColor(roi_bgr, cv.COLOR_BGR2GRAY)
    g = cv.GaussianBlur(g, (3, 3), 0)
    _, bw = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    if np.mean(bw) < 127: bw = 255 - bw
    return bw

def is_red_region(bgr_roi):
    hsv = cv.cvtColor(bgr_roi, cv.COLOR_BGR2HSV)
    s = hsv[:, :, 1]
    mask_colorful = cv.inRange(s, 60, 255)
    mask_r1 = cv.inRange(hsv, (0, 60, 60), (10, 255, 255))
    mask_r2 = cv.inRange(hsv, (170, 60, 60), (180, 255, 255))
    mask_red = cv.bitwise_or(mask_r1, mask_r2)
    mask = cv.bitwise_and(mask_red, mask_colorful)
    red_ratio = float(cv.countNonZero(mask)) / (mask.size + 1e-6)
    return red_ratio > 0.03

def load_templates(dirpath):
    lib = {}
    if not os.path.isdir(dirpath): return lib
    for fn in os.listdir(dirpath):
        path = os.path.join(dirpath, fn)
        if not os.path.isfile(path): continue
        key = os.path.splitext(fn)[0].upper()
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None: continue
        _, img_bin = cv.threshold(img, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        lib[key] = img_bin
    return lib

def match_best(roi_bin, templates):
    best_name, best_score = None, -1.0
    if not templates: return None, -1.0
    for name, tmpl in templates.items():
        t = cv.resize(tmpl, (roi_bin.shape[1], roi_bin.shape[0]))
        res = cv.matchTemplate(roi_bin, t, cv.TM_CCOEFF_NORMED)
        score = float(res.max())
        if score > best_score:
            best_score, best_name = score, name
    return best_name, best_score

def orb_match_score(roi_gray, tmpl_gray):
    orb = cv.ORB_create(nfeatures=300, fastThreshold=7, edgeThreshold=15, patchSize=31)
    kp1, des1 = orb.detectAndCompute(roi_gray, None)
    kp2, des2 = orb.detectAndCompute(tmpl_gray, None)
    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4: return 0.0
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    return min(1.0, len(good) / (len(kp1) + 1e-6))

def recognize_card_from_rois(card_bgr, rank_tpls, suit_tpls):
    # Retrieve config
    rank_thr = float(CFG["matching"]["rank_ncc_min"])
    suit_thr = float(CFG["matching"]["suit_ncc_min"])
    orb_min = float(CFG["matching"]["orb_min"])
    tl, br = CFG["roi"]["tl"], CFG["roi"]["br"]
    rank_split = float(CFG["roi"]["rank_split"])
    expand_px = int(CFG["roi"]["expand_px"])

    best_rank = best_suit = None
    best_conf, best_frac = 0.0, None

    for frac in (tl, br):
        roi = roi_from_frac(card_bgr, frac, expand_px=expand_px)
        roi_bin = prep_roi_bin(roi)
        h, w = roi_bin.shape[:2]
        split_y = max(1, min(h - 1, int(rank_split * h)))
        rank_bin, suit_bin = roi_bin[0:split_y, :], roi_bin[split_y:, :]

        rank, rs = match_best(rank_bin, rank_tpls)
        suit, ss = match_best(suit_bin, suit_tpls)

        # Suit color sanity check
        is_red = is_red_region(roi)
        if is_red and suit in ("S", "C"):
            # Force check red suits
            filtered = {k: v for k, v in suit_tpls.items() if k in {"H", "D"}}
            s2, ss2 = match_best(suit_bin, filtered)
            if ss2 > ss: suit, ss = s2, ss2
        elif (not is_red) and suit in ("H", "D"):
            # Force check black suits
            filtered = {k: v for k, v in suit_tpls.items() if k in {"S", "C"}}
            s2, ss2 = match_best(suit_bin, filtered)
            if ss2 > ss: suit, ss = s2, ss2

        # ORB Fallbacks
        if rs < rank_thr and rank_tpls:
            best_r, best_s = None, -1.0
            for k, t in rank_tpls.items():
                t_resized = cv.resize(t, (rank_bin.shape[1], rank_bin.shape[0]))
                s = orb_match_score(rank_bin, t_resized)
                if s > best_s: best_s, best_r = s, k
            if best_s > orb_min: rank, rs = best_r, max(rank_thr, rank_thr - 0.05 + 0.25 * best_s)

        if ss < suit_thr and suit_tpls:
            best_u, best_s = None, -1.0
            for k, t in suit_tpls.items():
                t_resized = cv.resize(t, (suit_bin.shape[1], suit_bin.shape[0]))
                s = orb_match_score(suit_bin, t_resized)
                if s > best_s: best_s, best_u = s, k
            if best_s > orb_min: suit, ss = best_u, max(suit_thr, suit_thr - 0.05 + 0.25 * best_s)

        if rank and suit and rs > 0.55 and ss > 0.60:
            conf = 0.5 * rs + 0.5 * ss
            if conf > best_conf:
                best_conf, best_rank, best_suit, best_frac = conf, rank, suit, frac

    return best_rank, best_suit, best_conf, best_frac

def recognize_card_upright(card_bgr, rank_tpls, suit_tpls):
    best_rank = best_suit = None
    best_conf, best_angle, best_frac = 0.0, 0, None
    rotations = [(0, None), (90, cv.ROTATE_90_CLOCKWISE), (180, cv.ROTATE_180), (270, cv.ROTATE_90_COUNTERCLOCKWISE)]

    for angle, code in rotations:
        rotated = card_bgr if code is None else cv.rotate(card_bgr, code)
        rank, suit, conf, used_frac = recognize_card_from_rois(rotated, rank_tpls, suit_tpls)
        if rank and suit and conf > best_conf:
            best_rank, best_suit, best_conf, best_angle, best_frac = rank, suit, conf, angle, used_frac
    return best_rank, best_suit, best_conf, best_angle, best_frac

def nms_detections(dets, iou_thr=None):
    if not dets: return []
    if iou_thr is None: iou_thr = float(CFG["matching"]["nms_iou"])
    boxes = [quad_to_bbox(d["quad"]) for d in dets]
    scores = [d["conf"] for d in dets]
    idxs = np.argsort(scores)[::-1].tolist()
    keep = []
    while idxs:
        i = idxs[0]
        keep.append(i)
        rest = []
        for j in idxs[1:]:
            if iou_bboxes(boxes[i], boxes[j]) < iou_thr:
                rest.append(j)
        idxs = rest
    return [dets[i] for i in keep]
