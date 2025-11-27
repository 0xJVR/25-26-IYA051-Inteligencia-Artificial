import cv2 as cv
import numpy as np
from config import CFG, get_card_dims
from utils import order_quad, quad_to_bbox, iou_bboxes

def non_green_mask(bgr):
    """Segment non-table regions + bright whites."""
    hsv = cv.cvtColor(bgr, cv.COLOR_BGR2HSV)
    lo = tuple(int(x) for x in CFG["segmentation"]["green_hsv_low"])
    hi = tuple(int(x) for x in CFG["segmentation"]["green_hsv_high"])
    green = cv.inRange(hsv, lo, hi)
    non_green = cv.bitwise_not(green)

    v = hsv[:, :, 2]
    white_v_min = int(CFG["segmentation"]["white_v_min"])
    whites = cv.inRange(v, white_v_min, 255)

    fg = cv.bitwise_or(non_green, whites)
    ksz = int(CFG["segmentation"]["morph_kernel"])
    k = cv.getStructuringElement(cv.MORPH_ELLIPSE, (ksz, ksz))
    fg = cv.morphologyEx(fg, cv.MORPH_OPEN, k, iterations=int(CFG["segmentation"]["open_iters"]))
    fg = cv.morphologyEx(fg, cv.MORPH_CLOSE, k, iterations=int(CFG["segmentation"]["close_iters"]))
    return fg

def watershed_segments(bgr, fg_mask):
    sure_bg = cv.dilate(fg_mask, np.ones((7, 7), np.uint8), iterations=3)
    dist = cv.distanceTransform(fg_mask, cv.DIST_L2, 5)
    thr = float(CFG["proposals"]["watershed_distance_rel"])
    _, sure_fg = cv.threshold(dist, thr * dist.max(), 255, 0)
    sure_fg = sure_fg.astype(np.uint8)
    unknown = cv.subtract(sure_bg, sure_fg)

    _, markers = cv.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv.watershed(bgr.copy(), markers)
    return markers

def find_card_quads(bgr, fg_mask):
    fg = cv.bitwise_and(bgr, bgr, mask=fg_mask)
    gray = cv.cvtColor(fg, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)
    edges = cv.Canny(gray, 50, 120)
    edges = cv.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    quads = []
    min_area = float(CFG["proposals"]["min_contour_area"])
    min_w = float(CFG["proposals"]["min_w"])
    min_h = float(CFG["proposals"]["min_h"])
    ar_min, ar_max = CFG["proposals"]["card_ar_range"]

    for c in contours:
        if cv.contourArea(c) < min_area: continue
        rect = cv.minAreaRect(c)
        (cx, cy), (w, h), ang = rect
        W, H = sorted((w, h))
        if W < min_w or H < min_h: continue
        ar = H / (W + 1e-6)
        if ar_min <= ar <= ar_max:
            box = cv.boxPoints(rect).astype(np.float32)
            quads.append(order_quad(box))
    return quads

def quads_from_watershed(markers):
    quads = []
    min_area = float(CFG["proposals"]["min_contour_area"])
    for sid in [i for i in np.unique(markers) if i > 1]:
        m = (markers == sid).astype(np.uint8) * 255
        contours, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not contours: continue
        c = max(contours, key=cv.contourArea)
        if cv.contourArea(c) < min_area: continue
        rect = cv.minAreaRect(c)
        box = cv.boxPoints(rect).astype(np.float32)
        quads.append(order_quad(box))
    return quads

def merge_quads(quads, iou_thr=None):
    if iou_thr is None: iou_thr = float(CFG["matching"]["nms_iou"])
    if not quads: return []
    boxes = [quad_to_bbox(q) for q in quads]
    idxs = list(range(len(quads)))
    keep = []
    while idxs:
        i = idxs[0]
        keep.append(i)
        rest = []
        for j in idxs[1:]:
            if iou_bboxes(boxes[i], boxes[j]) < iou_thr:
                rest.append(j)
        idxs = rest
    return [quads[i] for i in keep]

def warp_card(bgr, quad):
    CAN_W, CAN_H = get_card_dims()
    src = order_quad(quad).astype(np.float32)
    dst = np.array([[0, 0], [CAN_W - 1, 0], [CAN_W - 1, CAN_H - 1], [0, CAN_H - 1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(src, dst)
    return cv.warpPerspective(bgr, M, (CAN_W, CAN_H))
