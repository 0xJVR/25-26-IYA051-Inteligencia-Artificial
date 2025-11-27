import numpy as np

def order_quad(pts):
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = np.argmin(s)
    br = np.argmax(s)
    tr = np.argmin(diff)
    bl = np.argmax(diff)
    return np.array([pts[tl], pts[tr], pts[br], pts[bl]], dtype=np.float32)

def quad_to_bbox(quad):
    xs = quad[:, 0]
    ys = quad[:, 1]
    return np.array([xs.min(), ys.min(), xs.max(), ys.max()], dtype=np.float32)

def center_of_quad(q):
    return q.mean(axis=0)

def iou_bboxes(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / (a1 + a2 - inter + 1e-9)
