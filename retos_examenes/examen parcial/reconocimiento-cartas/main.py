import os
import cv2 as cv
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

# Local modules
from config import CFG, load_config, OUT_DIR, RANK_DIR, SUIT_DIR
from core import non_green_mask, watershed_segments, quads_from_watershed, find_card_quads, merge_quads, warp_card
from recognition import load_templates, recognize_card_upright, nms_detections, roi_from_frac
from overlay import draw_quad, draw_label, draw_bottom_overlay, center_of_quad
from builder import run_template_builder

DEBUG_CARD_ROI = False

def _recognize_one(bgr, quad, rank_tpls, suit_tpls):
    """Worker function for parallel recognition."""
    card = warp_card(bgr, quad)
    rank, suit, conf, angle, used_frac = recognize_card_upright(card, rank_tpls, suit_tpls)
    if rank is None or suit is None: return None
    return {
        "quad": quad, "rank": rank, "suit": suit, "conf": conf,
        "angle": angle, "corner_frac": used_frac
    }

def main():
    global DEBUG_CARD_ROI
    cam_idx = int(CFG["camera_index"])
    cap = cv.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Could not open camera {cam_idx}.")
        return

    rank_tpls = load_templates(RANK_DIR)
    suit_tpls = load_templates(SUIT_DIR)
    show_debug = True
    snap_id = 0

    print("Running... Press 'q' to quit, 't' for templates, 'd' for debug views.")

    while True:
        ok, frame = cap.read()
        if not ok: continue
        bgr = frame
        vis = bgr.copy()

        # 1. Segmentation & Proposals
        mask = non_green_mask(bgr)
        markers = watershed_segments(bgr, mask)
        quads = merge_quads(quads_from_watershed(markers) + find_card_quads(bgr, mask))

        # 2. Recognition
        dets = []
        PERF = CFG["performance"]
        if PERF.get("parallel_quads", True) and len(quads) > 1:
            workers = max(1, int(PERF.get("workers", 4)))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futs = [ex.submit(_recognize_one, bgr, q, rank_tpls, suit_tpls) for q in quads]
                for f in as_completed(futs):
                    if f.result(): dets.append(f.result())
        else:
            for q in quads:
                r = _recognize_one(bgr, q, rank_tpls, suit_tpls)
                if r: dets.append(r)

        dets = nms_detections(dets)

        # 3. Debug Visualization
        if DEBUG_CARD_ROI and dets:
            best_det = max(dets, key=lambda d: d["conf"])
            card_dbg = warp_card(bgr, best_det["quad"])
            angle = int(best_det.get("angle", 0))
            if angle == 90: card_dbg = cv.rotate(card_dbg, cv.ROTATE_90_CLOCKWISE)
            elif angle == 180: card_dbg = cv.rotate(card_dbg, cv.ROTATE_180)
            elif angle == 270: card_dbg = cv.rotate(card_dbg, cv.ROTATE_90_COUNTERCLOCKWISE)

            frac = best_det.get("corner_frac") or CFG["roi"]["tl"]
            roi_dbg = roi_from_frac(card_dbg, frac, expand_px=int(CFG["roi"]["expand_px"]))
            cv.imshow("debug_card", card_dbg)
            cv.imshow("debug_corner", roi_dbg)

        # 4. Drawing
        for d in dets:
            draw_quad(vis, d["quad"])
            cx, cy = center_of_quad(d["quad"])
            draw_label(vis, f'{d["rank"]}{d["suit"]} {d["conf"]:.2f}', (cx - 40, cy))

        vis = draw_bottom_overlay(vis, dets)

        if show_debug:
            dbg = vis.copy()
            ws = (markers == -1).astype(np.uint8) * 255
            dbg[ws > 0] = (0, 0, 255)
            cv.imshow("cards", dbg)
        else:
            cv.imshow("cards", vis)

        # 5. Controls
        k = cv.waitKey(1) & 0xFF
        if k in (27, ord('q'), ord('Q')): break
        elif k in (ord('g'), ord('G')): show_debug = not show_debug
        elif k in (ord('t'), ord('T')):
            cap.release()
            cv.destroyWindow("cards")
            run_template_builder()
            cap = cv.VideoCapture(cam_idx)
            rank_tpls = load_templates(RANK_DIR)
            suit_tpls = load_templates(SUIT_DIR)
        elif k in (ord('r'), ord('R')):
            load_config()
            print("Reloaded config.json.")
        elif k in (ord('s'), ord('S')):
            fn = os.path.join(OUT_DIR, f"snap_{snap_id:04d}.png")
            cv.imwrite(fn, vis)
            print(f"Saved {fn}")
            snap_id += 1
        elif k in (ord('z'), ord('Z')):
            DEBUG_CARD_ROI = not DEBUG_CARD_ROI
            if not DEBUG_CARD_ROI: cv.destroyAllWindows()

    cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
