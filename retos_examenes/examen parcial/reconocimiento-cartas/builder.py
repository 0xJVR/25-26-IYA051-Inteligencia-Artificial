import os
import cv2 as cv
from config import CFG, RANK_DIR, SUIT_DIR
from core import non_green_mask, find_card_quads, warp_card
from utils import quad_to_bbox
from recognition import roi_from_frac, prep_roi_bin
from overlay import prompt_overlay

def capture_one_card_roi(cap, prompt, save_rank=None, save_suit=None):
    expand_px = int(CFG["roi"]["expand_px"])
    rank_split = float(CFG["roi"]["rank_split"])

    while True:
        ok, frame = cap.read()
        if not ok: continue
        show = prompt_overlay(frame, prompt + "  [SPACE=Capture / ESC=Cancel]")
        cv.imshow("cards", show)
        k = cv.waitKey(1) & 0xFF

        if k == 27: return False
        if k == 32:
            bgr = frame.copy()
            mask = non_green_mask(bgr)
            quads = find_card_quads(bgr, mask)
            if not quads:
                cv.imshow("cards", prompt_overlay(bgr, "No card found, try again"))
                cv.waitKey(700)
                continue

            # Pick largest quad
            quads = sorted(quads, key=lambda q: (quad_to_bbox(q)[2] - quad_to_bbox(q)[0]) * (quad_to_bbox(q)[3] - quad_to_bbox(q)[1]), reverse=True)
            card = warp_card(bgr, quads[0])

            # Extract corner
            pip_block = roi_from_frac(card, CFG["roi"]["tl"], expand_px=expand_px)
            pip_bin = prep_roi_bin(pip_block)
            h, w = pip_bin.shape[:2]
            split_y = max(1, min(h - 1, int(rank_split * h)))
            rank_bin = pip_bin[0:split_y, :]
            suit_bin = pip_bin[split_y:, :]

            if save_rank: cv.imwrite(os.path.join(RANK_DIR, f"{save_rank}.png"), rank_bin)
            if save_suit: cv.imwrite(os.path.join(SUIT_DIR, f"{save_suit}.png"), suit_bin)

            cv.imshow("cards", prompt_overlay(card, "Captured checked"))
            cv.waitKey(700)
            return True

def run_template_builder():
    cam_idx = int(CFG["camera_index"])
    cap = cv.VideoCapture(cam_idx)
    if not cap.isOpened():
        print(f"Camera {cam_idx} not available.")
        return

    print("\n=== Template Builder ===")
    for r in ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]:
        if not capture_one_card_roi(cap, f"Show rank '{r}' (any suit).", save_rank=r): break
    for s, label in zip(["S", "H", "D", "C"], ["Spade (black)", "Heart (red)", "Diamond (red)", "Club (black)"]):
        if not capture_one_card_roi(cap, f"Show any {label}.", save_suit=s): break

    cap.release()
    print("Templates saved to ./templates/.")
