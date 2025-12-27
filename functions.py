import cv2
import numpy as np
from typing import List, Tuple

LINE_COLOR = (0, 140, 255)
WHITE_COLOR = (255, 255, 255)

def position_data(lmlist: List[Tuple[int, int]]) -> List[Tuple[int, int]]:

    if len(lmlist) < 21:
        raise ValueError("landmark list must contain at least 21 points.")

    keys = [0, 4, 5, 8, 9, 12, 16, 20]
    return [lmlist[i] for i in keys]

def calculate_distance(p1:Tuple[int, int], p2:Tuple[int, int]) -> float:
    return np.linalg.norm(np.array(p1) - np.array(p2))

def draw_line(
    frame: np.ndarray,
    p1: Tuple[int, int],
    p2:Tuple[int, int],
    color: Tuple[int, int, int] = LINE_COLOR,
    thickness: int = 5
) -> np.ndarray:

    cv2.line(frame, p1, p2, color, thickness)
    cv2.line(frame, p1, p2, WHITE_COLOR, max(1, thickness // 2))
    return frame

def overlay_image(
    target_img: np.ndarray,
    frame: np.ndarray,
    x: int, y: int,
    size: Tuple[int, int] = None
) -> np.ndarray:

    if size:
        try:
            target_img = cv2.resize(target_img, size)
        except cv2.error as e:
            raise ValueError(f"error resizing the target image: {e}")

    if target_img.shape[-1] != 4:
        raise ValueError("Target image must have 4 channels (RGBA).")

    b, g, r, a = cv2.split(target_img)
    overlay_color = cv2.merge((b, g, r))
    mask = cv2.medianBlur(a, 5)
    h, w, _ = overlay_color.shape

    if y + h > frame.shape[0] or x + w > frame.shape[1]:
        raise ValueError("Overlay exceeds frame boundaries.")

    roi = frame[y:y + h, x:x + w]

    img1 = cv2.bitwise_and(roi, roi, mask = cv2.bitwise_not(mask))
    img2 = cv2.bitwise_and(overlay_color, overlay_color, mask = mask)

    frame[y:y + h, x:x + w] = cv2.add(img1, img2)

    return frame
