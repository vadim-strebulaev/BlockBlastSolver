import argparse
import os
from typing import Optional, Tuple, List

import cv2
import numpy as np

import main


DEFAULT_DEVICE_ID = "m7caea6dfqpvmrss"
DEFAULT_CROP: Tuple[int, int, int, int] = (50, 1100, 650, 1350)


def detect_figure_cells(
    img: np.ndarray,
    crop: Tuple[int, int, int, int],
    step: int = 36,
    upper_black: int = 160,
    kernel_size: int = 3,
    min_area: int = 80,
    min_side: int = 12,
    max_side: int = 80,
    ratio_min: float = 0.70,
    ratio_max: float = 1.35,
):
    """Detect figure cubes by contour shape analysis.

    Returns a list of (x, y, w, h, slot) in full-image coordinates.
    """
    x1, y1, x2, y2 = crop
    crop_img = img[y1:y2, x1:x2]

    lower_black_np = np.array([0, 0, 0])
    upper_black_np = np.array([upper_black, upper_black, upper_black])
    mask = cv2.inRange(crop_img, lower_black_np, upper_black_np)
    inverted = cv2.bitwise_not(mask)

    kernel_size = max(1, int(kernel_size))
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    proc = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # visualize all candidate contours
    contour_candidates = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_candidates, contours, -1, (80, 80, 255), 1)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_side or h < min_side:
            continue

        ratio = w / float(h)
        if ratio < ratio_min or ratio > ratio_max:
            continue

        if w > max_side or h > max_side:
            continue

        cx = x + w // 2
        if 0 <= cx < 200:
            slot = 0
        elif 200 <= cx < 400:
            slot = 1
        else:
            slot = 2

        boxes.append((x1 + x, y1 + y, w, h, slot))

    # Deduplicate near-identical boxes by snapped center.
    uniq = {}
    for bx, by, bw, bh, slot in boxes:
        cx = int(round((bx + bw / 2) / step))
        cy = int(round((by + bh / 2) / step))
        key = (cx, cy, slot)
        if key not in uniq or (bw * bh) > (uniq[key][2] * uniq[key][3]):
            uniq[key] = (bx, by, bw, bh, slot)

    cells = sorted(list(uniq.values()), key=lambda t: (t[4], t[1], t[0]))

    debug = {
        "crop": crop_img,
        "mask": mask,
        "inverted": inverted,
        "morph": proc,
        "contour_candidates": contour_candidates,
    }

    return cells, debug


def draw_overlay(
    img: np.ndarray,
    cells: List[tuple],
    crop: Tuple[int, int, int, int],
    step: int = 36,
) -> np.ndarray:
    x1, y1, x2, y2 = crop
    out = img.copy()

    # Draw crop bounds
    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 180, 0), 2)

    # Draw slot separators
    cv2.line(out, (x1 + 200, y1), (x1 + 200, y2), (120, 120, 120), 1)
    cv2.line(out, (x1 + 400, y1), (x1 + 400, y2), (120, 120, 120), 1)

    slot_colors = {
        0: (0, 255, 0),
        1: (0, 200, 255),
        2: (255, 0, 255),
    }

    # Draw each detected contour bounding box
    for x_left, y_top, w, h, slot in cells:
        x_right = x_left + w
        y_bottom = y_top + h

        color = slot_colors.get(slot, (255, 255, 255))
        cv2.rectangle(out, (x_left, y_top), (x_right, y_bottom), color, 2)

    # Labels
    cv2.putText(
        out,
        "Slot 1",
        (x1 + 20, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        slot_colors[0],
        2,
    )
    cv2.putText(
        out,
        "Slot 2",
        (x1 + 220, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        slot_colors[1],
        2,
    )
    cv2.putText(
        out,
        "Slot 3",
        (x1 + 420, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        slot_colors[2],
        2,
    )

    return out


def run(device_id: Optional[str], output: str, crop: Tuple[int, int, int, int]):
    pass


def _gray_to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img.copy()


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def _make_debug_collage(
    stages: List[np.ndarray], labels: List[str], tile_w: int = 420, tile_h: int = 220
) -> np.ndarray:
    labeled = []
    for img, label in zip(stages, labels):
        bgr = _gray_to_bgr(img)
        bgr = cv2.resize(bgr, (tile_w, tile_h), interpolation=cv2.INTER_NEAREST)
        labeled.append(_label(bgr, label))

    # 3 columns grid
    rows = []
    cols = 3
    for i in range(0, len(labeled), cols):
        row = labeled[i : i + cols]
        while len(row) < cols:
            row.append(np.zeros_like(labeled[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def run(
    device_id: Optional[str],
    output: str,
    crop: Tuple[int, int, int, int],
    step: int,
    upper_black: int,
    kernel_size: int,
    min_area: int,
    min_side: int,
    max_side: int,
    ratio_min: float,
    ratio_max: float,
    show: bool,
    debug_output: str,
):
    main.take_screenshot(filename="screen.png", device_id=device_id)
    img = main.read_image("screen.png")

    cells, debug = detect_figure_cells(
        img,
        crop=crop,
        step=step,
        upper_black=upper_black,
        kernel_size=kernel_size,
        min_area=min_area,
        min_side=min_side,
        max_side=max_side,
        ratio_min=ratio_min,
        ratio_max=ratio_max,
    )
    out = draw_overlay(img, cells, crop=crop, step=step)

    if os.path.exists(output):
        os.remove(output)

    ok = cv2.imwrite(output, out)
    if not ok:
        raise RuntimeError(f"Не удалось сохранить overlay в '{output}'")

    stages = [
        debug["crop"],
        debug["mask"],
        debug["inverted"],
        debug["morph"],
        debug["contour_candidates"],
        out,
    ]
    labels = [
        "1) crop",
        f"2) mask upper_black={upper_black}",
        "3) inverted",
        f"4) morphology k={kernel_size}",
        "5) contours candidates",
        "6) final overlay",
    ]
    collage = _make_debug_collage(stages, labels)

    if os.path.exists(debug_output):
        os.remove(debug_output)
    ok2 = cv2.imwrite(debug_output, collage)
    if not ok2:
        raise RuntimeError(f"Не удалось сохранить debug collage в '{debug_output}'")

    if show:
        try:
            cv2.imshow("Figures detection pipeline", collage)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            print(
                "Warning: cv2.imshow недоступен в текущей среде. Смотри файл debug_output."
            )

    print(f"Saved overlay: {output}")
    print(f"Saved debug stages: {debug_output}")
    print(f"Detected cubes: {len(cells)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture screenshot and draw figure-cube squares"
    )
    parser.add_argument("--device-id", default=DEFAULT_DEVICE_ID)
    parser.add_argument("--output", default="figures_overlay.png")
    parser.add_argument("--debug-output", default="figures_pipeline.png")
    parser.add_argument(
        "--show",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Показать окно со всеми этапами (по умолчанию включено, отключить: --no-show)",
    )
    parser.add_argument("--step", type=int, default=36)
    parser.add_argument("--upper-black", type=int, default=150)
    parser.add_argument("--kernel-size", type=int, default=6)
    parser.add_argument("--min-area", type=int, default=80)
    parser.add_argument("--min-side", type=int, default=12)
    parser.add_argument("--max-side", type=int, default=80)
    parser.add_argument("--ratio-min", type=float, default=0.70)
    parser.add_argument("--ratio-max", type=float, default=1.35)
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        default=list(DEFAULT_CROP),
        metavar=("X1", "Y1", "X2", "Y2"),
    )
    args = parser.parse_args()

    run(
        device_id=args.device_id,
        output=args.output,
        crop=tuple(args.crop),
        step=args.step,
        upper_black=args.upper_black,
        kernel_size=args.kernel_size,
        min_area=args.min_area,
        min_side=args.min_side,
        max_side=args.max_side,
        ratio_min=args.ratio_min,
        ratio_max=args.ratio_max,
        show=args.show,
        debug_output=args.debug_output,
    )
