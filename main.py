"""phone_capture.py

Утилиты для снятия скриншота с Android-устройства через ADB и преобразования
изображения в numpy-матрицы (поле 8x8, матрицы фигур и т.п.).

Функции:
 - take_screenshot(filename, device_id)
 - read_image(filename)
 - image_to_board_matrix(img, background_bgr, threshold)
 - image_to_figures_matrices(img, crop)
 - capture_all(device_id)

Этот файл объединяет логику из `calibrate.py`, `checkField.py`, `FindFigures.py` и
`GetScore.py` в одну утилиту, которую можно импортировать или запускать напрямую.
"""

import subprocess
from typing import List, Tuple, Dict, Optional

import cv2
import numpy as np


DEFAULT_DEVICE_ID = "m7caea6dfqpvmrss"


def take_screenshot(
    filename: str = "screen.png", device_id: Optional[str] = None
) -> None:
    """Снимает скриншот с телефона и сохраняет в файл.

    Использует `adb exec-out screencap -p` и записывает stdout в файл.
    """
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]
    cmd += ["exec-out", "screencap", "-p"]

    with open(filename, "wb") as f:
        subprocess.run(cmd, stdout=f, check=True)


def read_image(filename: str = "screen.png") -> np.ndarray:
    img = cv2.imread(filename)
    if img is None:
        raise RuntimeError(f"Не удалось загрузить изображение '{filename}'")
    return img


def image_to_board_matrix(
    img: np.ndarray,
    background_bgr: Tuple[int, int, int] = (69, 35, 29),
    threshold: int = 30,
) -> List[List[int]]:
    """Преобразует переданное изображение в матрицу поля 8x8 (0/1).

    Логика повторяет `checkField.analyze_board`: берём пиксели в центрах клеток
    и сравниваем с цветом фона по евклидовой норме.
    """
    h, w = img.shape[:2]
    # Используем координаты как в оригинальном проекте (при необходимости изменить)
    result: List[List[int]] = []
    for row in range(8):
        row_data: List[int] = []
        for col in range(8):
            x = 80 + col * 80
            y = 450 + row * 80
            # Защита от выхода за границы
            if y >= h or x >= w:
                row_data.append(0)
                continue
            bgr = img[y, x].tolist()
            value = (
                0
                if np.linalg.norm(np.array(bgr) - np.array(background_bgr)) < threshold
                else 1
            )
            row_data.append(int(value))
        result.append(row_data)
    return result


def image_to_figures_matrices(
    img: np.ndarray, crop: Tuple[int, int, int, int] = (50, 1100, 650, 1350)
) -> List[List[List[int]]]:
    """Находит фигуры в обрезке экрана и возвращает список матриц (по умолчанию 3).

    Параметр crop — (x1, y1, x2, y2) в пикселях.
    Логика основана на `FindFigures.py`.
    """
    x1, y1, x2, y2 = crop
    crop_img = img[y1:y2, x1:x2]

    # Параметры контурного алгоритма (как в draw_figures_overlay.py)
    step = 36
    upper_black = 150
    kernel_size = 6
    min_area = 80
    min_side = 12
    max_side = 80
    ratio_min = 0.70
    ratio_max = 1.35

    lower_black_np = np.array([0, 0, 0])
    upper_black_np = np.array([upper_black, upper_black, upper_black])
    mask = cv2.inRange(crop_img, lower_black_np, upper_black_np)
    inverted = cv2.bitwise_not(mask)

    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    proc = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(proc, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Собираем найденные "кубики" (bounding boxes)
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

        boxes.append((x, y, w, h, slot))

    # Dedupe по привязке к сетке
    uniq = {}
    for bx, by, bw, bh, slot in boxes:
        cx = int(round((bx + bw / 2) / step))
        cy = int(round((by + bh / 2) / step))
        key = (cx, cy, slot)
        if key not in uniq or (bw * bh) > (uniq[key][2] * uniq[key][3]):
            uniq[key] = (bx, by, bw, bh, slot)

    # Переводим найденные боксы в дискретные точки сетки по слотам
    slots_points = [set(), set(), set()]
    for bx, by, bw, bh, slot in uniq.values():
        gx = int(round((bx + bw / 2) / step))
        gy = int(round((by + bh / 2) / step))
        slots_points[slot].add((gx, gy))

    def points_to_compact_matrix(points_set):
        """Строит компактную матрицу фигуры из всех точек слота.

        Важно: используем все валидные точки, а не только largest component.
        Это предотвращает схлопывание фигуры в 1 клетку при небольших артефактах
        снаппинга/дискретизации.
        """
        if not points_set:
            return [[0]]

        xs_sorted = sorted({x for x, _ in points_set})
        ys_sorted = sorted({y for _, y in points_set})

        x_to_idx = {x: i for i, x in enumerate(xs_sorted)}
        y_to_idx = {y: i for i, y in enumerate(ys_sorted)}

        w = len(xs_sorted)
        h = len(ys_sorted)
        matrix = [[0] * w for _ in range(h)]
        for x, y in points_set:
            matrix[y_to_idx[y]][x_to_idx[x]] = 1

        return matrix

    def matrix_to_5x5_centered(matrix: List[List[int]]) -> List[List[int]]:
        """Преобразует матрицу фигуры в 5x5 с центровкой по правилу:
        если размер по оси чётный, центром считается левый/верхний из двух центральных.
        """
        out = [[0] * 5 for _ in range(5)]

        if not matrix or not matrix[0]:
            return out

        h = len(matrix)
        w = len(matrix[0])

        # Индекс "центра" фигуры по требуемому правилу:
        # odd: floor(size/2), even: left/top central => size/2 - 1
        center_y = (h - 1) // 2
        center_x = (w - 1) // 2

        target_cy = 2
        target_cx = 2

        shift_y = target_cy - center_y
        shift_x = target_cx - center_x

        for y in range(h):
            for x in range(w):
                if matrix[y][x] == 1:
                    ny = y + shift_y
                    nx = x + shift_x
                    if 0 <= ny < 5 and 0 <= nx < 5:
                        out[ny][nx] = 1

        return out

    result = []
    for slot in range(3):
        slot_points = slots_points[slot]
        if slot_points:
            result.append(matrix_to_5x5_centered(points_to_compact_matrix(slot_points)))
        else:
            result.append([[0] * 5 for _ in range(5)])

    return result


def capture_all(device_id: Optional[str] = DEFAULT_DEVICE_ID) -> Dict[str, object]:
    """Выполняет полный цикл: снимает, читает изображение и возвращает словарь:
    {
        'image': np.ndarray,
        'board': list[list[int]],
        'figures': list[list[list[int]]]
    }
    """
    filename = "screen.png"
    take_screenshot(filename=filename, device_id=device_id)
    img = read_image(filename)
    board = image_to_board_matrix(img)
    figures = image_to_figures_matrices(img)
    return {"image": img, "board": board, "figures": figures}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Capture screen and convert to arrays")
    p.add_argument("--device", help="ADB device id", default=DEFAULT_DEVICE_ID)
    args = p.parse_args()

    out = capture_all(device_id=args.device)
    print("Board (8x8):")
    for r in out["board"]:
        print(r)
    print("Figures:")
    for i, f in enumerate(out["figures"], 1):
        print(f"Figure {i}:")
        for row in f:
            print(row)
    # Сохраним уменьшенную превью-картинку для отладки
    cv2.imwrite("screen_preview.jpg", cv2.resize(out["image"], (800, 800)))
