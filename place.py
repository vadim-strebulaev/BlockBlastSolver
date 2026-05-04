import subprocess


def place_figure(
    figure_id: int,
    col: int,
    row: int,
    honestX: int,
    honestY: int,
    device_id: str = "m7caea6dfqpvmrss",
):
    """
    Перемещает фигуру с указанным номером в клетку (row, col) на поле 8x8.

    :param figure_id: Номер фигуры (1, 2, 3)
    :param row: Номер строки (0–7)
    :param col: Номер столбца (0–7)
    :param device_id: Необязательный ID устройства ADB (если несколько устройств)
    """
    figures = {
        1: (142, 1216),
        2: (370, 1216),
        3: (574, 1216),
    }

    if figure_id not in figures:
        raise ValueError("Номер фигуры должен быть 1, 2 или 3.")
    if not (0 <= row <= 7 and 0 <= col <= 7):
        raise ValueError("Строка и столбец должны быть в пределах от 0 до 7.")

    start_x, start_y = figures[figure_id]

    # Вычисление координат конца (по формуле из запроса)
    x = 0.705 * (80 * col + 80 + 40 * honestX - start_x) + start_x
    y = 1650 - (0.755 * (80 * (7 - row) - 40 * honestY + 450 - 434) + 434) - 3 * row

    x, y = int(x), int(y)  # округляем до целых
    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]

    cmd += [
        "shell",
        "input",
        "swipe",
        str(start_x),
        str(start_y),
        str(x),
        str(y),
        "500",
    ]

    print(f"Свайп фигуры {figure_id} из ({start_x}, {start_y}) в ({x}, {y})")
    subprocess.run(cmd)


def place_figure_pixel(
    figure_id: int,
    target_x: int,
    target_y: int,
    device_id: str = "m7caea6dfqpvmrss",
):
    """Swipe a figure from its slot straight to target pixel coordinates (target_x,target_y).

    This function uses absolute pixel coordinates for the swipe end point instead of
    computing offsets by cell sizes. It's useful when you want pixel-accurate placement.
    """
    figures = {
        1: (142, 1216),
        2: (370, 1216),
        3: (574, 1216),
    }

    if figure_id not in figures:
        raise ValueError("Номер фигуры должен быть 1, 2 или 3.")

    start_x, start_y = figures[figure_id]
    x, y = int(target_x), int(target_y)

    cmd = ["adb"]
    if device_id:
        cmd += ["-s", device_id]

    cmd += [
        "shell",
        "input",
        "swipe",
        str(start_x),
        str(start_y),
        str(x),
        str(y),
        "1000",
    ]

    print(
        f"Свайп фигуры {figure_id} из ({start_x}, {start_y}) в ({x}, {y}) [pixel-target]"
    )
    subprocess.run(cmd)


if __name__ == "__main__":
    place_figure(3, 1, 1, 0, 1)
