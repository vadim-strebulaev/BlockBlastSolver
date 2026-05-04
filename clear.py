from __future__ import annotations

import argparse
import subprocess

from main import DEFAULT_DEVICE_ID


TAPS = [
    (642, 136),
    (380, 1200),
    (620, 1560),
]


def adb_tap(device_id: str, x: int, y: int) -> None:
    cmd = ["adb", "-s", device_id, "shell", "input", "tap", str(x), str(y)]
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Тапы очистки перед ходом")
    parser.add_argument("--device", default=DEFAULT_DEVICE_ID, help="ADB device id")
    args = parser.parse_args()

    for x, y in TAPS:
        print(f"tap {x} {y}")
        adb_tap(args.device, x, y)


if __name__ == "__main__":
    main()
