from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from main import DEFAULT_DEVICE_ID


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Циклический запуск clear.py + solver.py"
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE_ID, help="ADB device id")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    clear_script = root / "clear.py"
    solver_script = root / "solver.py"

    env = os.environ.copy()
    env["BLOCK_DEVICE_ID"] = args.device

    iteration = 0
    print(f"Старт цикла для устройства: {args.device}")
    try:
        while True:
            iteration += 1
            print(f"\n=== Итерация {iteration} ===")

            subprocess.run(
                [sys.executable, str(clear_script), "--device", args.device],
                check=True,
                cwd=str(root),
                env=env,
            )

            time.sleep(0.2)

            subprocess.run(
                [sys.executable, str(solver_script)],
                check=True,
                cwd=str(root),
                env=env,
            )

            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nОстановлено пользователем")


if __name__ == "__main__":
    main()
