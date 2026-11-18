"""
Repo-root entry point: forwards to src/main.py, so both forms work.

    python main.py --input <degraded.csv> --output estimated_trajectory.txt --meta metadata.json
    python src/main.py --input <degraded.csv> --output estimated_trajectory.txt --meta metadata.json
"""

import os
import runpy

if __name__ == "__main__":
    runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "main.py"),
                   run_name="__main__")
