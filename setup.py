"""Setuptools build hook that embeds the source commit in wheel artifacts."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

from setuptools import setup
from setuptools.command.build_py import build_py


class BuildPyWithCommit(build_py):
    def run(self):
        super().run()
        commit = os.environ.get("RR_VFIQA_BUILD_COMMIT")
        dirty = None
        if not commit:
            try:
                result = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=Path(__file__).resolve().parent,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                commit = result.stdout.strip()
                status = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=Path(__file__).resolve().parent,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                dirty = bool(status.stdout.strip())
            except (OSError, subprocess.SubprocessError):
                commit = "unknown"
        target = Path(self.build_lib) / "rr_vfiqa" / "_build_info.py"
        target.write_text(
            '"""Generated build-time metadata."""\n\n'
            f"BUILD_COMMIT = {commit!r}\n"
            f"BUILD_DIRTY = {dirty!r}\n",
            encoding="utf-8",
        )


setup(cmdclass={"build_py": BuildPyWithCommit})
