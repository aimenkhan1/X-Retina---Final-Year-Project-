"""Kaggle download and inspection helpers (Stage 1: data collection).

Uses the `kaggle` command line tool (never `import kaggle`, which tries to
authenticate at import time). Auth is set up by the Colab bootstrap cell:
KAGGLE_API_TOKEN (new token, preferred) or ~/.kaggle/kaggle.json (legacy).
"""
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

MARKER = ".download_complete"


class KaggleError(RuntimeError):
    pass


def _kaggle_bin() -> str:
    exe = shutil.which("kaggle")
    if exe is None:
        raise KaggleError("`kaggle` command not found. Run: pip install kaggle")
    return exe


def check_auth() -> None:
    """Fail early with a clear message if Kaggle credentials do not work."""
    r = subprocess.run([_kaggle_bin(), "datasets", "list", "-s", "aptos2019"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise KaggleError(
            "Kaggle authentication failed:\n" + (r.stderr or r.stdout).strip()[-600:] +
            "\n\nFix: (1) Kaggle > Settings > API > create a new token; (2) put it in Colab Secrets as "
            "KAGGLE_API_TOKEN (or legacy KAGGLE_USERNAME + KAGGLE_KEY); (3) turn on 'Notebook access'; "
            "(4) re-run the bootstrap cell.")
    print("Kaggle auth OK.")


def free_disk_gb(path="/content") -> float:
    return shutil.disk_usage(path).free / 1024 ** 3


def _to_gb(value: float, unit: str) -> float:
    return value * {"KB": 1 / 1024 ** 2, "MB": 1 / 1024, "GB": 1, "TB": 1024}[unit.upper()]



def dataset_size(slug: str):
    """Return (listing_line, size_in_gb) from Kaggle's dataset listing."""
    owner, name = slug.split("/")

    for extra in (["--user", owner, "-s", name], ["-s", name]):
        r = subprocess.run(
            [_kaggle_bin(), "datasets", "list"] + extra,
            capture_output=True,
            text=True
        )

        if r.returncode != 0:
            continue

        for line in r.stdout.splitlines():
            if slug.lower() in line.lower():
                # First, try sizes written with units, e.g. 32.9 GB
                m = re.search(
                    r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)\b",
                    line,
                    re.I
                )

                if m:
                    size_gb = _to_gb(float(m.group(1)), m.group(2))
                    return line.strip(), size_gb

                # Otherwise, detect a plain byte count in the listing
                parts = line.split()

                for part in parts:
                    if part.isdigit() and len(part) >= 9:
                        size_gb = int(part) / (1024 ** 3)
                        return line.strip(), size_gb

                return line.strip(), None

    return None, None


def download_dataset(slug: str, dest, force: bool = False) -> Path:
    """Download + unzip a Kaggle dataset into `dest`. Skips if a previous download finished."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / MARKER).exists() and not force:
        print(f"[skip] {slug} already downloaded at {dest}")
        return dest
    print(f"Downloading {slug} -> {dest}  (free disk: {free_disk_gb(dest):.1f} GB)")
    proc = subprocess.Popen([_kaggle_bin(), "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-5:]
        print(line[-160:])
    if proc.wait() != 0:
        msg = "\n".join(tail)
        hint = ""
        if "403" in msg or "Forbidden" in msg:
            hint = "\nHint: open the dataset page on kaggle.com while logged in and accept its terms, then retry."
        if "404" in msg or "Not Found" in msg:
            hint = "\nHint: check the dataset slug in configs/datasets.yaml."
        raise KaggleError(f"Download failed for {slug}:\n{msg}{hint}")
    (dest / MARKER).write_text("ok")
    print(f"Done. Free disk now: {free_disk_gb(dest):.1f} GB")
    return dest


def describe_tree(root, max_depth: int = 2, max_dirs: int = 30) -> None:
    """Print folders (to max_depth) with file counts per extension."""
    root = Path(root)
    shown = 0
    for dirpath, dirnames, filenames in sorted(
            ((p, d, f) for p, d, f in __import__("os").walk(root)), key=lambda t: t[0]):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            continue
        counts = Counter(Path(f).suffix.lower() or "(none)" for f in filenames)
        label = "." if depth == 0 else str(Path(dirpath).relative_to(root))
        print(f"{'  ' * depth}{label}/  {dict(counts) if counts else ''}")
        shown += 1
        if shown >= max_dirs:
            print("  ... (truncated)")
            break
