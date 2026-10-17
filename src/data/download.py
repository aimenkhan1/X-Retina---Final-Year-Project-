"""Kaggle download and inspection helpers (Stage 1: data collection).

Uses the `kaggle` command line tool (never `import kaggle`, which tries to
authenticate at import time). Auth is set up by the Colab bootstrap cell:
KAGGLE_API_TOKEN (new token, preferred) or ~/.kaggle/kaggle.json (legacy).
"""
import os
import re
import shutil
import subprocess
import zipfile
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
    """Free disk space in GB at `path` (falls back to the nearest existing parent folder)."""
    path = Path(path)
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1024 ** 3


def _to_gb(value: float, unit: str) -> float:
    return value * {"KB": 1 / 1024 ** 2, "MB": 1 / 1024, "GB": 1, "TB": 1024}[unit.upper()]


def dataset_size(slug: str):
    """Return (listing_line, size_in_GiB) from `kaggle datasets list`, or (None, None) if not found.

    Handles both listing styles: a size with a unit ("8GB") and a raw byte count.
    """
    owner, name = slug.split("/")

    for extra in (["--user", owner, "-s", name], ["-s", name]):
        r = subprocess.run([_kaggle_bin(), "datasets", "list"] + extra,
                           capture_output=True, text=True)
        if r.returncode != 0:
            continue

        for line in r.stdout.splitlines():
            if slug.lower() not in line.lower():
                continue

            # 1) a size with an explicit unit
            match = re.search(r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)\b", line, re.I)
            if match:
                return line.strip(), _to_gb(float(match.group(1)), match.group(2))

            # 2) otherwise a long integer = size in bytes
            for part in line.split():
                if part.isdigit() and len(part) >= 9:
                    return line.strip(), int(part) / (1024 ** 3)

            return line.strip(), None

    return None, None


def _make_printer():
    """Print download progress only when it crosses a new 10% step (instead of hundreds of lines)."""
    pat = re.compile(r"(\d+)%\|")
    state = {"last": -1}

    def emit(line: str) -> None:
        line = line.strip()
        if not line:
            return
        m = pat.search(line)
        if m:
            step = int(m.group(1)) // 10
            if step != state["last"]:
                state["last"] = step
                print(line[-120:])
        else:
            print(line[-160:])
    return emit


def _run_download(cmd, label: str) -> None:
    emit = _make_printer()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    tail = []
    for line in proc.stdout:
        tail = (tail + [line.strip()])[-5:]
        emit(line)
    if proc.wait() != 0:
        msg = "\n".join(t for t in tail if t)
        hint = ""
        if "403" in msg or "Forbidden" in msg:
            hint = "\nHint: open the dataset page on kaggle.com while logged in and accept its terms, then retry."
        if "404" in msg or "Not Found" in msg:
            hint = "\nHint: check the dataset slug / file name in configs/datasets.yaml."
        raise KaggleError(f"Download failed for {label}:\n{msg}{hint}")


def _extract_zips(dest: Path) -> None:
    """Unzip every .zip under dest in place, then delete the zip to save disk."""
    for z in sorted(dest.rglob("*.zip")):
        if zipfile.is_zipfile(z):
            print(f"Unzipping {z.name} ...")
            with zipfile.ZipFile(z) as zf:
                zf.extractall(z.parent)
            z.unlink()


def download_dataset(slug: str, dest, force: bool = False, files=None) -> Path:
    """Download a Kaggle dataset into `dest` and unzip it.

    files: optional list of file names inside the dataset. Only those are downloaded
           (use for big multi-file datasets). Leave None to download everything.
    A completion marker makes re-runs skip finished downloads.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / MARKER).exists() and not force:
        print(f"[skip] {slug} already downloaded at {dest}")
        return dest
    print(f"Downloading {slug} -> {dest}  (free disk: {free_disk_gb(dest):.1f} GB)")
    if files:
        for f in files:
            print(f"- file: {f}")
            _run_download([_kaggle_bin(), "datasets", "download", "-d", slug, "-f", f, "-p", str(dest)],
                          f"{slug} / {f}")
        _extract_zips(dest)
    else:
        _run_download([_kaggle_bin(), "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], slug)
    (dest / MARKER).write_text("ok")
    print(f"Done. Free disk now: {free_disk_gb(dest):.1f} GB")
    return dest


def delete_raw(path) -> None:
    """Delete a raw dataset folder after it has been preprocessed (frees local disk; re-download is easy)."""
    path = Path(path)
    before = free_disk_gb(path)
    if path.exists():
        shutil.rmtree(path)
    print(f"Deleted {path}. Free disk: {before:.1f} -> {free_disk_gb(path):.1f} GB")


def describe_tree(root, max_depth: int = 2, max_dirs: int = 30) -> None:
    """Print folders (to max_depth) with file counts per extension."""
    root = Path(root)
    shown = 0
    for dirpath, dirnames, filenames in sorted(os.walk(root), key=lambda t: t[0]):
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