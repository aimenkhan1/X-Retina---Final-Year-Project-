# ===== X-Retina bootstrap: paste as the FIRST cell of every notebook =====
# Edit the 4 values below once. Needs Colab Secrets: GITHUB_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY
GITHUB_USER = "YOUR_GITHUB_USERNAME"   # owner of the repo
REPO_NAME   = "x-retina"
BRANCH      = "main"                   # or your own branch, e.g. "aiman/data"
GIT_NAME    = "Your Name"              # used for commits made from Colab
GIT_EMAIL   = "you@example.com"

import os, sys, json, subprocess
from google.colab import drive, userdata

# 1) Drive + shared folder
drive.mount("/content/drive")
DRIVE_ROOT = "/content/drive/MyDrive/XRetina_Shared"
assert os.path.isdir(DRIVE_ROOT), (
    f"Shared folder not found at {DRIVE_ROOT}. Add a shortcut named 'XRetina_Shared' to My Drive.")
for _sub in ("processed", "splits", "checkpoints", "outputs"):
    os.makedirs(f"{DRIVE_ROOT}/{_sub}", exist_ok=True)
os.environ["XRETINA_DRIVE_ROOT"] = DRIVE_ROOT
os.makedirs("/content/data/raw", exist_ok=True)
os.environ["XRETINA_RAW_DIR"] = "/content/data/raw"

# 2) Clone or update the repo (token is hidden from error messages)
_token = userdata.get("GITHUB_TOKEN")
REPO_DIR = f"/content/{REPO_NAME}"
_url = f"https://{_token}@github.com/{GITHUB_USER}/{REPO_NAME}.git"

def _run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:4])} failed:\n{r.stderr.replace(_token, '***')}")
    return r.stdout.strip()

if not os.path.isdir(REPO_DIR):
    _run(["git", "clone", _url, REPO_DIR])
_run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", _url])
_run(["git", "-C", REPO_DIR, "fetch", "origin"])
try:
    _run(["git", "-C", REPO_DIR, "checkout", BRANCH])
except RuntimeError:
    _run(["git", "-C", REPO_DIR, "checkout", "-b", BRANCH])  # first time on a new branch
try:
    _run(["git", "-C", REPO_DIR, "pull", "origin", BRANCH])
except RuntimeError as e:
    print("Pull skipped (new local branch or local changes):", str(e).splitlines()[-1])
_run(["git", "-C", REPO_DIR, "config", "user.name", GIT_NAME])
_run(["git", "-C", REPO_DIR, "config", "user.email", GIT_EMAIL])

# 3) Requirements
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{REPO_DIR}/requirements.txt"], check=True)

# 4) Kaggle credentials (optional per notebook)
try:
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    _kp = os.path.expanduser("~/.kaggle/kaggle.json")
    with open(_kp, "w") as f:
        json.dump({"username": userdata.get("KAGGLE_USERNAME"), "key": userdata.get("KAGGLE_KEY")}, f)
    os.chmod(_kp, 0o600)
except Exception as e:
    print("Kaggle secrets not set (only needed for dataset download):", e)

# 5) Make `import src...` work
os.chdir(REPO_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

def git_push(message):
    """Commit and push your work from Colab. Refuses to push to main (use a PR)."""
    if BRANCH == "main":
        raise RuntimeError("Do not push to main. Set BRANCH to your own branch (e.g. 'aiman/data') and re-run the bootstrap.")
    _run(["git", "-C", REPO_DIR, "add", "-A"])
    print(_run(["git", "-C", REPO_DIR, "commit", "-m", message]))
    print(_run(["git", "-C", REPO_DIR, "push", "-u", "origin", BRANCH]))

from src.config import get_paths
from src.utils import seed_everything, get_device
PATHS = get_paths()
print("Repo:", REPO_DIR, "| Branch:", BRANCH, "| Device:", get_device())
print({k: str(v) for k, v in PATHS.items()})
