"""Find label CSVs and image files inside a raw dataset folder, without assuming its layout."""
import re
from pathlib import Path

import pandas as pd

IMG_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
ID_CANDIDATES = ["id_code", "image", "image_id", "image_name", "filename", "file_name", "id"]
LABEL_CANDIDATES = ["diagnosis", "level", "label", "grade", "dr_grade", "class"]


def index_images(root):
    """Map image stem -> path for every image under root. Returns (index, duplicate_stems)."""
    index, dups = {}, []
    for p in Path(root).rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            if p.stem in index:
                dups.append(p.stem)
            else:
                index[p.stem] = p
    return index, dups


def find_label_csvs(root, id_candidates=ID_CANDIDATES, label_candidates=LABEL_CANDIDATES):
    """Inspect every CSV under root. Returns a list of dicts (usable ones have id_col and label_col)."""
    found = []
    for p in sorted(Path(root).rglob("*.csv")):
        try:
            head = pd.read_csv(p, nrows=5)
        except Exception as e:  # unreadable csv
            found.append({"path": p, "usable": False, "reason": f"unreadable: {e}"})
            continue
        lower = {c.lower().strip(): c for c in head.columns}
        id_col = next((lower[c] for c in id_candidates if c in lower), None)
        label_col = next((lower[c] for c in label_candidates if c in lower), None)
        ok = id_col is not None and label_col is not None
        found.append({"path": p, "usable": ok, "id_col": id_col, "label_col": label_col,
                      "columns": list(head.columns),
                      "reason": "" if ok else "no id/label column match"})
    return found


def load_labels(root, verbose: bool = True):
    """Combine all usable label CSVs under root and attach the matching image path.

    Returns (df, report). df columns: image_id, label, source_csv, image_path.
    """
    imgs, dups = index_images(root)
    csvs = find_label_csvs(root)
    frames, skipped = [], []
    for info in csvs:
        if not info["usable"]:
            skipped.append((info["path"].name, info["reason"]))
            continue
        df = pd.read_csv(info["path"], usecols=[info["id_col"], info["label_col"]])
        df.columns = ["image_id", "label"]
        df["image_id"] = df["image_id"].astype(str).map(lambda s: Path(s).stem)
        df["label"] = pd.to_numeric(df["label"], errors="coerce")
        df["source_csv"] = info["path"].name
        frames.append(df)
    if not frames:
        raise ValueError(f"No usable label CSV under {root}. CSVs seen: {[c['path'].name for c in csvs]}")
    df = pd.concat(frames, ignore_index=True)

    n_raw = len(df)
    bad_label = df["label"].isna() | ~df["label"].isin([0, 1, 2, 3, 4])
    df = df[~bad_label].copy()
    df["label"] = df["label"].astype(int)

    conflicts = (df.groupby("image_id")["label"].nunique() > 1).sum()
    df = df.drop_duplicates("image_id", keep="first")
    df["image_path"] = df["image_id"].map(imgs)
    missing_img = int(df["image_path"].isna().sum())
    df = df[df["image_path"].notna()].reset_index(drop=True)

    report = {"rows_in_csvs": n_raw, "dropped_bad_label": int(bad_label.sum()),
              "label_conflicts": int(conflicts), "labeled_without_image": missing_img,
              "images_on_disk": len(imgs), "duplicate_image_stems": len(dups),
              "usable_csvs": [c["path"].name for c in csvs if c["usable"]], "skipped_csvs": skipped,
              "n_labeled": len(df)}
    if verbose:
        print("Label report:")
        for k, v in report.items():
            print(f"  {k}: {v}")
    return df, report


def patient_id_from_name(image_id: str):
    """EyePACS names look like '10_left' / '10_right' -> patient '10'. Returns None otherwise."""
    m = re.match(r"^(\d+)_(left|right)", str(image_id), re.I)
    return m.group(1) if m else None


def class_table(df, class_names=None) -> pd.DataFrame:
    """Counts and percentages per DR grade (works on any df with a 'label' column)."""
    names = class_names or ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
    counts = df["label"].value_counts().reindex(range(5), fill_value=0)
    out = pd.DataFrame({"grade": range(5), "name": names, "count": counts.values})
    out["percent"] = (out["count"] / max(out["count"].sum(), 1) * 100).round(1)
    return out
