import numpy as np
import pandas as pd
import cv2
from src.data.labels import load_labels, patient_id_from_name, class_table


def _img(p):
    p.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p), np.zeros((8, 8, 3), np.uint8))


def test_load_labels_matches_images_across_folders(tmp_path):
    for i in ["a1", "a2", "a3"]:
        _img(tmp_path / "train_images" / f"{i}.png")
    _img(tmp_path / "val_images" / "b1.png")
    pd.DataFrame({"id_code": ["a1", "a2", "a3"], "diagnosis": [0, 2, 4]}).to_csv(tmp_path / "train_1.csv", index=False)
    pd.DataFrame({"id_code": ["b1", "ghost"], "diagnosis": [1, 3]}).to_csv(tmp_path / "valid.csv", index=False)
    pd.DataFrame({"foo": [1]}).to_csv(tmp_path / "other.csv", index=False)
    df, rep = load_labels(tmp_path, verbose=False)
    assert len(df) == 4 and rep["labeled_without_image"] == 1
    assert sorted(df["label"]) == [0, 1, 2, 4]
    assert rep["skipped_csvs"][0][0] == "other.csv"


def test_patient_id_and_class_table():
    assert patient_id_from_name("10_left") == "10" and patient_id_from_name("abc") is None
    t = class_table(pd.DataFrame({"label": [0, 0, 1, 4]}))
    assert t["count"].tolist() == [2, 1, 0, 0, 1]
