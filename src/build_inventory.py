"""Fase 1 - Inventario dos DICOMs.

Varre DATA_DICOM/, le o header de cada arquivo DICOM (sem pixel data) e gera
data/inventory_files.csv com 1 linha por arquivo.
"""
import csv
import re
from pathlib import Path

import pydicom

ROOT = Path(__file__).resolve().parents[1]
DICOM_DIR = ROOT / "dynamicrenalstudy_DICOM_package" / "database" / "DATA_DICOM"
OUT = ROOT / "data" / "inventory_files.csv"


def classify_file(name: str, study: str) -> str:
    """Classifica o papel do arquivo dentro do estudo a partir do nome."""
    suffix = name.replace(study + "_", "").replace(".dcm", "")
    if "flood" in suffix:
        return "flood_calibracao"
    if re.fullmatch(r"(POST|ANT)", suffix):
        return "dinamica_principal"
    if re.fullmatch(r"(POST2|ANT2)", suffix):
        return "pos_miccao"
    if suffix.endswith("_PM"):
        return "pos_miccao"
    if re.fullmatch(r"[LH]\d+", suffix):  # drshyb: L05, H20...
        return "dinamica_principal"
    if "GS" in suffix:  # drslund gold standard
        return "dinamica_gold_standard"
    if re.fullmatch(r"(POST|ANT)_\d+", suffix):  # drslund POST_50 etc.
        return "dinamica_principal"
    return "outro"


def frame_schedule(ds) -> tuple[int | None, str]:
    """Retorna (n_frames, descricao das fases 'NxDs+...')."""
    n = getattr(ds, "NumberOfFrames", None)
    phases = getattr(ds, "PhaseInformationSequence", None)
    if phases:
        parts = [
            f"{getattr(p, 'NumberOfFramesInPhase', '?')}x{round(int(getattr(p, 'ActualFrameDuration', 0)) / 1000)}s"
            for p in phases
        ]
        return n, "+".join(parts)
    dur = getattr(ds, "ActualFrameDuration", None)
    return n, f"{n}x{round(int(dur) / 1000)}s" if dur else ""


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    rows = []
    for study_dir in sorted(DICOM_DIR.iterdir()):
        if not study_dir.is_dir():
            continue
        study = study_dir.name
        dataset = study.split("_")[0]
        for f in sorted(study_dir.glob("*.dcm")):
            ds = pydicom.dcmread(f, stop_before_pixels=True)
            n_frames, schedule = frame_schedule(ds)
            spacing = getattr(ds, "PixelSpacing", [None, None])
            radio = None
            ris = getattr(ds, "RadiopharmaceuticalInformationSequence", None)
            if ris:
                radio = getattr(ris[0], "Radiopharmaceutical", None) or None
                if not radio:
                    codes = getattr(ris[0], "RadionuclideCodeSequence", [])
                    radio = codes[0].CodeMeaning if codes else None
            rows.append({
                "dataset": dataset,
                "study": study,
                "file": f.name,
                "path": str(f.relative_to(ROOT)),
                "role": classify_file(f.name, study),
                "n_frames": n_frames,
                "rows": getattr(ds, "Rows", None),
                "cols": getattr(ds, "Columns", None),
                "frame_schedule": schedule,
                "pixel_spacing_mm": round(float(spacing[0]), 4) if spacing and spacing[0] else None,
                "radiopharmaceutical": radio,
                "study_description": getattr(ds, "StudyDescription", None),
                "series_description": getattr(ds, "SeriesDescription", None),
                "image_id": getattr(ds, "ImageID", None),
                "manufacturer": getattr(ds, "Manufacturer", None),
                "study_date": getattr(ds, "StudyDate", None),
                "patient_sex": getattr(ds, "PatientSex", None) or None,
                "patient_age": getattr(ds, "PatientAge", None) or None,
                "size_mb": round(f.stat().st_size / 1e6, 2),
            })
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} arquivos inventariados -> {OUT}")


if __name__ == "__main__":
    main()
