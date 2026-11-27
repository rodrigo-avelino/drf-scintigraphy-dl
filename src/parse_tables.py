"""Fase 1 - Parser das tabelas de TABLES_b.

Extrai os labels (funcao diferencial, GFR, NORA, OE, diagnostico...) de cada
conjunto e salva CSVs em data/labels/.
"""
from pathlib import Path

import openpyxl
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "dynamicrenalstudy_DICOM_package" / "database" / "TABLES_b"
OUT = ROOT / "data" / "labels"


def sheet_rows(path: Path, sheet: str, skip: int = 2, max_col: int | None = None):
    """Linhas de dados de uma aba (pula cabecalho de 2 linhas por padrao)."""
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb[sheet]
    rows = []
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < skip:
            continue
        rows.append(row[:max_col] if max_col else row)
    wb.close()
    return rows


def clean(rows, id_prefix):
    """Mantem apenas linhas cujo 1o campo comeca com o prefixo do conjunto."""
    return [r for r in rows if r and isinstance(r[0], str) and r[0].startswith(id_prefix)]


def parse_drsbru():
    path = TABLES / "drsbru" / "table_drsbru_data_140504.xlsx"
    diag = clean(sheet_rows(path, "clinical diagnosis", max_col=2), "drsbru")
    df_diag = pd.DataFrame(diag, columns=["study", "diagnosis"])

    pat = clean(sheet_rows(path, "patient data", max_col=14), "drsbru")
    cols = ["study", "sex", "age_yrs", "gfr_ml_min", "lk_pct_integral", "lk_pct_pr",
            "nora20_lk", "nora_pm_lk", "nora20_rk", "nora_pm_rk",
            "oe20_lk", "oe_pm_lk", "oe20_rk", "oe_pm_rk"]
    df_pat = pd.DataFrame(pat, columns=cols)

    df = df_pat.merge(df_diag, on="study", how="outer")
    df.to_csv(OUT / "labels_drsbru.csv", index=False)
    return df


def parse_drsprg():
    lst = TABLES / "drsprg" / "data_list_export_120217.xlsx"
    rows = clean(sheet_rows(lst, "DATABASE_LIST_120317(1)", max_col=10), "drsprg")
    cols = ["study", "sex", "age_yrs", "ckd_stage", "lk_pct",
            "has_scr", "has_crc", "has_mag3", "has_edta", "has_flood"]
    df = pd.DataFrame(rows, columns=cols)

    tbl = TABLES / "drsprg" / "data_tables_export_120719.xlsx"
    diag = clean(sheet_rows(tbl, "AVAILABLE_CLINICAL_DIAGNOSIS", max_col=2), "drsprg")
    df = df.merge(pd.DataFrame(diag, columns=["study", "diagnosis"]), on="study", how="left")

    clin = clean(sheet_rows(tbl, "CLINICAL_DATA", max_col=8), "drsprg")
    df = df.merge(
        pd.DataFrame(clin, columns=["study", "sex2", "age2", "weight_kg", "height_cm",
                                    "bmi", "bsa_h", "bsa_d"]).drop(columns=["sex2", "age2"]),
        on="study", how="left")

    lab = clean(sheet_rows(tbl, "LABORATORY_DATA", max_col=8), "drsprg")
    df = df.merge(
        pd.DataFrame(lab, columns=["study", "tfs_days", "screat_umol_l", "screat_mg100ml",
                                   "ccreat_ml_s", "ccreat_ml_min", "surea_mmol_l",
                                   "surea_mg100ml"]).drop(columns=["screat_mg100ml", "surea_mg100ml"]),
        on="study", how="left")

    df.to_csv(OUT / "labels_drsprg.csv", index=False)
    return df


def parse_drshyb():
    path = TABLES / "drshyb" / "table_drshyb_data_140504.xlsx"
    rows = sheet_rows(path, "drshyb_data_description", skip=3, max_col=8)
    out, study = [], None
    for r in rows:
        if r[0] and str(r[0]).startswith("drshyb"):
            study = r[0]
        if r[1] and str(r[1]).startswith("drshyb"):
            out.append({"study": study, "file": r[1], "kb_contrast": r[6],
                        "lk_pct_true": r[7]})
    df = pd.DataFrame(out)
    # uptake vale para as duas variantes do estudo -> propaga
    df["lk_pct_true"] = df.groupby("study")["lk_pct_true"].transform(
        lambda s: s.ffill().bfill())
    df.to_csv(OUT / "labels_drshyb.csv", index=False)
    return df


def parse_drslund():
    path = TABLES / "drslund" / "table_drslund_data_140504.xlsx"
    rows = sheet_rows(path, "drslund_data_description", skip=3, max_col=9)
    out, study = [], None
    for r in rows:
        if r[0] and str(r[0]).startswith("drslund"):
            study = r[0]
        if r[1] and str(r[1]).startswith("drslund"):
            out.append({"study": study, "file": r[1], "injected_mbq": r[5],
                        "lk_pct_true": r[7], "clearance_ml_min": r[8]})
    df = pd.DataFrame(out)
    for c in ["lk_pct_true", "clearance_ml_min"]:
        df[c] = df.groupby("study")[c].transform(lambda s: s.ffill().bfill())
    df.to_csv(OUT / "labels_drslund.csv", index=False)
    return df


def parse_drsmich():
    path = TABLES / "drsmich" / "table_drsmich_data_140504.xlsx"
    rows = clean(sheet_rows(path, "drsmich_data_description", skip=3, max_col=8), "drsmich")
    df = pd.DataFrame(
        [{"study": r[0], "injected_pct": r[5], "lk_pct_true": r[7]} for r in rows])
    df.to_csv(OUT / "labels_drsmich.csv", index=False)
    return df


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in [parse_drsbru, parse_drsprg, parse_drshyb, parse_drslund, parse_drsmich]:
        df = fn()
        print(f"{fn.__name__}: {len(df)} linhas, colunas: {list(df.columns)}")
