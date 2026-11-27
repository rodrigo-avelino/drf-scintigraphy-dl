"""Fase 1 - Dataframe mestre: 1 linha por estudo.

Cruza o inventario de DICOMs (data/inventory_files.csv) com os labels das
tabelas (data/labels/*.csv) e reporta inconsistencias. Salva data/master.csv
e data/master.parquet.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def pick_path(g: pd.DataFrame, pattern: str) -> str | None:
    hit = g[g.file.str.contains(pattern, regex=True)]
    return hit.path.iloc[0] if len(hit) else None


def main() -> None:
    inv = pd.read_csv(DATA / "inventory_files.csv")

    # ---- 1 linha por estudo a partir do inventario -------------------------
    rows = []
    for study, g in inv.groupby("study"):
        dataset = g.dataset.iloc[0]
        main = g[g.role == "dinamica_principal"]
        # projecao posterior de referencia por conjunto
        post_pattern = {
            "drsprg": r"_POST\.dcm", "drsbru": r"_POST\.dcm",
            "drsmich": r"_POST\.dcm", "drshyb": r"_H\d+\.dcm",
            "drslund": r"_POST_100\.dcm",
        }[dataset]
        ref = main[main.file.str.contains(post_pattern)]
        ref_row = ref.iloc[0] if len(ref) else main.iloc[0]
        rows.append({
            "dataset": dataset,
            "study": study,
            "path_post": ref_row.path,
            "path_ant": pick_path(main, r"_ANT\.dcm|_ANT_100\.dcm"),
            "path_pos_miccao": pick_path(g[g.role == "pos_miccao"], r"POST"),
            "paths_extras": ";".join(
                sorted(set(main.path) | set(g[g.role == "dinamica_gold_standard"].path)
                       - {ref_row.path})) or None,
            "n_frames": ref_row.n_frames,
            "frame_schedule": ref_row.frame_schedule,
            "img_size": ref_row["rows"],
            "pixel_spacing_mm": ref_row.pixel_spacing_mm,
            "n_files": len(g),
        })
    master = pd.DataFrame(rows)

    # ---- labels ------------------------------------------------------------
    bru = pd.read_csv(DATA / "labels" / "labels_drsbru.csv")
    prg = pd.read_csv(DATA / "labels" / "labels_drsprg.csv")
    hyb = pd.read_csv(DATA / "labels" / "labels_drshyb.csv").drop_duplicates("study")
    lund = pd.read_csv(DATA / "labels" / "labels_drslund.csv").drop_duplicates("study")
    mich = pd.read_csv(DATA / "labels" / "labels_drsmich.csv")

    # label alvo unificado: funcao diferencial do rim esquerdo (%)
    bru["lk_pct_label"] = bru.lk_pct_integral
    prg["lk_pct_label"] = prg.lk_pct
    for df in (hyb, lund, mich):
        df["lk_pct_label"] = df.lk_pct_true

    labels = pd.concat([
        bru[["study", "lk_pct_label", "sex", "age_yrs", "gfr_ml_min",
             "lk_pct_integral", "lk_pct_pr", "nora20_lk", "nora20_rk",
             "oe20_lk", "oe20_rk", "diagnosis"]],
        prg[["study", "lk_pct_label", "sex", "age_yrs", "ckd_stage",
             "ccreat_ml_min", "diagnosis"]],
        hyb[["study", "lk_pct_label"]],
        lund[["study", "lk_pct_label", "clearance_ml_min"]],
        mich[["study", "lk_pct_label"]],
    ], ignore_index=True)

    master = master.merge(labels, on="study", how="outer", indicator=True)

    # ---- checagens ---------------------------------------------------------
    sem_tabela = master[master._merge == "left_only"].study.tolist()
    sem_imagem = master[master._merge == "right_only"].study.tolist()
    sem_label = master[master.lk_pct_label.isna()].study.tolist()
    print(f"Estudos com imagem e sem linha na tabela: {sem_tabela or 'nenhum'}")
    print(f"Estudos na tabela sem pasta de imagem:   {sem_imagem or 'nenhum'}")
    print(f"Estudos sem label de funcao diferencial: {len(sem_label)} -> {sem_label}")

    # arquivos listados nas tabelas de imagem que nao existem no disco (drslund)
    lund_files = pd.read_csv(DATA / "labels" / "labels_drslund.csv")
    inv_files = set(inv.file)
    faltando = [f for f in lund_files.file if f not in inv_files]
    print(f"Arquivos listados na tabela drslund ausentes no disco: {faltando or 'nenhum'}")

    master = master.drop(columns="_merge")
    master.to_csv(DATA / "master.csv", index=False)
    master.to_parquet(DATA / "master.parquet", index=False)
    print(f"\nMaster: {len(master)} estudos -> data/master.csv | data/master.parquet")
    print(master.groupby("dataset").agg(
        estudos=("study", "count"),
        com_label=("lk_pct_label", "count"),
        lk_pct_medio=("lk_pct_label", "mean")).round(1))


if __name__ == "__main__":
    main()
