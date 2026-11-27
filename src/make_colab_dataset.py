"""Fase 2 - Empacota o dataset em um .npz para os treinos (Colab e execucao local).

Gera data/colab_renal_lk.npz com n = 245 amostras: 207 estudos reais (107
drsprg e 100 drsbru) e 38 variantes hibridas (19 estudos drshyb x 2), so
para treino:
  X_full       (n,128,128) soma da POST, janela 0-20 min
  X_uptake     (n,128,128) soma da POST, janela de captacao 60-180 s
                           (fase que o metodo de Gates usa para o DRF)
  X_blocos     (n,20,128,128) uint16 - soma da POST em blocos de 60 s,
                           minutos 0 a 20 (entrada das representacoes da
                           fase3_lib, que descarta o primeiro bloco)
  X_ant_full   (n,128,128) soma da ANT espelhada horizontalmente (alinhada
  X_ant_uptake (n,128,128)  com a POST); zeros onde nao ha ANT (drsbru!)
  tem_ant      (n,)  bool  - True apenas no drsprg (drsbru so tem POST)
  hibrido      (n,)  bool  - True nas variantes drshyb (nunca na validacao)
  y            (n,)        - funcao diferencial LK% (tabelas = ground truth)
  study        (n,)        - id do estudo (= paciente; usar no split!)
  grupo        (n,)        - drsprg (adulto) | drsbru (pediatrico) | drshyb
  idade, sexo  (n,)
  lk_cru       (n,)        - baseline classico (ROI automatica); NaN no drshyb

Aquisicao principal apenas (POST/ANT); POST2/ANT2 (pos-miccao) ficam FORA da
soma - sao aquisicoes tardias separadas, com 15 drsbru ausentes. Upload do
.npz para o Drive; uso nos notebooks do Colab e na execucao local, ambos via
src/fase3_lib.py (carregar_dados).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_sanity import load_dynamic

ROOT = Path(__file__).resolve().parents[1]
JANELA_FULL_S = 1200        # 0-20 min
JANELA_UPTAKE = (60, 180)   # fase de captacao, 1-3 min
N_BLOCOS, BLOCO_S = 20, 60  # 20 blocos de 60 s (soma de 6 frames de 10 s)


def somas(cube, times):
    full = cube[times < JANELA_FULL_S].sum(axis=0).astype(np.float32)
    upt = cube[(times >= JANELA_UPTAKE[0]) & (times < JANELA_UPTAKE[1])
               ].sum(axis=0).astype(np.float32)
    assert full.shape == (128, 128), f"shape {full.shape}"
    return full, upt


def blocos_temporais(cube, times):
    """(N_BLOCOS,128,128) uint16: soma dos frames de cada janela de 60 s."""
    b = (times // BLOCO_S).astype(int)
    out = np.zeros((N_BLOCOS, 128, 128), dtype=np.float32)
    for k in range(N_BLOCOS):
        out[k] = cube[b == k].sum(axis=0)
    assert out.max() < 65535, "bloco estoura uint16"
    return out.astype(np.uint16)


master = pd.read_csv(ROOT / "data" / "master.csv")
master = master[master.dataset.isin(["drsprg", "drsbru"])].reset_index(drop=True)
cru = pd.read_csv(ROOT / "data" / "sanity_lk_pct_cru.csv").set_index("study")

cols = {k: [] for k in ["X_full", "X_uptake", "X_blocos", "X_ant_full",
                        "X_ant_uptake", "tem_ant", "hibrido", "y", "study",
                        "grupo", "idade", "sexo", "lk_cru"]}
for k, row in master.iterrows():
    cube, times, _ = load_dynamic(row.path_post)
    full, upt = somas(cube, times)
    cols["X_full"].append(full)
    cols["X_uptake"].append(upt)
    cols["X_blocos"].append(blocos_temporais(cube, times))
    if pd.notna(row.path_ant):
        cube_a, times_a, _ = load_dynamic(row.path_ant)
        a_full, a_upt = somas(cube_a, times_a)
        # espelha a ANT para alinhar esquerda/direita com a POST
        cols["X_ant_full"].append(np.fliplr(a_full).copy())
        cols["X_ant_uptake"].append(np.fliplr(a_upt).copy())
        cols["tem_ant"].append(True)
    else:
        cols["X_ant_full"].append(np.zeros((128, 128), dtype=np.float32))
        cols["X_ant_uptake"].append(np.zeros((128, 128), dtype=np.float32))
        cols["tem_ant"].append(False)
    cols["hibrido"].append(False)
    cols["y"].append(row.lk_pct_label)
    cols["study"].append(row.study)
    cols["grupo"].append(row.dataset)
    cols["idade"].append(row.age_yrs if pd.notna(row.age_yrs) else np.nan)
    cols["sexo"].append(str(row.sex) if pd.notna(row.sex) else "?")
    cols["lk_cru"].append(cru.lk_pct_cru.get(row.study, np.nan))
    if (k + 1) % 50 == 0 or k + 1 == len(master):
        print(f"  {k + 1}/{len(master)}")

# ---- drshyb: hibridos com label exato, TREINO APENAS -----------------
# As 2 variantes (H/L = contraste rim-fundo alto/baixo) de cada estudo
# entram como amostras de treino para enriquecer os extremos de LK%; a
# flag 'hibrido' permite exclui-los da validacao SEMPRE.
hyb = pd.read_csv(ROOT / "data" / "labels" / "labels_drshyb.csv")
DICOM_DIR = ROOT / "dynamicrenalstudy_DICOM_package" / "database" / "DATA_DICOM"
for _, r in hyb.iterrows():
    cube, times, _ = load_dynamic(DICOM_DIR / r.study / r.file)
    full, upt = somas(cube, times)
    variante = r.file.replace(".dcm", "").replace(r.study + "_", "")
    cols["X_full"].append(full)
    cols["X_uptake"].append(upt)
    cols["X_blocos"].append(blocos_temporais(cube, times))
    cols["X_ant_full"].append(np.zeros((128, 128), dtype=np.float32))
    cols["X_ant_uptake"].append(np.zeros((128, 128), dtype=np.float32))
    cols["tem_ant"].append(False)
    cols["hibrido"].append(True)
    cols["y"].append(float(r.lk_pct_true))
    cols["study"].append(f"{r.study}_{variante}")
    cols["grupo"].append("drshyb")
    cols["idade"].append(np.nan)
    cols["sexo"].append("?")
    cols["lk_cru"].append(np.nan)
print(f"  + {len(hyb)} variantes drshyb (treino apenas)")

out = ROOT / "data" / "colab_renal_lk.npz"
np.savez_compressed(
    out,
    X_full=np.stack(cols["X_full"]),
    X_uptake=np.stack(cols["X_uptake"]),
    X_blocos=np.stack(cols["X_blocos"]),
    X_ant_full=np.stack(cols["X_ant_full"]),
    X_ant_uptake=np.stack(cols["X_ant_uptake"]),
    tem_ant=np.array(cols["tem_ant"]),
    hibrido=np.array(cols["hibrido"]),
    y=np.array(cols["y"], dtype=np.float32),
    study=np.array(cols["study"]), grupo=np.array(cols["grupo"]),
    idade=np.array(cols["idade"], dtype=np.float32),
    sexo=np.array(cols["sexo"]),
    lk_cru=np.array(cols["lk_cru"], dtype=np.float32),
)
n_ant = int(np.sum(cols["tem_ant"]))
print(f"ok -> {out} ({out.stat().st_size / 1e6:.1f} MB, {len(cols['y'])} estudos, "
      f"{n_ant} com ANT)")
