"""Fase 1 - Visualizacoes de sanidade.

Gera em reports/figures/:
  fig1_imagens_somadas.png   - imagem somada de 1 estudo por conjunto, com ROIs
  fig2_curvas_tac.png        - curvas tempo-atividade LK vs RK em 3 casos drsbru
  fig3_validacao_lkpct.png   - LK% cru (ROI) vs LK% da tabela (drsprg+drsbru)
  fig4_distribuicao_labels.png - histogramas dos labels

ROI grosseira: pico da imagem somada da fase de captacao (1-2.5 min) em cada
metade da imagem, caixa de ~7 cm. Serve apenas para sanidade, nao para medida.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import pydicom
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"


def load_dynamic(path):
    """Retorna (cube [t,y,x] float, tempos de inicio de frame em segundos,
    tamanho do pixel em mm)."""
    ds = pydicom.dcmread(ROOT / path)
    cube = ds.pixel_array.astype(np.float32)
    times, t = [], 0.0
    phases = getattr(ds, "PhaseInformationSequence", None)
    if phases:
        for p in phases:
            dur = int(p.ActualFrameDuration) / 1000.0
            for _ in range(int(p.NumberOfFramesInPhase)):
                times.append(t)
                t += dur
    else:
        dur = int(getattr(ds, "ActualFrameDuration", 10000)) / 1000.0
        times = [i * dur for i in range(cube.shape[0])]
    spacing = float(getattr(ds, "PixelSpacing", [3.0])[0])
    return cube, np.array(times), spacing


def kidney_rois(cube, times, spacing):
    """Acha ROIs grosseiras dos 2 rins na fase de captacao (60-150 s)."""
    sel = (times >= 60) & (times < 150)
    early = ndimage.gaussian_filter(cube[sel].sum(axis=0), 2.5)
    ny, nx = early.shape
    band = early[int(0.2 * ny):int(0.8 * ny)]  # evita coracao/bexiga
    half = max(6, int(35.0 / spacing))  # meia-caixa ~3.5 cm
    rois = {}
    for name, sl in [("esq_img", slice(0, nx // 2)), ("dir_img", slice(nx // 2, nx))]:
        sub = band[:, sl]
        y, x = np.unravel_index(np.argmax(sub), sub.shape)
        y += int(0.2 * ny)
        x += sl.start
        rois[name] = (max(0, y - half), min(ny, y + half),
                      max(0, x - half), min(nx, x + half))
    return rois


def tac(cube, roi):
    y0, y1, x0, x1 = roi
    return cube[:, y0:y1, x0:x1].sum(axis=(1, 2))


def crude_lk_pct(path):
    """LK% cru: contagens na fase de captacao, ROI esquerda vs direita.

    Projecao POSTERIOR: lado esquerdo do paciente aparece a ESQUERDA da imagem.
    """
    cube, times, spacing = load_dynamic(path)
    rois = kidney_rois(cube, times, spacing)
    sel = (times >= 60) & (times < 150)
    lk = cube[sel][:, slice(*rois["esq_img"][:2]), slice(*rois["esq_img"][2:])].sum()
    rk = cube[sel][:, slice(*rois["dir_img"][:2]), slice(*rois["dir_img"][2:])].sum()
    return 100.0 * lk / (lk + rk)


def fig_imagens_somadas(master):
    exemplos = master.groupby("dataset").first().reset_index()
    fig, axes = plt.subplots(1, len(exemplos), figsize=(4 * len(exemplos), 4.4))
    for ax, (_, row) in zip(axes, exemplos.iterrows()):
        cube, times, spacing = load_dynamic(row.path_post)
        rois = kidney_rois(cube, times, spacing)
        ax.imshow(np.log1p(cube.sum(axis=0)), cmap="inferno")
        for name, color in [("esq_img", "cyan"), ("dir_img", "lime")]:
            y0, y1, x0, x1 = rois[name]
            ax.add_patch(mpatches.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                            fill=False, color=color, lw=1.5))
        ax.set_title(f"{row.study}\nLK={row.lk_pct_label:.0f}%  "
                     f"{row.frame_schedule}", fontsize=9)
        ax.axis("off")
    fig.suptitle("Imagem somada (log) por conjunto - ROI ciano = rim esquerdo, "
                 "verde = rim direito (projecao posterior)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_imagens_somadas.png", dpi=130)
    plt.close(fig)


def fig_curvas_tac(master):
    b = master[master.dataset == "drsbru"].copy()
    normal = b[(b.lk_pct_label.between(45, 55)) & (b.nora20_lk < 0.5)
               & (b.nora20_rk < 0.5)].iloc[0]
    assimetrico = b.loc[(b.lk_pct_label - 50).abs().idxmax()]
    retencao = b.loc[b.nora20_lk.idxmax()]
    casos = [(normal, "simetrico, boa drenagem"),
             (assimetrico, "funcao assimetrica"),
             (retencao, "retencao (NORA20 LK alto)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, (row, desc) in zip(axes, casos):
        cube, times, spacing = load_dynamic(row.path_post)
        rois = kidney_rois(cube, times, spacing)
        ax.plot(times / 60, tac(cube, rois["esq_img"]), label="rim esquerdo", lw=1.6)
        ax.plot(times / 60, tac(cube, rois["dir_img"]), label="rim direito", lw=1.6)
        ax.set_title(f"{row.study} - {desc}\nLK={row.lk_pct_label:.0f}%  "
                     f"NORA20 LK={row.nora20_lk}  RK={row.nora20_rk}", fontsize=9)
        ax.set_xlabel("tempo (min)")
        ax.set_ylabel("contagens na ROI")
        ax.legend(fontsize=8)
    fig.suptitle("Curvas tempo-atividade (renogramas) com ROI grosseira - drsbru",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_curvas_tac.png", dpi=130)
    plt.close(fig)


def fig_validacao(master):
    sub = master[master.dataset.isin(["drsprg", "drsbru"])].copy()
    print(f"Calculando LK% cru em {len(sub)} estudos (pode demorar um pouco)...")
    sub["lk_pct_cru"] = [crude_lk_pct(p) for p in sub.path_post]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, ds_name in zip(axes, ["drsprg", "drsbru"]):
        g = sub[sub.dataset == ds_name]
        r = np.corrcoef(g.lk_pct_cru, g.lk_pct_label)[0, 1]
        ax.scatter(g.lk_pct_label, g.lk_pct_cru, s=18, alpha=0.6)
        ax.plot([0, 100], [0, 100], "k--", lw=1)
        ax.set_xlabel("LK% tabela (referencia)")
        ax.set_ylabel("LK% cru (ROI automatica, sem fundo)")
        ax.set_title(f"{ds_name}  (r = {r:.3f}, n = {len(g)})")
    fig.suptitle("Sanidade: funcao diferencial crua vs valores de referencia",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_validacao_lkpct.png", dpi=130)
    plt.close(fig)
    sub[["study", "lk_pct_label", "lk_pct_cru"]].to_csv(
        ROOT / "data" / "sanity_lk_pct_cru.csv", index=False)
    return sub


def fig_distribuicoes(master):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    b = master[master.dataset == "drsbru"]
    p = master[master.dataset == "drsprg"]
    axes[0, 0].hist([p.lk_pct_label, b.lk_pct_label], bins=20,
                    label=["drsprg (adulto)", "drsbru (pediatrico)"], stacked=True)
    axes[0, 0].set_title("Funcao diferencial LK (%)")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].hist(b.gfr_ml_min.dropna(), bins=20, color="tab:green")
    axes[0, 1].set_title("GFR (ml/min) - drsbru")
    axes[0, 2].bar(*np.unique(p.ckd_stage.dropna(), return_counts=True),
                   color="tab:orange")
    axes[0, 2].set_title("Estagio CKD - drsprg")
    axes[1, 0].hist([b.nora20_lk.dropna(), b.nora20_rk.dropna()], bins=20,
                    label=["LK", "RK"])
    axes[1, 0].set_title("NORA(20) - drsbru")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].hist([b.oe20_lk.dropna(), b.oe20_rk.dropna()], bins=20,
                    label=["LK", "RK"])
    axes[1, 1].set_title("OE(20) (%) - drsbru")
    axes[1, 1].legend(fontsize=8)
    axes[1, 2].hist([p.age_yrs.dropna(), b.age_yrs.dropna()], bins=20,
                    label=["drsprg", "drsbru"])
    axes[1, 2].set_title("Idade (anos)")
    axes[1, 2].legend(fontsize=8)
    fig.suptitle("Distribuicao dos labels de referencia", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_distribuicao_labels.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    master = pd.read_csv(ROOT / "data" / "master.csv")
    fig_imagens_somadas(master)
    print("fig1 ok")
    fig_curvas_tac(master)
    print("fig2 ok")
    fig_distribuicoes(master)
    print("fig4 ok")
    fig_validacao(master)
    print("fig3 ok - figuras em reports/figures/")
