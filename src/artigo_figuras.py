# -*- coding: utf-8 -*-
"""Figuras do artigo (EJNMMI Physics), em ingles, em artigo/figures/.

fig1_inputs: as tres entradas (imagem somada, mosaico 3 x 3 e sequencia de 19
  quadros de 1 min) de um estudo SIMULADO do banco (drslund_001, projecao
  posterior, atividade de 100%; simulacao de Monte Carlo, sem dado de
  paciente), com o mesmo pre-processamento da rede (blocos de 60 s,
  agrupar_tempo, log1p e divisao pelo maximo do estudo).
fig2_pred_vs_ref: predito contra referencia dos tres metodos com validacao
  interna (as 4 sementes de cada estudo) e da PDI recalibrada.
fig3_bland_altman: Bland-Altman dos mesmos quatro metodos; vies e limites
  como na Tabela 1 (media dos valores por semente; PDI nos 207 estudos).
fig4_error_by_range: MAE por faixa da LK de referencia com IC 95% (bootstrap
  dos estudos da faixa, 5000 reamostragens, semente 0) e vies por faixa.

Cores: aqua, laranja e azul para os tres metodos e cinza para a referencia
classica. A paleta foi conferida com um validador de diferenca de cor (OKLab):
todos os pares das tres cores ficam acima de Delta E 8 sob simulacao de
daltonismo (pior par: 9,2, deuteranopia). O aqua tem contraste de 2,7:1 com o
branco, abaixo de 3:1, entao ha legenda e marcadores de formas diferentes.

Uso: .venv\\Scripts\\python -X utf8 src\\artigo_figuras.py
Saidas: artigo/figures/fig1_inputs.pdf ... fig4_error_by_range.pdf (vetoriais,
fontes embutidas) e previas em PNG na pasta indicada por --previas.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fase3_lib as f3  # noqa: E402
import fase4_analise as fa  # noqa: E402
import fase4_fila as q  # noqa: E402
from eda_sanity import load_dynamic  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SAIDA = ROOT / "artigo" / "figures"
MM = 1 / 25.4                       # polegadas por mm
LARGURA = 170 * MM                  # figura de pagina inteira na revista
ESTUDO_SIMULADO = ("dynamicrenalstudy_DICOM_package/database/DATA_DICOM/"
                   "drslund_001/drslund_001_POST_100.dcm")

# Tokens da paleta de referencia (modo claro); a superficie e branca (impressao).
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE = "#e1e0d9", "#c3c2b7"
METODOS = {   # rotulo da analise -> (nome em ingles, cor, marcador)
    "2D colapsado (grade 1 × 1)": ("2D collapsed", "#1baf7a", "^"),
    "mosaico 3 × 3": ("Temporal mosaic", "#eb6834", "s"),
    "R3D-18 com cabeça espacial e aquecimento": ("3D CNN (R3D-18)", "#2a78d6", "o"),
    "PDI recalibrada": ("Automatic ROI, recalibrated", "#52514e", "D"),
}
FAIXAS_EN = ["<35%", "35–45%", "45–55%", "55–65%", "≥65%"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": BASE, "axes.linewidth": 0.6, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
})


# ------------------------------------------------------------------ dados
def carregar():
    """Predicoes por semente dos tres metodos com validacao interna e a PDI
    por estudo (as mesmas fontes de src/fase4_analise.py)."""
    fontes = [(q.PASTA, q.NOME), (q.PASTA_F3, "fase3")]
    preds, falt = fa._ler_bracos(fa._bracos_artigo(True), fontes)
    if falt:
        raise RuntimeError(f"bracos incompletos: {falt}")
    pdi, _ = fa.pdi_por_estudo()
    if pdi is None:
        raise RuntimeError("sem data/colab_renal_lk.npz para a PDI")
    p = pdi.reset_index()[["study", "fold", "y", "pdi_recal"]].rename(
        columns={"pdi_recal": "pred"})
    p["semente"] = 0
    preds["PDI recalibrada"] = p
    return preds


def reta_media(d):
    """Inclinacao e intercepto medios entre sementes (como na Tabela 1)."""
    coef = [np.polyfit(g.y, g.pred, 1) for _, g in d.groupby("semente")]
    return np.mean(coef, axis=0)


def ba_medio(d):
    """Vies e limites de 95% por semente, em media (como na Tabela 1)."""
    v = []
    for _, g in d.groupby("semente"):
        dif = g.pred - g.y
        m, s = dif.mean(), dif.std(ddof=1)
        v.append((m, m - 1.96 * s, m + 1.96 * s))
    return np.mean(v, axis=0)


def salvar(fig, nome, previas):
    SAIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SAIDA / f"{nome}.pdf", bbox_inches="tight", pad_inches=0.02)
    if previas:
        Path(previas).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(previas) / f"{nome}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def letra(ax, t, x=-0.02, y=1.04):
    ax.text(x, y, t, transform=ax.transAxes, fontsize=9, fontweight="bold",
            color=INK, ha="right", va="bottom")


# --------------------------------------------------------------- figura 1
def blocos_60s(cube, times):
    """Blocos de 60 s como em src/make_colab_dataset.py (20 x 128 x 128)."""
    b = (times // 60).astype(int)
    return np.stack([cube[b == k].sum(axis=0) for k in range(20)])


def normalizar(x):
    x = np.log1p(x.astype(np.float64))
    return x / x.max()


def fig1_entradas(previas):
    cube, times, _ = load_dynamic(ESTUDO_SIMULADO)
    xb = blocos_60s(cube, times)[None]                 # (1, 20, H, W)
    soma = normalizar(f3.agrupar_tempo(xb, 1)[0])[0]
    celulas = normalizar(f3.agrupar_tempo(xb, 9)[0])
    quadros = normalizar(f3.agrupar_tempo(xb, 19)[0])
    mosaico = np.vstack([np.hstack(celulas[r * 3:(r + 1) * 3]) for r in range(3)])
    lim_min = np.cumsum([0, 3] + [2] * 8) + 1          # 1, 4, 6, ..., 20 min

    fig = plt.figure(figsize=(LARGURA, 62 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.28], wspace=0.12)
    opc = dict(cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")

    ax = fig.add_subplot(gs[0])
    ax.imshow(soma, **opc)
    # projecao posterior: o lado esquerdo do paciente fica a esquerda da imagem
    for xx, t, ha in ((3, "L", "left"), (soma.shape[1] - 4, "R", "right")):
        ax.text(xx, soma.shape[0] - 4, t, fontsize=7, fontweight="bold",
                color=INK2, ha=ha, va="bottom")
    ax.set_title("Summed image (1–20 min)", color=INK, pad=4)
    letra(ax, "a")
    ax = fig.add_subplot(gs[1])
    ax.imshow(mosaico, **opc)
    n = celulas.shape[1]
    for i in range(9):
        r, c = divmod(i, 3)
        ax.text(c * n + 3, r * n + 3, f"{lim_min[i]}–{lim_min[i + 1]} min",
                fontsize=5.5, color=INK2, ha="left", va="top",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                          pad=0.6))
    for k in (1, 2):
        ax.axhline(k * n - 0.5, color="white", lw=1.2)
        ax.axvline(k * n - 0.5, color="white", lw=1.2)
    ax.set_title("Temporal mosaic (3 × 3)", color=INK, pad=4)
    letra(ax, "b")
    sub = gs[2].subgridspec(4, 5, wspace=0.06, hspace=0.06)
    for i in range(20):
        ax = fig.add_subplot(sub[i // 5, i % 5])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if i < 19:
            ax.imshow(quadros[i], **opc)
            ax.text(3, 3, str(i + 1), fontsize=5.5, color=INK2, ha="left", va="top")
        if i == 2:
            ax.set_title("Frame sequence (19 × 1 min)", color=INK, pad=4)
        if i == 0:
            letra(ax, "c", x=-0.08, y=1.12)
    for ax in fig.axes[:2]:
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    salvar(fig, "fig1_inputs", previas)


# --------------------------------------------------------------- figura 2
def eixos_drf(ax):
    ax.set_xlim(0, 100), ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 20)), ax.set_yticks(range(0, 101, 20))
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


def fig2_predito(preds, previas):
    fig, eixos = plt.subplots(2, 2, figsize=(LARGURA, 150 * MM))
    x = np.array([0, 100])
    for ax, (rot, d) in zip(eixos.ravel(), preds.items()):
        nome, cor, mk = METODOS[rot]
        eixos_drf(ax)
        ax.plot(x, x, color=BASE, lw=0.8, zorder=1)
        ax.scatter(d.y, d.pred, s=9, marker=mk, color=cor, alpha=0.35,
                   linewidths=0, zorder=2)
        b, a = reta_media(d)
        ax.plot(x, a + b * x, color=INK, lw=1.0, zorder=3)
        ax.text(3, 97, f"slope {b:.3f}", fontsize=7, color=INK2, va="top")
        ax.set_title(nome, color=INK, pad=4)
        ax.set_xlabel("Reference DRF (%)")
        ax.set_ylabel("Predicted DRF (%)")
    # chave das retas dentro do grafico, como pedem as normas da revista
    chave = [Line2D([], [], color=BASE, lw=0.8, label="Identity"),
             Line2D([], [], color=INK, lw=1.0, label="Least-squares fit")]
    eixos[0, 0].legend(handles=chave, loc="lower right", frameon=False,
                       handlelength=1.8)
    for ax, t in zip(eixos.ravel(), "abcd"):
        letra(ax, t, x=-0.12)
    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    salvar(fig, "fig2_pred_vs_ref", previas)


# --------------------------------------------------------------- figura 3
def fig3_bland_altman(preds, previas):
    fig, eixos = plt.subplots(2, 2, figsize=(LARGURA, 130 * MM), sharey=True)
    for ax, (rot, d) in zip(eixos.ravel(), preds.items()):
        nome, cor, mk = METODOS[rot]
        media, dif = (d.pred + d.y) / 2, d.pred - d.y
        vies, li, ls = ba_medio(d)
        ax.grid(True, color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.axhline(0, color=BASE, lw=0.8, zorder=1)
        ax.scatter(media, dif, s=9, marker=mk, color=cor, alpha=0.35,
                   linewidths=0, zorder=2)
        for val, lw, txt in ((vies, 1.0, f"bias {vies:+.2f}"),
                             (li, 0.8, f"−1.96 SD {li:.1f}"),
                             (ls, 0.8, f"+1.96 SD {ls:+.1f}")):
            ax.axhline(val, color=INK if lw > 0.9 else INK2, lw=lw, zorder=3)
            ax.text(99, val + 1.2, txt.replace("-", "−"), fontsize=6.5,
                    color=INK2, ha="right", va="bottom", zorder=4,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.85,
                              pad=0.6))
        ax.set_xlim(0, 100), ax.set_ylim(-55, 55)
        ax.set_xticks(range(0, 101, 20)), ax.set_yticks(range(-40, 41, 20))
        ax.set_title(nome, color=INK, pad=4)
        ax.set_xlabel("Mean of predicted and reference DRF (%)")
        ax.set_ylabel("Predicted − reference (pp)")
    for ax, t in zip(eixos.ravel(), "abcd"):
        letra(ax, t, x=-0.10)
    fig.tight_layout(w_pad=2.0, h_pad=1.6)
    salvar(fig, "fig3_bland_altman", previas)


# --------------------------------------------------------------- figura 4
def erros_por_estudo(d):
    e = d.assign(ea=(d.pred - d.y).abs(), es=d.pred - d.y)
    return e.groupby("study").agg(ea=("ea", "mean"), es=("es", "mean"),
                                  y=("y", "first"))


def fig4_faixas(preds, previas):
    ep = {rot: erros_por_estudo(d) for rot, d in preds.items()}
    idx = ep["mosaico 3 × 3"].index
    fx = pd.Series(fa.faixas_lk(ep["mosaico 3 × 3"].y), index=idx)
    n = [int((fx == g).sum()) for g in fa.FAIXAS_LK]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(LARGURA, 68 * MM))
    pos = np.arange(len(fa.FAIXAS_LK))
    desl = np.linspace(-0.24, 0.24, len(ep))
    for k, (rot, e) in enumerate(ep.items()):
        nome, cor, mk = METODOS[rot]
        e = e.loc[idx]
        med, lo, hi, vies = [], [], [], []
        for g in fa.FAIXAS_LK:
            v = e.ea[(fx == g).to_numpy()].to_numpy()
            m, a, b = fa.bootstrap_dif(v, np.zeros(len(v)))
            med.append(m), lo.append(a), hi.append(b)
            vies.append(e.es[(fx == g).to_numpy()].mean())
        med, lo, hi = map(np.array, (med, lo, hi))
        xk = pos + desl[k]
        a1.errorbar(xk, med, yerr=[med - lo, hi - med], fmt=mk, color=cor,
                    ms=4.2, mew=0, elinewidth=0.9, capsize=0, label=nome,
                    zorder=3)
        a1.plot(xk, med, color=cor, lw=0.8, alpha=0.6, zorder=2)
        a2.plot(xk, vies, marker=mk, color=cor, ms=4.2, mew=0, lw=0.8,
                label=nome, zorder=3)
    rotulos = [f"{f}\nn = {k}" for f, k in zip(FAIXAS_EN, n)]
    for ax in (a1, a2):
        ax.set_xticks(pos, rotulos)
        ax.grid(True, axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ax.set_xlabel("Reference DRF range")
    a1.set_ylim(0, 20)
    a1.set_yticks(range(0, 21, 5))
    a1.set_ylabel("MAE (pp), with 95% CI")
    a2.axhline(0, color=BASE, lw=0.8, zorder=1)
    a2.set_ylim(-13, 13)
    a2.set_ylabel("Mean signed error (pp)")
    letra(a1, "a", x=-0.10), letra(a2, "b", x=-0.10)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.06), handletextpad=0.3, columnspacing=1.4)
    fig.tight_layout(w_pad=3.0)
    salvar(fig, "fig4_error_by_range", previas)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--previas", default=None,
                   help="pasta para previas em PNG (opcional)")
    a = p.parse_args()
    preds = carregar()
    fig1_entradas(a.previas)
    fig2_predito(preds, a.previas)
    fig3_bland_altman(preds, a.previas)
    fig4_faixas(preds, a.previas)
    print("figuras em", SAIDA)


if __name__ == "__main__":
    main()
