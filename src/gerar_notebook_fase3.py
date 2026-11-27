# -*- coding: utf-8 -*-
"""Gera os notebooks do Colab embutindo src/fase3_lib.py.

  python src/gerar_notebook_fase3.py           -> notebooks/fase3_exploratorio.ipynb
  python src/gerar_notebook_fase3.py --fase4   -> notebooks/fase4_colab.ipynb
                                                  (rodada final do R(2+1)D-18)

O notebook do Colab e a execucao local (src/fase3_rodar.py) usam o MESMO
codigo: altere src/fase3_lib.py e rode este gerador de novo.
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
lib = (ROOT / "src" / "fase3_lib.py").read_text(encoding="utf-8")
FASE4 = "--fase4" in sys.argv

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

if not FASE4:
    md("""# Fase 3: rodada exploratória dos modelos espaço-temporais

Os modelos recebem o **vídeo do exame**: um tensor (3, T, 112, 112) com os quadros em
ordem temporal. Padrão: **T = 19**, um quadro por minuto (minutos 1 a 20). O controle é o
mosaico VGG19 selecionado na Fase 2, re-executado nesta mesma plataforma.

Protocolo fixo (igual à Fase 2): mesmas partições por paciente, híbridos só no treino,
reamostragem padrão, Adam lr 1e-4, early stopping 25, batch 32, augmentation **moderado**.
Precisão bf16 (a mesma no Colab e no PC local).

**Como usar:** suba `colab_renal_lk.npz` no Drive, ajuste a seção 1 e "Executar tudo".
Cada partição é gravada assim que termina (retomada automática se a sessão cair).
Resultados em `MyDrive/fase3/`: `folds_fase3.csv` (uma linha por partição),
`resultados_fase3.csv` (resumo por configuração) e `historicos_fase3/` (curvas e
predições por estudo).""")

    code("""# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
USAR_DRIVE = True
PASTA = "/content/drive/MyDrive" if USAR_DRIVE else "/content"
NPZ = f"{PASTA}/colab_renal_lk.npz"
SAIDA = f"{PASTA}/fase3"
NOME_BASE = "fase3"

FOLDS_USAR = [0, 1, 2]          # rodada exploratória em 3 partições

# Cada item é uma configuração; o que não for informado segue o padrão
# (T=19, aug moderado, BN treinável). Divisão pelo pico de memória:
# no Colab ficam os modelos que não cabem nos 8 GB da RTX 4060 local.
FILA = [
    dict(abordagem="mosaico_vgg19"),   # controle (roda também no PC local)
    dict(abordagem="r2plus1d_18"),     # 10,2 GB de pico
    dict(abordagem="vgg19_lstm"),      # 7,4 GB + overhead: estoura os 8 GB
    dict(abordagem="vgg19_media"),     # mesmo encoder do anterior
    # no PC local: r3d_18, mc3_18, s3d, x3d_s (e o controle)
]
SEMENTES = [42]
NUM_WORKERS = 2""")
else:
    md("""# Fase 4: rodada final no Colab (R(2+1)D-18 e controle)

O **R(2+1)D-18** recebe as mesmas mudanças aplicadas ao R3D-18 na Fase 3 (cabeça espacial,
saída iniciada na média do alvo e aquecimento do lr nas 3 primeiras épocas), com lr 5e-5 e
BN treinável (não cabe nos 8 GB da RTX 4060 local). O **controle** é o mosaico VGG19 3 × 3,
re-executado aqui com as mesmas sementes: números do Colab só se comparam com o controle
do Colab.

Rodada final: partições fixas `fase3-sk191`, **folds 0 a 4**, **sementes 5042, 6042, 7042 e
8042**, intercaladas por semente (se a sessão cair no meio, já há pares completos). A TTA
pela simetria é registrada em todo treino (colunas `*_tta`).

**Como usar:**
1. confira se `colab_renal_lk.npz` está na raiz do MyDrive;
2. em "Ambiente de execução > Alterar o tipo de ambiente de execução", escolha a GPU **A100**;
3. escolha `SESSAO` na seção 1: `"ambos"` roda tudo numa janela só (cerca de 2 h 15);
   `"controle"` e `"r2plus1d"` dividem a fila em duas janelas em paralelo (cerca de 45 min e
   1 h 30), cada uma gravando na sua pasta;
4. "Executar tudo" e **deixe a janela do navegador aberta, sem minimizar**, até o fim;
5. se a sessão cair, "Executar tudo" de novo: a fila retoma do ponto em que parou;
6. no fim, baixe a pasta `MyDrive/fase4_colab/` (zip) com os resultados.""")

    code("""# ============================================================
# 1. CONFIGURAÇÃO
# ============================================================
USAR_DRIVE = True
PASTA = "/content/drive/MyDrive" if USAR_DRIVE else "/content"
NPZ = f"{PASTA}/colab_renal_lk.npz"

SESSAO = "ambos"                # "ambos" | "controle" | "r2plus1d"

FOLDS_USAR = [0, 1, 2, 3, 4]
SEMENTES = [5042, 6042, 7042, 8042]       # rodada final: nunca usadas antes
NOME_BASE = "fase4"
SAIDA = f"{PASTA}/fase4_colab/{SESSAO}"   # uma pasta por sessão

CONTROLE = dict(abordagem="mosaico_vgg19")
R2PLUS1D = dict(abordagem="r2plus1d_18", cabeca="espacial", aquecimento=3,
                lr=5e-5)
FILA = {"ambos": [CONTROLE, R2PLUS1D], "controle": [CONTROLE],
        "r2plus1d": [R2PLUS1D]}[SESSAO]
NUM_WORKERS = 2""")

code("""# ============================================================
# 2. AMBIENTE
# ============================================================
import torch
if USAR_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Dispositivo:", dispositivo,
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")""")

code("# ============================================================\n"
     "# 3. BIBLIOTECA DA FASE 3 (gerada de src/fase3_lib.py; não editar aqui)\n"
     "# ============================================================\n"
     + lib)

celula_dados = """# ============================================================
# 4. DADOS E PARTIÇÕES
# ============================================================
dados = carregar_dados(NPZ)
folds = montar_folds(dados)
print(f"fase3_lib {VERSAO} | partição {PARTICAO_ID} | {len(dados['y'])} estudos "
      f"({int(dados['hibrido'].sum())} híbridos, só treino)")
for f in FOLDS_USAR:
    print(f"  fold {f}: {len(folds[f][0])} reais no treino, "
          f"{len(folds[f][1])} na validação")

# Conferência: a PDI recalibrada é determinística; com as partições corretas
# ela dá 6.77 5.50 7.37 6.21 7.99 nos folds 0 a 4, em qualquer máquina.
pdi = [referencias_pdi(dados["lk_cru"], dados["y"], tr, va)[1]["mae"]
       for tr, va in folds]
print("PDI recalibrada por fold:", " ".join(f"{v:.2f}" for v in pdi))
assert [round(v, 2) for v in pdi] == [6.77, 5.50, 7.37, 6.21, 7.99], \\
    "partições diferentes das da Fase 3: não rode a fila!\""""
if FASE4:
    versao_lib = lib.split('VERSAO = "', 1)[1].split('"', 1)[0]
    celula_dados += (f'\nassert VERSAO == "{versao_lib}", '
                     '"versão da biblioteca diferente da usada ao gerar o notebook"')
code(celula_dados)

code("""# ============================================================
# 5. FILA (retomada automática por partição; intercalada por semente)
# ============================================================
for s in SEMENTES:
    for cfg in FILA:
        resumo = rodar_configuracao(dict(cfg, seed=s), dados, folds,
                                    FOLDS_USAR, SAIDA, dispositivo,
                                    num_workers=NUM_WORKERS,
                                    nome_base=NOME_BASE)
        if resumo:
            print(f"### {resumo['tag']}: MAE {resumo['mae']} ± "
                  f"{resumo['dp_mae']} | RMSE {resumo['rmse']} | r {resumo['r']}")""")

code("""# ============================================================
# 6. RESULTADOS E CURVAS
# ============================================================
import glob
import matplotlib.pyplot as plt

res = pd.read_csv(f"{SAIDA}/resultados_{NOME_BASE}.csv").sort_values("mae")
cols = [c for c in ["tag", "n_folds", "mae", "dp_mae", "mae_tta", "rmse", "r",
                    "r2", "mae_bn_recal", "mae_pdi_recal", "minutos_por_fold",
                    "hardware"] if c in res.columns]
print(res[cols].to_string(index=False))

tags = [tag_config(dict(c, seed=s)) for c in FILA for s in SEMENTES]
fig, axes = plt.subplots(len(tags), 1, figsize=(9, 2.6 * len(tags)),
                         squeeze=False)
for ax, tag in zip(axes[:, 0], tags):
    padrao = f"{SAIDA}/historicos_{NOME_BASE}/hist_{tag}_fold*.csv"
    for arq in sorted(glob.glob(padrao)):
        h = pd.read_csv(arq)
        f = arq.rsplit("fold", 1)[1].split(".")[0]
        ax.plot(h.epoca, h.mae_tr, lw=0.9, alpha=0.6, label=f"treino f{f}")
        ax.plot(h.epoca, h.mae_va, lw=1.6, label=f"validação f{f}")
    ax.set_title(tag, fontsize=10)
    ax.set_ylabel("MAE (pp)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=3)
axes[-1, 0].set_xlabel("época")
fig.tight_layout()
plt.show()""")

nb["cells"] = cells
nb["metadata"]["language_info"] = {"name": "python"}
nb["metadata"]["colab"] = {"provenance": []}
nb["metadata"]["accelerator"] = "GPU"

out = ROOT / "notebooks" / ("fase4_colab.ipynb" if FASE4
                            else "fase3_exploratorio.ipynb")
nbf.write(nb, str(out))
print("ok ->", out)
