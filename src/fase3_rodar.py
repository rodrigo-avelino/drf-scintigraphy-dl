# -*- coding: utf-8 -*-
"""Executa a Fase 3 localmente com a mesma biblioteca do notebook do Colab.

Uso (a partir da raiz do projeto):
  .venv\\Scripts\\python src\\fase3_rodar.py --teste
      -> 2 epocas no fold 0 por abordagem: tempo por epoca e memoria de pico
  .venv\\Scripts\\python src\\fase3_rodar.py --abordagens r2plus1d_18 mc3_18 --folds 0 1 2
      -> rodada exploratoria; grava em data/fase3/ com retomada por fold
  Opcoes da investida 3D (v3.4/3.5): --seed, --bn congelado_dominio,
  --micro-lote 8, --cabeca espacial, --aquecimento 3, --lr 5e-5,
  --sem-stride-t, --limite-seg-epoca 30.
  Opcoes da Fase 4 (v3.6): --saida-media, --agendamento cosseno,
  --grid 2 3 (linhas colunas do mosaico), --nome-base fase4
  (arquivos folds_fase4.csv, resultados_fase4.csv e historicos_fase4/).
  A TTA pela simetria e registrada em todo treino (colunas *_tta).
  Opcao do artigo (v3.7): --validacao-interna (o fold seguinte escolhe lr,
  parada e melhor epoca; o fold avaliado fica fora de toda escolha; tag _vi).
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fase3_lib as f3  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

p = argparse.ArgumentParser()
p.add_argument("--teste", action="store_true")
p.add_argument("--teste-csv", default=None,
               help="grava a tabela do --teste neste CSV")
p.add_argument("--abordagens", nargs="+", default=list(f3.ABORDAGENS))
p.add_argument("--T", type=int, default=f3.PADRAO["T"])
p.add_argument("--aug", default=f3.PADRAO["aug"])
p.add_argument("--bn", default=f3.PADRAO["bn"], choices=f3.REGIMES_BN)
p.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])
p.add_argument("--seed", type=int, default=f3.PADRAO["seed"])
p.add_argument("--lr", type=float, default=f3.PADRAO["lr"])
p.add_argument("--cabeca", default=f3.PADRAO["cabeca"],
               choices=("padrao", "espacial"))
p.add_argument("--aquecimento", type=int, default=f3.PADRAO["aquecimento"])
p.add_argument("--micro-lote", type=int, default=f3.PADRAO["micro_lote"])
p.add_argument("--sem-stride-t", action="store_true")
p.add_argument("--limite-seg-epoca", type=float,
               default=f3.PADRAO["limite_seg_epoca"])
p.add_argument("--saida-media", action="store_true")
p.add_argument("--agendamento", default=f3.PADRAO["agendamento"],
               choices=tuple(f3.AGENDAMENTOS))
p.add_argument("--grid", nargs=2, type=int, default=list(f3.PADRAO["grid"]),
               metavar=("LINHAS", "COLUNAS"))
p.add_argument("--validacao-interna", action="store_true",
               help="3.7: o fold seguinte escolhe lr, parada e melhor epoca")
p.add_argument("--epocas-teste", type=int, default=2)
p.add_argument("--max-epocas", type=int, default=f3.PADRAO["max_epocas"],
               help="so para ensaios da fila (nao entra na tag)")
p.add_argument("--npz", default=str(ROOT / "data" / "colab_renal_lk.npz"))
p.add_argument("--saida", default=str(ROOT / "data" / "fase3"))
p.add_argument("--nome-base", default="fase3")
args = p.parse_args()

dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print(f"fase3_lib {f3.VERSAO} | {f3.descrever_hardware()}")

dados = f3.carregar_dados(args.npz)
folds = f3.montar_folds(dados)

opcoes = dict(aug=args.aug, bn=args.bn, seed=args.seed, lr=args.lr,
              cabeca=args.cabeca, aquecimento=args.aquecimento,
              micro_lote=args.micro_lote, sem_stride_t=args.sem_stride_t,
              limite_seg_epoca=args.limite_seg_epoca,
              saida_media=args.saida_media, agendamento=args.agendamento,
              grid=tuple(args.grid), max_epocas=args.max_epocas,
              validacao_interna=args.validacao_interna)

if args.teste:
    tab = f3.teste_rapido(args.abordagens, dados, folds, dispositivo,
                          T=args.T, epocas=args.epocas_teste, extra=opcoes)
    print("\n" + tab.to_string(index=False))
    if args.teste_csv:
        tab.to_csv(args.teste_csv, index=False)
else:
    for nome in args.abordagens:
        cfg = dict(abordagem=nome, T=args.T, **opcoes)
        resumo = f3.rodar_configuracao(cfg, dados, folds, args.folds,
                                       args.saida, dispositivo,
                                       nome_base=args.nome_base)
        if resumo:
            print(f"### {resumo['tag']}: MAE {resumo['mae']} +- "
                  f"{resumo['dp_mae']} | RMSE {resumo['rmse']} | "
                  f"r {resumo['r']}")
