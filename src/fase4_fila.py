# -*- coding: utf-8 -*-
"""Fase 4: filas desatendidas de refino e rodada final.

Fila 1 (etapas triagem, final e ablacao): triagem do caminho A (refino do
mosaico), rodada final e ablacao do caminho B (modelos 3D). Fila 2 (etapas
atribuicao, grades e final_grades): teste de atribuicao com T = 9, triagem
das grades nao quadradas do mosaico e rodada final da grade aprovada, se
houver.

As regras de decisao da fila 1 e da fila 2 foram fixadas antes dos
respectivos treinos e estao implementadas aqui exatamente como foram
fixadas; nada e ajustado depois de ver a validacao.

Roda na GPU local, um treino por vez (subprocessos de src/fase3_rodar.py,
com retomada por fold), grava em data/fase4/ (folds_fase4.csv,
resultados_fase4.csv e historicos_fase4/) e registra leituras e decisoes em
data/fase4/fase4_decisoes.json ao fim de cada etapa.

Referencias da triagem e da ablacao (ja gravadas, versao 3.5.0): controle e
R3D-18 com cabeca espacial e aquecimento, sementes 3042 e 4042, folds 0 a 2,
em data/fase3/folds_fase3.csv. Se a mesma tag e fold existirem em data/fase4
(re-execucao com a 3.6.0, opcao --recontrole), vale a gravada em data/fase4.

Uso (raiz do projeto):
  .venv\\Scripts\\python -u -X utf8 src\\fase4_fila.py --simular
      -> lista a fila e os tempos estimados, sem treinar
  .venv\\Scripts\\python -u -X utf8 src\\fase4_fila.py [--esperar-pid N]
      [--etapas triagem final ablacao] [--recontrole]
  fila 2: ... src\\fase4_fila.py --etapas atribuicao grades final_grades
  artigo (2D colapsado): ... src\\fase4_fila.py --etapas colapsado
  artigo (os tres metodos com validacao interna):
      ... src\\fase4_fila.py --etapas validacao_interna
"""
import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fase3_lib as f3  # noqa: E402
from fase3_investida import (_json, _media_completa, _r,  # noqa: E402
                             _todos_finitos, esperar)

ROOT = Path(__file__).resolve().parents[1]
PASTA = ROOT / "data" / "fase4"
NOME = "fase4"
PASTA_F3 = ROOT / "data" / "fase3"
RODAR = str(ROOT / "src" / "fase3_rodar.py")
ARQ_DECISOES = PASTA / "fase4_decisoes.json"
LIMITE_SEG = 30.0        # s por epoca; acima disso, transbordo da VRAM

F3 = (0, 1, 2)           # triagem e ablacao
F5 = (0, 1, 2, 3, 4)     # rodada final
SEM_TRIAGEM = (3042, 4042)
SEM_FINAL = (5042, 6042, 7042, 8042)     # nunca usadas antes

LIMIAR_TRIAGEM = -0.25   # triagem: delta medio <= -0,25 pp ...
VIT_TRIAGEM = 2          #      ... e delta < 0 em >= 2 de 3 folds
LIMIAR_FINAL_A = -0.3    # rodada final (mosaico refinado): delta medio <= -0,3 pp ...
VIT_FINAL_A = 4          #      ... e delta < 0 em >= 4 de 5 folds
LIMIAR_3D = -1.0         # 3D contra o mosaico: delta < -1 pp ou 5 de 5 folds
                         # (regra fixada antes das execucoes, descrita no artigo)
LIMIAR_ABLACAO = 0.25    # ablacao: |efeito| >= 0,25 pp, mesmo sinal em >= 2 de 3
MAE_EP1_MAX = 20.0       # teste de 2 epocas: sanidade da saida iniciada na media

CTL = dict(abordagem="mosaico_vgg19")
R3D_REF = dict(abordagem="r3d_18", cabeca="espacial", aquecimento=3)
CANDIDATOS_A = {
    "A1_aq3": dict(CTL, aquecimento=3),
    "A2_sm": dict(CTL, saida_media=True),
    "A3_aq3_sm": dict(CTL, aquecimento=3, saida_media=True),
    "A4_aq3_cos": dict(CTL, aquecimento=3, agendamento="cosseno"),
}
ABLACAO_B = {
    "B1_cab_espacial_sem_aq": dict(abordagem="r3d_18", cabeca="espacial"),
    "B2_cab_padrao_sm_aq3": dict(abordagem="r3d_18", aquecimento=3,
                                 saida_media=True),
    "B3_r3d_exploratorio": dict(abordagem="r3d_18"),
}
# efeito = MAE com o fator - MAE sem o fator (negativo: o fator reduz o erro)
LEITURAS_B = {
    "aquecimento_dada_a_cabeca_espacial":
        ("referencia", "B1_cab_espacial_sem_aq"),
    "cabeca_espacial_em_si": ("referencia", "B2_cab_padrao_sm_aq3"),
    "saida_media_e_aquecimento_com_media_global":
        ("B2_cab_padrao_sm_aq3", "B3_r3d_exploratorio"),
}
# ---- fila 2 (atribuicao com T = 9 e grades; regras fixadas antes de rodar)
# atribuicao: mesmo 3D da rodada final com T = 9, a mesma agregacao temporal
# das celulas do mosaico 3 x 3; comparado com os bracos da rodada final
R3D_T9 = dict(R3D_REF, T=9)
LIMIAR_ATRIB = 0.25      # atribuicao: |delta medio| >= 0,25 pp ...
VIT_ATRIB = 4            #      ... com o mesmo sinal em >= 4 de 5 folds
# grades: celulas quadradas e mesma area; receita do controle (fixada depois
# da fila 1); triagem com a regra do caminho A, contra o 3 x 3 gravado
GRADES = {f"G_{r}x{c}": dict(CTL, grid=(r, c))
          for r, c in ((2, 3), (3, 2), (3, 4), (4, 3))}

# ---- Colab: R(2+1)D-18 contra o controle do Colab, mesmas sementes e
# folds; a pasta espelha MyDrive/fase4_colab/<SESSAO>/
R21_COLAB = dict(abordagem="r2plus1d_18", cabeca="espacial", aquecimento=3,
                 lr=5e-5)

# ---- artigo: linha de base 2D colapsada, a soma dos 19 quadros depois do
# 1o minuto numa unica imagem (grade 1 x 1), com a receita do controle;
# rodada final, descritiva, sem regra de decisao
COLAPSADO = dict(CTL, grid=(1, 1))
PASTA_COLAB = ROOT / "data" / "fase4_colab" / "ambos"

# ---- artigo: os tres metodos com validacao interna (lib 3.7): em cada
# fold f, o fold seguinte, (f + 1) % 5, escolhe lr, parada e melhor epoca,
# e o fold f fica fora de toda escolha; mesmas receitas, folds e sementes
# da rodada final; tags com _vi
VI = dict(validacao_interna=True)
BRACOS_VI = {"mosaico_3x3_vi": dict(CTL, **VI),
             "r3d_cab_espacial_aq3_vi": dict(R3D_REF, **VI),
             "colapsado_1x1_vi": dict(COLAPSADO, **VI)}

# minutos por fold na 4060 (Fase 3) e por teste de 2 epocas, para --simular;
# T = 9: media dos T = 9 da Fase 3 (cabeca padrao) e estimativa para a espacial
MIN_FOLD = dict(mosaico=3.8, mosaico_cos=5.6, r3d_cab=6.0, r3d_padrao=7.6,
                r3d_cab_T9=3.5, r3d_padrao_T9=4.4)
MIN_TESTE = 1.5


# ------------------------------------------------------------- leitura
class Leitor:
    """Le os CSV por fold (particoes fixas); a primeira fonte tem prioridade."""

    def __init__(self, fontes=None):
        if fontes is None:       # resolvido na chamada (os ensaios trocam PASTA)
            fontes = ((PASTA, NOME), (PASTA_F3, "fase3"))
        self.fontes = [(Path(p), n) for p, n in fontes]

    def _dfs(self):
        out = []
        for pasta, nome in self.fontes:
            arq = pasta / f"folds_{nome}.csv"
            if arq.exists():
                df = pd.read_csv(arq)
                out.append((nome, df[df.particao == f3.PARTICAO_ID]))
        return out

    def linha(self, cfg, fold):
        tag = f3.tag_config(cfg)
        for nome, df in self._dfs():
            d = df[(df.tag == tag) & (df.fold.astype(int) == int(fold))]
            if len(d):
                lin = d.iloc[-1].copy()
                lin["fonte"] = nome
                return lin
        return None

    def valor(self, cfg, fold, coluna="mae"):
        lin = self.linha(cfg, fold)
        if lin is None or coluna not in lin.index or pd.isna(lin[coluna]):
            return np.nan
        return float(lin[coluna])

    def mae(self, cfg, fold):
        return self.valor(cfg, fold, "mae")

    def fontes_de(self, cfg, sementes, folds):
        """Origem e versao das linhas usadas (para o registro)."""
        vistos = set()
        for s in sementes:
            for f in folds:
                lin = self.linha(dict(cfg, seed=s), f)
                if lin is not None:
                    vistos.add(f"{lin['fonte']} {lin.get('versao', '?')}")
        return sorted(vistos)


def _por_fold(L, cfg, sementes, folds, coluna="mae"):
    """Valor por fold (media das sementes; NaN se faltar alguma) e por
    semente."""
    por_semente = {s: [L.valor(dict(cfg, seed=s), f, coluna) for f in folds]
                   for s in sementes}
    por_fold = [_media_completa([por_semente[s][i] for s in sementes])
                for i in range(len(folds))]
    return por_fold, por_semente


# ------------------------------------------------------------ decisoes
def decidir_triagem(L, candidatos=None):
    """Regra da triagem: passa se delta medio <= -0,25 e delta < 0 em >= 2
    de 3 folds; finalista = menor delta medio entre as que passarem.
    Candidatos: caminho A (padrao) ou as grades nao quadradas (GRADES),
    sempre contra o controle 3 x 3 gravado."""
    candidatos = CANDIDATOS_A if candidatos is None else candidatos
    ref, ref_sem = _por_fold(L, CTL, SEM_TRIAGEM, F3)
    res = {}
    for nome, cfg in candidatos.items():
        pf, ps = _por_fold(L, cfg, SEM_TRIAGEM, F3)
        dl = [a - b for a, b in zip(pf, ref)]
        ok = _todos_finitos(dl)
        md = _r(np.mean(dl)) if ok else np.nan
        vit = int(sum(x < 0 for x in dl)) if ok else 0
        res[nome] = dict(
            tag=f3.tag_config(cfg), mae_por_semente=ps, mae_por_fold=pf,
            mae_tta_por_fold=_por_fold(L, cfg, SEM_TRIAGEM, F3, "mae_tta")[0],
            deltas=dl, delta_medio=md, vitorias=vit,
            # fold faltante (treino que falhou duas vezes) nao passa
            passa=bool(ok and md <= LIMIAR_TRIAGEM and vit >= VIT_TRIAGEM),
            status="completo" if ok else "incompleto (nao passa)")
    passaram = [n for n in res if res[n]["passa"]]
    fin = (min(passaram, key=lambda n: res[n]["delta_medio"]) if passaram
           else None)
    return dict(regra=f"delta medio <= {LIMIAR_TRIAGEM} e vitoria em >= "
                      f"{VIT_TRIAGEM} de 3 folds",
                controle_ref_por_fold=ref, controle_ref_por_semente=ref_sem,
                fontes_controle=L.fontes_de(CTL, SEM_TRIAGEM, F3),
                configuracoes=res, passaram=passaram, finalista=fin)


def decidir_final(L, cfg_a=None):
    """Regras da rodada final: mosaico refinado (delta <= -0,3 e >= 4 de 5
    folds) e 3D pela regra fixada antes das execucoes (delta < -1 pp ou 5 de
    5 folds; descrita no artigo)."""
    bracos = {"controle": CTL, "r3d_cab_espacial_aq3": R3D_REF}
    if cfg_a is not None:
        bracos["finalista_A"] = cfg_a
    tab = {}
    for nome, cfg in bracos.items():
        pf, ps = _por_fold(L, cfg, SEM_FINAL, F5)
        pt, _ = _por_fold(L, cfg, SEM_FINAL, F5, "mae_tta")
        tab[nome] = dict(tag=f3.tag_config(cfg), mae_por_fold=pf,
                         mae_por_semente=ps, mae_medio=_media_completa(pf),
                         mae_tta_por_fold=pt, mae_tta_medio=_media_completa(pt))
    ctl = tab["controle"]["mae_por_fold"]
    comp = {}
    for nome in bracos:
        if nome == "controle":
            continue
        dl = [a - b for a, b in zip(tab[nome]["mae_por_fold"], ctl)]
        ok = _todos_finitos(dl)
        md = _r(np.mean(dl)) if ok else np.nan
        vit = int(sum(x < 0 for x in dl)) if ok else None
        if not ok:           # falta de dado nao e empate: a regra nao se aplica
            veredito = "incompleto"
        elif nome == "finalista_A":
            veredito = ("mosaico refinado passa a ser a referencia"
                        if md <= LIMIAR_FINAL_A and vit >= VIT_FINAL_A
                        else "controle segue como referencia")
        else:
            veredito = ("3D vence" if md < LIMIAR_3D or vit == len(F5)
                        else "empate tecnico")
        comp[nome] = dict(deltas=dl, delta_medio=md, vitorias=vit,
                          delta_folds_3_4=_r(np.mean(dl[3:5])) if ok
                          else np.nan, veredito=veredito)
    return dict(bracos=tab, comparacoes_com_o_controle=comp)


def decidir_ablacao(L):
    """Leituras da ablacao (descritivas): efeito detectado se |efeito medio|
    >= 0,25 pp, com o mesmo sinal em pelo menos 2 dos 3 folds."""
    pf = {"referencia": _por_fold(L, R3D_REF, SEM_TRIAGEM, F3)[0]}
    for nome, cfg in ABLACAO_B.items():
        pf[nome] = _por_fold(L, cfg, SEM_TRIAGEM, F3)[0]
    leituras = {}
    for nome, (com, sem) in LEITURAS_B.items():
        ef = [a - b for a, b in zip(pf[com], pf[sem])]
        if not _todos_finitos(ef):
            leituras[nome] = dict(com=com, sem=sem, efeito_por_fold=ef,
                                  leitura="incompleto")
            continue
        me = _r(np.mean(ef))
        mesmo_sinal = int(sum(np.sign(x) == np.sign(me) for x in ef))
        if abs(me) >= LIMIAR_ABLACAO and mesmo_sinal >= 2:
            leitura = ("detectado: o fator reduz o erro" if me < 0
                       else "detectado: o fator aumenta o erro")
        else:
            leitura = "nao detectado"
        leituras[nome] = dict(com=com, sem=sem, efeito_por_fold=ef,
                              efeito_medio=me, folds_com_o_mesmo_sinal=mesmo_sinal,
                              leitura=leitura)
    return dict(mae_por_fold=pf,
                controle_por_fold=_por_fold(L, CTL, SEM_TRIAGEM, F3)[0],
                fontes_referencia=L.fontes_de(R3D_REF, SEM_TRIAGEM, F3),
                leituras=leituras)


def _braco(L, cfg):
    """MAE por fold e por semente nas condicoes da rodada final."""
    pf, ps = _por_fold(L, cfg, SEM_FINAL, F5)
    pt, _ = _por_fold(L, cfg, SEM_FINAL, F5, "mae_tta")
    return dict(tag=f3.tag_config(cfg), mae_por_fold=pf, mae_por_semente=ps,
                mae_medio=_media_completa(pf), mae_tta_por_fold=pt,
                mae_tta_medio=_media_completa(pt))


def _comparar(pf, ref):
    """Delta por fold (braco - referencia), delta medio e folds vencidos;
    devolve tambem se todos os folds estao completos."""
    dl = [a - b for a, b in zip(pf, ref)]
    ok = _todos_finitos(dl)
    return dict(deltas=dl, delta_medio=_r(np.mean(dl)) if ok else np.nan,
                vitorias=int(sum(x < 0 for x in dl)) if ok else None,
                delta_folds_3_4=_r(np.mean(dl[3:5])) if ok else np.nan), ok


def decidir_atribuicao(L):
    """Leitura da atribuicao (descritiva): T = 9 contra o T = 19 e contra o
    mosaico 3 x 3, nos bracos da rodada final. Efeito detectado se |delta
    medio| >= 0,25 pp, com o mesmo sinal em pelo menos 4 dos 5 folds."""
    tab = {"r3d_T9": _braco(L, R3D_T9), "r3d_T19": _braco(L, R3D_REF),
           "mosaico_3x3": _braco(L, CTL)}
    leituras = {}
    for nome, ref in (("T9_menos_T19", "r3d_T19"),
                      ("T9_menos_mosaico", "mosaico_3x3")):
        ef = [a - b for a, b in zip(tab["r3d_T9"]["mae_por_fold"],
                                    tab[ref]["mae_por_fold"])]
        if not _todos_finitos(ef):
            leituras[nome] = dict(efeito_por_fold=ef, efeito_medio=np.nan,
                                  detectado=None, leitura="incompleto")
            continue
        me = _r(np.mean(ef))
        mesmo_sinal = int(sum(np.sign(x) == np.sign(me) for x in ef))
        det = bool(abs(me) >= LIMIAR_ATRIB and mesmo_sinal >= VIT_ATRIB)
        leituras[nome] = dict(
            efeito_por_fold=ef, efeito_medio=me,
            folds_com_o_mesmo_sinal=mesmo_sinal, detectado=det,
            efeito_folds_3_4=_r(np.mean(ef[3:5])),
            leitura=("detectado: T = 9 erra menos" if det and me < 0 else
                     "detectado: T = 9 erra mais" if det else "nao detectado"))
    lt, lm = leituras["T9_menos_T19"], leituras["T9_menos_mosaico"]
    if "incompleto" in (lt["leitura"], lm["leitura"]):
        interp = "incompleto"
    else:
        mantido = lm["detectado"] and lm["efeito_medio"] < 0
        dos_19 = lt["detectado"] and lt["efeito_medio"] > 0
        interp = ("ganho do 3D mantido com os 9 momentos do mosaico"
                  if mantido and not dos_19 else
                  "ganho do 3D dependente dos 19 quadros"
                  if dos_19 and not mantido else "inconclusivo")
    return dict(regra=f"efeito detectado se |delta medio| >= {LIMIAR_ATRIB} "
                      f"com o mesmo sinal em >= {VIT_ATRIB} de 5 folds",
                bracos=tab, leituras=leituras, interpretacao=interp)


def decidir_final_grade(L, cfg_g):
    """Grades do mosaico: a grade finalista passa a ser a referencia do
    mosaico pela regra do mosaico refinado na rodada final (delta <= -0,3 e
    < 0 em >= 4 de 5 folds, contra o 3 x 3 da rodada final). Se passar, o 3D
    e comparado com ela pela regra do 3D (delta < -1 pp ou 5 de 5), sem
    treinos novos."""
    tab = {"controle_3x3": _braco(L, CTL), "grade": _braco(L, cfg_g),
           "r3d_cab_espacial_aq3": _braco(L, R3D_REF)}
    c, ok = _comparar(tab["grade"]["mae_por_fold"],
                      tab["controle_3x3"]["mae_por_fold"])
    passa = bool(ok and c["delta_medio"] <= LIMIAR_FINAL_A
                 and c["vitorias"] >= VIT_FINAL_A)
    c["veredito"] = ("incompleto" if not ok else
                     "grade passa a ser a referencia do mosaico" if passa
                     else "3x3 segue como referencia")
    out = dict(bracos=tab, grade_contra_3x3=c)
    if passa:
        c3, ok3 = _comparar(tab["r3d_cab_espacial_aq3"]["mae_por_fold"],
                            tab["grade"]["mae_por_fold"])
        c3["veredito"] = ("incompleto" if not ok3 else "3D vence"
                          if c3["delta_medio"] < LIMIAR_3D
                          or c3["vitorias"] == len(F5) else "empate tecnico")
        out["r3d_contra_a_grade"] = c3
    return out


def decidir_colapsado(L):
    """Artigo: linha de base 2D colapsada na rodada final.
    Descritiva, sem regra de decisao: delta por fold do colapsado contra o
    mosaico 3 x 3 e contra o R3D-18 de referencia."""
    tab = {"colapsado_1x1": _braco(L, COLAPSADO), "mosaico_3x3": _braco(L, CTL),
           "r3d_cab_espacial_aq3": _braco(L, R3D_REF)}
    comp = {}
    for nome in ("mosaico_3x3", "r3d_cab_espacial_aq3"):
        c, ok = _comparar(tab["colapsado_1x1"]["mae_por_fold"],
                          tab[nome]["mae_por_fold"])
        c["status"] = "completo" if ok else "incompleto"
        comp[f"colapsado_menos_{nome}"] = c
    return dict(bracos=tab, comparacoes=comp)


def decidir_validacao_interna(L):
    """Artigo: os tres metodos com validacao interna. R3D-18 contra o
    mosaico pela regra fixada antes das execucoes (delta < -1 pp ou 5 de 5
    folds; caso contrario, empate tecnico; descrita no artigo), como em
    decidir_final; o 2D colapsado e descritivo (delta contra os dois)."""
    tab = {nome: _braco(L, cfg) for nome, cfg in BRACOS_VI.items()}
    comp = {}
    c, ok = _comparar(tab["r3d_cab_espacial_aq3_vi"]["mae_por_fold"],
                      tab["mosaico_3x3_vi"]["mae_por_fold"])
    if not ok:               # falta de dado nao e empate: a regra nao se aplica
        c["veredito"] = "incompleto"
    else:
        c["veredito"] = ("3D vence" if c["delta_medio"] < LIMIAR_3D
                         or c["vitorias"] == len(F5) else "empate tecnico")
    comp["r3d_menos_mosaico"] = c
    for nome in ("mosaico_3x3_vi", "r3d_cab_espacial_aq3_vi"):
        c, ok = _comparar(tab["colapsado_1x1_vi"]["mae_por_fold"],
                          tab[nome]["mae_por_fold"])
        c["status"] = "completo" if ok else "incompleto"
        comp[f"colapsado_menos_{nome}"] = c
    return dict(bracos=tab, comparacoes=comp)


def decidir_colab(L):
    """Colab: R(2+1)D-18 contra o controle do Colab, pela regra do 3D
    (delta < -1 pp ou 5 de 5 folds). L deve ler so a pasta do Colab:
    numeros do Colab nunca sao pareados com os da 4060."""
    tab = {"controle_colab": _braco(L, CTL),
           "r2plus1d_cab_espacial_aq3": _braco(L, R21_COLAB)}
    c, ok = _comparar(tab["r2plus1d_cab_espacial_aq3"]["mae_por_fold"],
                      tab["controle_colab"]["mae_por_fold"])
    c["veredito"] = ("incompleto" if not ok else "3D vence"
                     if c["delta_medio"] < LIMIAR_3D
                     or c["vitorias"] == len(F5) else "empate tecnico")
    return dict(bracos=tab, r2plus1d_contra_o_controle=c)


# ------------------------------------------------------------- execucao
FALHAS = []    # treinos e testes que falharam (registrados nas decisoes)


def avisar(msg):
    print(f"[fase4 {datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def args_cli(cfg):
    a = ["--abordagens", cfg["abordagem"]]
    for chave, flag in (("T", "--T"), ("bn", "--bn"), ("seed", "--seed"),
                        ("lr", "--lr"), ("cabeca", "--cabeca"),
                        ("aquecimento", "--aquecimento"),
                        ("micro_lote", "--micro-lote"),
                        ("agendamento", "--agendamento")):
        if cfg.get(chave) is not None:
            a += [flag, repr(cfg[chave]) if chave == "lr" else str(cfg[chave])]
    if cfg.get("sem_stride_t"):
        a.append("--sem-stride-t")
    if cfg.get("saida_media"):
        a.append("--saida-media")
    if cfg.get("validacao_interna"):
        a.append("--validacao-interna")
    if cfg.get("grid") is not None:
        a += ["--grid", str(cfg["grid"][0]), str(cfg["grid"][1])]
    return a


def cmd_rodar(cfg, folds):
    return [sys.executable, "-u", "-X", "utf8", RODAR, *args_cli(cfg),
            "--folds", *map(str, folds), "--limite-seg-epoca", str(LIMITE_SEG),
            "--saida", str(PASTA), "--nome-base", NOME]


def _executar(cmd, rotulo):
    with open(PASTA / f"log_fase4_{rotulo}.txt", "a", encoding="utf-8") as fo:
        return subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT,
                              cwd=ROOT).returncode


def rodar(cfg, folds, rotulo):
    """Treina os folds que faltam em data/fase4; uma nova tentativa se o
    processo falhar (a retomada por fold refaz so o que faltou)."""
    so_fase4 = Leitor(fontes=((PASTA, NOME),))
    if _todos_finitos([so_fase4.mae(cfg, f) for f in folds]):
        avisar(f"{rotulo}: {f3.tag_config(cfg)} ja completo, pulando")
        return 0
    rc = None
    for tentativa in (1, 2):
        t = time.time()
        rc = _executar(cmd_rodar(cfg, folds), rotulo)
        avisar(f"{rotulo}: {f3.tag_config(cfg)} folds {list(folds)} -> codigo "
               f"{rc} ({(time.time() - t) / 60:.1f} min, tentativa "
               f"{tentativa})")
        if rc == 0:
            return 0
        time.sleep(60)          # tempo para o driver se recuperar
    FALHAS.append(dict(rotulo=rotulo, tag=f3.tag_config(cfg),
                       folds=list(folds), codigo=rc))
    return rc


def testar(cfg, rotulo, dec):
    """Teste de sanidade: 2 epocas no fold 0, sem gravar, uma vez por braco."""
    anterior = dec["testes"].get(rotulo)
    if anterior and anterior.get("ok"):
        return True
    csv = PASTA / f"teste_fase4_{rotulo}.csv"
    rc = None
    for _ in (1, 2):
        if csv.exists():
            csv.unlink()
        rc = _executar([sys.executable, "-u", "-X", "utf8", RODAR, "--teste",
                        "--teste-csv", str(csv), *args_cli(cfg)],
                       f"teste_{rotulo}")
        if rc == 0 and csv.exists():
            break
        time.sleep(60)
    if rc != 0 or not csv.exists():
        t = dict(ok=False, erro=f"codigo {rc}")
    else:
        t = pd.read_csv(csv).iloc[0].to_dict()
        t["ok"] = bool(t["ok"]) and float(t["seg_por_epoca"]) <= LIMITE_SEG
        if t["ok"] and (cfg.get("saida_media") or cfg.get("cabeca") == "espacial"):
            # sanidade da saida iniciada na media: 1a epoca no nivel da
            # predicao constante (~13 a 14 pp) ou abaixo
            t["ok"] = float(t["mae_val_ep1"]) <= MAE_EP1_MAX
            if not t["ok"]:
                t["erro"] = f"MAE da 1a epoca {t['mae_val_ep1']} > {MAE_EP1_MAX}"
    erro = t.get("erro")
    avisar(f"teste {rotulo}: ok={t['ok']} | {t.get('seg_por_epoca')} s/epoca | "
           f"pico {t.get('pico_mem_gb')} GB | MAE val ep1 {t.get('mae_val_ep1')}"
           f"{'' if pd.isna(erro) or erro == '' else ' | ' + str(erro)}")
    dec["testes"][rotulo] = t
    if not t["ok"]:
        FALHAS.append(dict(rotulo=f"teste_{rotulo}", tag=f3.tag_config(cfg),
                           folds=[0], codigo=str(t.get("erro"))))
    return t["ok"]


def gravar(dec):
    try:
        dec["atualizado"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        dec["falhas"] = list(FALHAS)
        with open(ARQ_DECISOES, "w", encoding="utf-8") as fo:
            json.dump(_json(dec), fo, ensure_ascii=False, indent=1)
    except Exception as e:      # ex.: arquivo aberto em outro programa
        avisar(f"ERRO ao gravar decisoes: {type(e).__name__}: {e}")


def _decidir(func, *a, rotulo="", reserva=None):
    try:
        return func(*a)
    except Exception as e:      # a fila nao pode parar por erro de leitura
        avisar(f"ERRO na decisao {rotulo}: {type(e).__name__}: {e}")
        return dict(reserva or {}, erro=f"{type(e).__name__}: {e}")


def _etapa(nome, func):
    """Executa uma etapa sem deixar uma excecao derrubar a fila inteira."""
    try:
        func()
    except Exception as e:
        avisar(f"ERRO na etapa {nome}: {type(e).__name__}: {e}")
        FALHAS.append(dict(rotulo=f"etapa_{nome}", tag="", folds=[],
                           codigo=f"{type(e).__name__}: {e}"))


# ------------------------------------------------------------- simulacao
def _min_fold(cfg):
    if cfg["abordagem"] == "mosaico_vgg19":
        return MIN_FOLD["mosaico_cos" if cfg.get("agendamento") == "cosseno"
                        else "mosaico"]
    chave = "r3d_cab" if cfg.get("cabeca") == "espacial" else "r3d_padrao"
    if cfg.get("T", f3.PADRAO["T"]) == 9:
        chave += "_T9"
    return MIN_FOLD[chave]


def simular(etapas, recontrole):
    linhas, total, condicional = [], 0.0, 0.0

    def item(etapa, nome, cfg, folds, sementes, teste=True, se_houver=False):
        nonlocal total, condicional
        m = _min_fold(cfg) * len(folds) * len(sementes) + (MIN_TESTE if teste
                                                           else 0)
        total += m
        condicional += m if se_houver else 0
        linhas.append((etapa, nome, f3.tag_config(cfg),
                       "-".join(map(str, folds)),
                       ",".join(map(str, sementes)), len(folds) * len(sementes),
                       round(m)))

    if "triagem" in etapas:
        if recontrole:
            item("triagem", "recontrole", CTL, F3, SEM_TRIAGEM, teste=False)
        for nome, cfg in CANDIDATOS_A.items():
            item("triagem", nome, cfg, F3, SEM_TRIAGEM)
    if "final" in etapas:
        for nome, cfg in (("controle", CTL), ("r3d_cab_espacial_aq3", R3D_REF)):
            item("final", nome, cfg, F5, SEM_FINAL)
        item("final", "finalista_A (se houver; estimado como mosaico)", CTL,
             F5, SEM_FINAL, se_houver=True)
    if "ablacao" in etapas:
        for nome, cfg in ABLACAO_B.items():
            item("ablacao", nome, cfg, F3, SEM_TRIAGEM)
    if "atribuicao" in etapas:
        item("atribuicao", "r3d_T9", R3D_T9, F5, SEM_FINAL)
    if "grades" in etapas:
        for nome, cfg in GRADES.items():
            item("grades", nome, cfg, F3, SEM_TRIAGEM)
    if "final_grades" in etapas:
        item("final_grades", "grade finalista (se houver; estimada como 2x3)",
             GRADES["G_2x3"], F5, SEM_FINAL, se_houver=True)
    if "colapsado" in etapas:
        item("colapsado", "colapsado_1x1", COLAPSADO, F5, SEM_FINAL)
    if "validacao_interna" in etapas:
        for nome, cfg in BRACOS_VI.items():
            item("validacao_interna", nome, cfg, F5, SEM_FINAL)
    tab = pd.DataFrame(linhas, columns=["etapa", "braco", "tag (sem semente)",
                                        "folds", "sementes", "treinos",
                                        "min"])
    print(tab.to_string(index=False))
    print(f"\nTotal: {tab.treinos.sum()} treinos, cerca de {total / 60:.1f} h "
          f"(sem os itens condicionais: {(total - condicional) / 60:.1f} h), "
          "mais cerca de 15 s por subprocesso para carregar dados e pesos.")
    cfg_ex, folds_ex = ((dict(R3D_T9, seed=SEM_FINAL[0]), F5)
                        if "atribuicao" in etapas
                        else (dict(CTL, seed=SEM_FINAL[0]), F5))
    print("\nExemplo de comando:")
    print(" ".join(cmd_rodar(cfg_ex, folds_ex)))


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esperar-pid", type=int, default=None)
    p.add_argument("--etapas", nargs="+", default=["triagem", "final",
                                                   "ablacao"],
                   choices=["triagem", "final", "ablacao", "atribuicao",
                            "grades", "final_grades", "colapsado",
                            "validacao_interna"],
                   help="as etapas rodam sempre na ordem do codigo; fila 2: "
                        "atribuicao grades final_grades; artigo: colapsado "
                        "e validacao_interna")
    p.add_argument("--recontrole", action="store_true",
                   help="re-executa com a 3.6.0 o controle da triagem "
                        "(sementes 3042 e 4042, folds 0 a 2) antes dos "
                        "candidatos")
    p.add_argument("--simular", action="store_true")
    a = p.parse_args()
    if a.simular:
        simular(a.etapas, a.recontrole)
        return
    PASTA.mkdir(parents=True, exist_ok=True)
    if sys.platform == "win32":
        import ctypes   # impede a suspensao por inatividade enquanto roda
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if a.esperar_pid:
        esperar(a.esperar_pid)
    L = Leitor()
    dec = {}
    if ARQ_DECISOES.exists():       # retomada: preserva as etapas anteriores
        dec = json.loads(ARQ_DECISOES.read_text(encoding="utf-8"))
        FALHAS.extend(dec.get("falhas", []))
    dec.setdefault("testes", {})
    dec["versao_lib"] = f3.VERSAO
    dec.setdefault("execucoes", []).append(dict(
        inicio=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        etapas=a.etapas, recontrole=a.recontrole))
    gravar(dec)

    # ---------------- 1. triagem do caminho A (folds 0 a 2, 3042 e 4042)
    if "triagem" in a.etapas:
        avisar("ETAPA 1: triagem do caminho A")

        def triagem():
            if a.recontrole:
                for s in SEM_TRIAGEM:
                    rodar(dict(CTL, seed=s), F3, f"recontrole_s{s}")
            for nome, cfg in CANDIDATOS_A.items():
                if testar(cfg, nome, dec):
                    for s in SEM_TRIAGEM:
                        rodar(dict(cfg, seed=s), F3, f"triagem_{nome}_s{s}")
                gravar(dec)
        _etapa("triagem", triagem)
        dec["triagem"] = _decidir(decidir_triagem, L, rotulo="triagem",
                                  reserva=dict(finalista=None))
        gravar(dec)
        avisar(f"TRIAGEM: passaram {dec['triagem'].get('passaram')} | "
               f"finalista = {dec['triagem'].get('finalista')}")

    fin = dec.get("triagem", {}).get("finalista")
    cfg_a = CANDIDATOS_A.get(fin) if fin else None

    # ---------------- 2. rodada final (folds 0 a 4, 5042 a 8042)
    if "final" in a.etapas:
        avisar(f"ETAPA 2: rodada final (finalista do caminho A: {fin})")
        bracos = [("controle", CTL), ("r3d_cab_espacial_aq3", R3D_REF)]
        if cfg_a is not None:
            bracos.append(("finalista_A", cfg_a))

        def final():
            ok = {nome: testar(cfg, f"final_{nome}", dec)
                  for nome, cfg in bracos}
            gravar(dec)
            for s in SEM_FINAL:       # intercalado: conjuntos pareados completos
                for nome, cfg in bracos:
                    if ok[nome]:
                        rodar(dict(cfg, seed=s), F5, f"final_{nome}_s{s}")
                gravar(dec)
        _etapa("final", final)
        dec["final"] = _decidir(decidir_final, L, cfg_a, rotulo="final")
        dec["final"]["finalista_A"] = fin
        gravar(dec)
        for nome, c in dec["final"].get("comparacoes_com_o_controle",
                                        {}).items():
            avisar(f"FINAL {nome}: delta medio {c.get('delta_medio')} | "
                   f"vitorias {c.get('vitorias')} | {c.get('veredito')}")

    # ---------------- 3. ablacao do caminho B (folds 0 a 2, 3042 e 4042)
    if "ablacao" in a.etapas:
        avisar("ETAPA 3: ablacao do caminho B")

        def ablacao():
            for nome, cfg in ABLACAO_B.items():
                if testar(cfg, nome, dec):
                    for s in SEM_TRIAGEM:
                        rodar(dict(cfg, seed=s), F3, f"ablacao_{nome}_s{s}")
                gravar(dec)
        _etapa("ablacao", ablacao)
        dec["ablacao"] = _decidir(decidir_ablacao, L, rotulo="ablacao")
        gravar(dec)
        for nome, l in dec["ablacao"].get("leituras", {}).items():
            avisar(f"ABLACAO {nome}: {l.get('efeito_medio')} -> "
                   f"{l.get('leitura')}")

    # ---------------- 4. atribuicao com T = 9 (fila 2; folds 0 a 4)
    if "atribuicao" in a.etapas:
        avisar("ETAPA 4: atribuicao com T = 9")

        def atribuicao():
            if testar(R3D_T9, "atrib_r3d_T9", dec):
                gravar(dec)
                for s in SEM_FINAL:
                    rodar(dict(R3D_T9, seed=s), F5, f"atrib_r3d_T9_s{s}")
                    gravar(dec)
            gravar(dec)
        _etapa("atribuicao", atribuicao)
        dec["atribuicao"] = _decidir(decidir_atribuicao, L,
                                     rotulo="atribuicao")
        gravar(dec)
        for nome, l in dec["atribuicao"].get("leituras", {}).items():
            avisar(f"ATRIBUICAO {nome}: {l.get('efeito_medio')} -> "
                   f"{l.get('leitura')}")
        avisar(f"ATRIBUICAO: {dec['atribuicao'].get('interpretacao')}")

    # ---------------- 5. triagem das grades (fila 2; folds 0 a 2)
    if "grades" in a.etapas:
        avisar("ETAPA 5: triagem das grades do mosaico")

        def grades():
            for nome, cfg in GRADES.items():
                if testar(cfg, nome, dec):
                    for s in SEM_TRIAGEM:
                        rodar(dict(cfg, seed=s), F3, f"grades_{nome}_s{s}")
                gravar(dec)
        _etapa("grades", grades)
        dec["triagem_grades"] = _decidir(decidir_triagem, L, GRADES,
                                         rotulo="triagem_grades",
                                         reserva=dict(finalista=None))
        gravar(dec)
        avisar(f"GRADES: passaram {dec['triagem_grades'].get('passaram')} | "
               f"finalista = {dec['triagem_grades'].get('finalista')}")

    fin_g = dec.get("triagem_grades", {}).get("finalista")
    cfg_g = GRADES.get(fin_g) if fin_g else None

    # ---------------- 6. rodada final da grade aprovada (folds 0 a 4)
    if "final_grades" in a.etapas:
        if cfg_g is None:
            avisar("ETAPA 6: sem grade finalista; a rodada final das grades "
                   "nao roda")
            dec["final_grades"] = dict(finalista=None)
        else:
            avisar(f"ETAPA 6: rodada final da grade {fin_g}")

            def final_grade():
                if testar(cfg_g, "final_grade", dec):
                    gravar(dec)
                    for s in SEM_FINAL:
                        rodar(dict(cfg_g, seed=s), F5,
                              f"final_grade_{fin_g}_s{s}")
                        gravar(dec)
            _etapa("final_grades", final_grade)
            dec["final_grades"] = _decidir(decidir_final_grade, L, cfg_g,
                                           rotulo="final_grades")
            dec["final_grades"]["finalista"] = fin_g
            for chave in ("grade_contra_3x3", "r3d_contra_a_grade"):
                c = dec["final_grades"].get(chave)
                if c:
                    avisar(f"FINAL_GRADE {chave}: delta medio "
                           f"{c.get('delta_medio')} | vitorias "
                           f"{c.get('vitorias')} | {c.get('veredito')}")
        gravar(dec)

    # ---------------- 7. linha de base 2D colapsada (artigo; folds 0 a 4)
    if "colapsado" in a.etapas:
        avisar("ETAPA 7: linha de base 2D colapsada (grade 1 x 1)")

        def colapsado():
            if testar(COLAPSADO, "colapsado_1x1", dec):
                gravar(dec)
                for s in SEM_FINAL:
                    rodar(dict(COLAPSADO, seed=s), F5, f"colapsado_1x1_s{s}")
                    gravar(dec)
        _etapa("colapsado", colapsado)
        dec["colapsado"] = _decidir(decidir_colapsado, L, rotulo="colapsado")
        gravar(dec)
        for nome, c in dec["colapsado"].get("comparacoes", {}).items():
            avisar(f"COLAPSADO {nome}: delta medio {c.get('delta_medio')} | "
                   f"folds com erro menor {c.get('vitorias')}")

    # ---------------- 8. tres metodos com validacao interna (artigo; folds 0 a 4)
    if "validacao_interna" in a.etapas:
        avisar("ETAPA 8: tres metodos com validacao interna")

        def validacao_interna():
            ok = {nome: testar(cfg, f"vi_{nome}", dec)
                  for nome, cfg in BRACOS_VI.items()}
            gravar(dec)
            for s in SEM_FINAL:       # intercalado: conjuntos pareados completos
                for nome, cfg in BRACOS_VI.items():
                    if ok[nome]:
                        rodar(dict(cfg, seed=s), F5, f"vi_{nome}_s{s}")
                gravar(dec)
        _etapa("validacao_interna", validacao_interna)
        dec["validacao_interna"] = _decidir(decidir_validacao_interna, L,
                                            rotulo="validacao_interna")
        gravar(dec)
        for nome, c in dec["validacao_interna"].get("comparacoes",
                                                    {}).items():
            avisar(f"VI {nome}: delta medio {c.get('delta_medio')} | folds "
                   f"com erro menor {c.get('vitorias')}"
                   f"{' | ' + c['veredito'] if c.get('veredito') else ''}")
    gravar(dec)
    avisar("FILA_FIM")


if __name__ == "__main__":
    main()
