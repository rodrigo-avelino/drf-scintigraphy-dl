# -*- coding: utf-8 -*-
"""Fase 3, investida nos modelos 3D: fila automatica dos Experimentos 1 a 4.

Exp. 1: replicas de semente (R3D-18 no fold 1 e controle) e R3D-18 com
T = 9; Exp. 2: BN congelado com estatisticas do dominio; Exp. 3: triagem do
pacote estrutural (cabeca espacial e aquecimento do lr) contra o controle;
Exp. 4: confirmacao do vencedor, se houver, em 5 folds com sementes novas.

Criterios fixados antes de qualquer execucao. As regras de decisao entre as
etapas estao implementadas aqui exatamente como foram fixadas; nada e
ajustado depois de ver a validacao.

Roda tudo na GPU local, um treino por vez (subprocessos de
src/fase3_rodar.py, com retomada por fold), e grava as leituras e decisoes
em data/fase3/investida_decisoes.json ao fim de cada etapa.

Uso (raiz do projeto):
  .venv\\Scripts\\python -u src\\fase3_investida.py [--esperar-pid N]
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

ROOT = Path(__file__).resolve().parents[1]
PASTA = ROOT / "data" / "fase3"
RODAR = str(ROOT / "src" / "fase3_rodar.py")
LIMITE_SEG = 30.0        # s por epoca; acima disso, transbordo da VRAM

F3 = (0, 1, 2)           # triagem
F5 = (0, 1, 2, 3, 4)     # confirmacao
SEMENTES_E1 = (42, 1042, 2042)      # R3D T=19 no fold 1
SEMENTES_CTL = (42, 1042)           # controle na triagem
SEMENTES_E4 = (3042, 4042)          # confirmacao (nunca usadas antes)
SIGMA_3D_RESERVA = 1.0   # DP por execucao se o Exp. 1 nao puder medir
                         # (ResNets com BN na Fase 2)

CTL = dict(abordagem="mosaico_vgg19")
R3D = dict(abordagem="r3d_18")
R21 = dict(abordagem="r2plus1d_18")
R3D_CONG = dict(R3D, bn="congelado_dominio")
R21_CONG = dict(R21, bn="congelado_dominio", micro_lote=8)
PACOTE = dict(cabeca="espacial", aquecimento=3)


# ------------------------------------------------------------- leitura
class Leitor:
    """Le os resultados gravados por rodar_configuracao (particoes fixas)."""

    def __init__(self, pasta=PASTA):
        self.pasta = Path(pasta)

    def _df(self):
        arq = self.pasta / "folds_fase3.csv"
        if not arq.exists():
            return pd.DataFrame(columns=["tag", "fold", "mae", "particao"])
        df = pd.read_csv(arq)
        return df[df.particao == f3.PARTICAO_ID]

    def linha(self, cfg, fold):
        df = self._df()
        d = df[(df.tag == f3.tag_config(cfg)) & (df.fold.astype(int) == fold)]
        return d.iloc[-1] if len(d) else None

    def mae(self, cfg, fold):
        lin = self.linha(cfg, fold)
        return float(lin["mae"]) if lin is not None else np.nan

    def recal_menos_mae(self, cfg, fold):
        lin = self.linha(cfg, fold)
        if lin is None or pd.isna(lin.get("mae_bn_recal", np.nan)):
            return np.nan
        return float(lin["mae_bn_recal"]) - float(lin["mae"])

    def nivel(self, cfg, fold, janela=5):
        """Nivel tipico: mediana do MAE de validacao nas +-5 epocas em torno
        da melhor epoca (menor mse_va)."""
        arq = (self.pasta / "historicos_fase3"
               / f"hist_{f3.tag_config(cfg)}_fold{fold}.csv")
        if not arq.exists():
            return np.nan
        h = pd.read_csv(arq)
        b = int(h.mse_va.idxmin())
        return float(h.mae_va.iloc[max(0, b - janela):b + janela + 1].median())


def _media(valores):
    v = [x for x in valores if np.isfinite(x)]
    return float(np.mean(v)) if v else np.nan


def _media_completa(valores):
    """Media so com todas as execucoes previstas (senao NaN)."""
    return float(np.mean(valores)) if _todos_finitos(valores) else np.nan


def _todos_finitos(valores):
    return all(np.isfinite(x) for x in valores)


def _r(x):
    """Arredonda antes de comparar com limiares (MAEs gravados com 3 casas):
    evita que 5,2 e 5,4 deem media 5,300000000000001 > 5,3."""
    return round(float(x), 6) if np.isfinite(x) else np.nan


# ------------------------------------------------------------ decisoes
def decidir_e1(L):
    """Leituras do Exp. 1 (registradas antes de rodar)."""
    d = {}
    f1 = [L.mae(dict(R3D, seed=s), 1) for s in SEMENTES_E1]
    novas = _r(_media_completa(f1[1:]))
    d["r3d_T19_fold1_por_semente"] = dict(zip(SEMENTES_E1, f1))
    d["r3d_T19_fold1_media_sementes_novas"] = novas
    if not np.isfinite(novas):
        d["leitura_fold1"] = "incompleto (falta execucao das sementes novas)"
    elif novas <= 5.3:
        d["leitura_fold1"] = "acaso de otimizacao (<= 5,3)"
    elif novas >= 6.0:
        d["leitura_fold1"] = "deficit sistematico (>= 6,0)"
    else:
        d["leitura_fold1"] = "inconclusivo (entre 5,3 e 6,0)"

    validos = [v for v in f1 if np.isfinite(v)]
    d["n_execucoes_r3d_fold1"] = len(validos)
    sig3 = float(np.std(validos, ddof=1)) if len(validos) >= 2 else np.nan
    ctl = {f: [L.mae(dict(CTL, seed=s), f) for s in SEMENTES_CTL] for f in F3}
    variancias = [np.var(v, ddof=1) for v in ctl.values() if _todos_finitos(v)]
    sigc = float(np.sqrt(np.mean(variancias))) if variancias else np.nan
    d["controle_por_fold_e_semente"] = ctl
    d["sigma_3d_fold1"] = sig3
    d["sigma_controle"] = sigc
    if not np.isfinite(sig3):
        # mesma reserva do Exp. 2, para as duas regras ficarem coerentes
        sig3 = SIGMA_3D_RESERVA
        d["sigma_3d_usado"] = sig3
        d["sigma_3d_origem"] = "reserva (DP das ResNets com BN na Fase 2)"

    # limiar do Exp. 3: -0,5 pp; recalculado se o DP do 3D passar de 0,8 pp
    if _r(sig3) > 0.8:
        sc = sigc if np.isfinite(sigc) else sig3
        d["limiar_e3"] = _r(-0.8 * np.sqrt(sig3 ** 2 + sc ** 2 / 2))
        d["limiar_e3_origem"] = "recalculado: -0,8 x raiz(s3d^2 + sctl^2/2)"
    else:
        d["limiar_e3"] = -0.5
        d["limiar_e3_origem"] = "padrao (-0,5 pp)"

    # T = 9 contra T = 19 (fold 1 contra a media das 3 execucoes)
    t19 = {0: L.mae(R3D, 0), 1: _media(f1), 2: L.mae(R3D, 2)}
    t9 = {f: L.mae(dict(R3D, T=9), f) for f in F3}
    dif = [t9[f] - t19[f] for f in F3]
    n19 = {0: L.nivel(R3D, 0),
           1: _media([L.nivel(dict(R3D, seed=s), 1) for s in SEMENTES_E1]),
           2: L.nivel(R3D, 2)}
    n9 = {f: L.nivel(dict(R3D, T=9), f) for f in F3}
    dn = [n9[f] - n19[f] for f in F3]
    d.update(t9=t9, t19_ref=t19, dif_t9_menos_t19=dif,
             nivel_tipico_t9=n9, nivel_tipico_t19=n19,
             dif_nivel_tipico=dn)
    if not (_todos_finitos(dif) and _todos_finitos(dn)):
        d["leitura_T"] = "sem dados"
    else:
        md, mn = _r(np.mean(dif)), _r(np.mean(dn))
        d["dif_media"], d["dif_nivel_medio"] = md, mn
        if md > 0.7 and sum(x > 0 for x in dif) >= 2 and mn > 0:
            d["leitura_T"] = "T importa"
        elif md < -0.7 and sum(x < 0 for x in dif) >= 2 and mn < 0:
            d["leitura_T"] = "T=9 melhor"
        else:
            d["leitura_T"] = "indiferente"

    # previsao: a recalibracao de BN piora o MAE (+0,5 pp nas 2 sementes)
    rec = [_r(L.recal_menos_mae(dict(R3D, seed=s), 1))
           for s in SEMENTES_E1[1:]]
    d["recal_menos_mae_fold1_sementes_novas"] = rec
    if not _todos_finitos(rec):
        d["previsao_recal"] = "sem dados"
    elif all(r >= 0.5 for r in rec):
        d["previsao_recal"] = "confirmada (recalibracao piora >= 0,5 pp)"
    else:
        d["previsao_recal"] = "nao confirmada"
    return d


def decidir_e2(L, d1):
    """Regra de recuo do BN para o pacote do Exp. 3."""
    d = {}
    trein = _media([L.mae(R3D, 0), _media(list(d1["r3d_T19_fold1_por_semente"]
                                               .values())), L.mae(R3D, 2)])
    cong = [L.mae(R3D_CONG, f) for f in F3]
    sig = d1.get("sigma_3d_fold1", np.nan)
    if sig is None or not np.isfinite(sig):
        sig = SIGMA_3D_RESERVA
    d.update(r3d_treinavel_ref=trein, r3d_congelado_por_fold=cong,
             r2plus1d_congelado_por_fold=[L.mae(R21_CONG, f) for f in F3],
             limiar_recuo=sig)
    if not _todos_finitos(cong):
        # o recuo exige piora MEDIDA; sem os 3 folds vale o padrao registrado
        d["bn_pacote"] = "congelado_dominio"
        d["status"] = "incompleto"
        d["motivo"] = ("recuo nao avaliavel: faltam folds do R3D-18 congelado; "
                       "mantido o padrao pre-registrado")
    else:
        d["r3d_congelado_media"] = float(np.mean(cong))
        d["congelado_menos_treinavel"] = _r(float(np.mean(cong)) - trein)
        if d["congelado_menos_treinavel"] > _r(sig):
            d["bn_pacote"] = "treinavel"
            d["motivo"] = "recuo: congelado pior que o treinavel alem do DP"
        else:
            d["bn_pacote"] = "congelado_dominio"
            d["motivo"] = "congelado nao piora alem do DP entre execucoes"
    return d


def candidatos_e3(d1, d2):
    c = {"r3d_pacote": dict(R3D, **PACOTE, bn=d2["bn_pacote"])}
    if d2["bn_pacote"] == "congelado_dominio":
        # com BN treinavel o R(2+1)D nao cabe na 4060: sai da investida
        c["r2plus1d_pacote"] = dict(R21, **PACOTE, bn="congelado_dominio",
                                    lr=5e-5, micro_lote=8)
    if d1.get("leitura_T") == "T importa":
        c["r3d_pacote_sem_stride"] = dict(c["r3d_pacote"], sem_stride_t=True)
    return c


def decidir_e3(L, d1, cands):
    ctl_ref = {f: _media([L.mae(dict(CTL, seed=s), f) for s in SEMENTES_CTL])
               for f in F3}
    lim = d1.get("limiar_e3", -0.5)
    r3d_atual = _media([L.mae(R3D, 0),
                        _media(list(d1["r3d_T19_fold1_por_semente"].values())),
                        L.mae(R3D, 2)])
    res = {}
    for nome, cfg in cands.items():
        maes = [L.mae(cfg, f) for f in F3]
        dl = [m - ctl_ref[f] for m, f in zip(maes, F3)]
        ok = _todos_finitos(dl)
        md = _r(np.mean(dl)) if ok else np.nan
        vit = int(sum(x < 0 for x in dl)) if ok else 0
        res[nome] = dict(
            tag=f3.tag_config(cfg), mae=maes, deltas=dl, delta_medio=md,
            vitorias=vit,
            # dado faltando nao e reprovacao: fica registrado como incompleto
            passa=bool(md <= _r(lim) and vit >= 2) if ok else None,
            status="completo" if ok else "incompleto",
            pacote_malsucedido=bool(ok and np.mean(maes) > r3d_atual
                                    and nome.startswith("r3d")))
    passaram = [n for n in res if res[n]["passa"]]
    venc = min(passaram, key=lambda n: res[n]["delta_medio"]) if passaram \
        else None
    return dict(controle_ref=ctl_ref, limiar=lim, r3d_atual_ref=r3d_atual,
                configuracoes=res, vencedor=venc,
                incompletos=[n for n in res if res[n]["passa"] is None])


def decidir_e4(L, cfg_v):
    faltam = [(f3.tag_config(dict(b, seed=s)), f) for b in (cfg_v, CTL)
              for s in SEMENTES_E4 for f in F5
              if not np.isfinite(L.mae(dict(b, seed=s), f))]
    por_fold = []
    for f in F5:
        v = _media_completa([L.mae(dict(cfg_v, seed=s), f)
                             for s in SEMENTES_E4])
        c = _media_completa([L.mae(dict(CTL, seed=s), f)
                             for s in SEMENTES_E4])
        por_fold.append(dict(fold=f, modelo=v, controle=c, delta=v - c))
    dl = [p["delta"] for p in por_fold]
    t9 = [L.mae(dict(cfg_v, T=9, seed=SEMENTES_E4[0]), f) for f in F5]
    t19 = [p["modelo"] for p in por_fold]
    base = dict(por_fold=por_fold, t9_por_fold=t9,
                t9_menos_t19=[a - b for a, b in zip(t9, t19)])
    if faltam:   # falta de dado nao e empate: a regra nao e aplicada
        return dict(base, veredito="incompleto", faltantes=faltam,
                    delta_medio=np.nan, vitorias=None)
    md = _r(np.mean(dl))
    vence = bool(md < -1.0 or all(x < 0 for x in dl))
    return dict(base, delta_medio=md, vitorias=int(sum(x < 0 for x in dl)),
                veredito="3D vence" if vence else "empate tecnico")


# ------------------------------------------------------------- execucao
def avisar(msg):
    print(f"[investida {datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def args_cli(cfg):
    a = ["--abordagens", cfg["abordagem"]]
    for chave, flag in (("T", "--T"), ("bn", "--bn"), ("seed", "--seed"),
                        ("lr", "--lr"), ("cabeca", "--cabeca"),
                        ("aquecimento", "--aquecimento"),
                        ("micro_lote", "--micro-lote")):
        if cfg.get(chave) is not None:
            a += [flag, repr(cfg[chave]) if chave == "lr" else str(cfg[chave])]
    if cfg.get("sem_stride_t"):
        a.append("--sem-stride-t")
    return a


FALHAS = []    # (rotulo, tag, folds, codigo) que falharam duas vezes


def _executar(cmd, rotulo):
    log = PASTA / f"log_investida_{rotulo}.txt"
    with open(log, "a", encoding="utf-8") as fo:
        return subprocess.run(cmd, stdout=fo, stderr=subprocess.STDOUT,
                              cwd=ROOT).returncode


def rodar(cfg, folds, rotulo):
    """Treina os folds que faltam; uma nova tentativa se o processo falhar
    (a retomada por fold refaz so o que faltou)."""
    if _todos_finitos([Leitor().mae(cfg, f) for f in folds]):
        avisar(f"{rotulo}: {f3.tag_config(cfg)} ja completo, pulando")
        return 0
    cmd = [sys.executable, "-u", "-X", "utf8", RODAR, *args_cli(cfg),
           "--folds", *map(str, folds), "--limite-seg-epoca", str(LIMITE_SEG)]
    for tentativa in (1, 2):
        t = time.time()
        rc = _executar(cmd, rotulo)
        avisar(f"{rotulo}: {f3.tag_config(cfg)} folds {list(folds)} -> codigo "
               f"{rc} ({(time.time() - t) / 60:.1f} min, tentativa "
               f"{tentativa})")
        if rc == 0:
            return 0
        time.sleep(60)          # tempo para o driver se recuperar
    FALHAS.append(dict(rotulo=rotulo, tag=f3.tag_config(cfg),
                       folds=list(folds), codigo=rc))
    return rc


def testar(cfg, rotulo):
    """2 epocas no fold 0 (sem gravar): memoria e tempo antes da fila."""
    csv = PASTA / f"teste_investida_{rotulo}.csv"
    rc = None
    for tentativa in (1, 2):
        if csv.exists():
            csv.unlink()
        rc = _executar([sys.executable, "-u", "-X", "utf8", RODAR, "--teste",
                        "--teste-csv", str(csv), *args_cli(cfg)],
                       f"teste_{rotulo}")
        if rc == 0 and csv.exists():
            break
        time.sleep(60)
    if rc != 0 or not csv.exists():
        avisar(f"TESTE FALHOU {rotulo}: codigo {rc}")
        FALHAS.append(dict(rotulo=f"teste_{rotulo}", tag=f3.tag_config(cfg),
                           folds=[0], codigo=rc))
        return dict(ok=False, motivo=f"codigo {rc}")
    t = pd.read_csv(csv).iloc[0].to_dict()
    ok = bool(t["ok"]) and float(t["seg_por_epoca"]) <= LIMITE_SEG
    if ok and cfg.get("cabeca") == "espacial":
        # sanidade da cabeca (vies na media): MAE na faixa de prever a media
        # (~13 a 14 pp) ou abaixo, nunca os 24 a 79 da cabeca original
        ok = float(t["mae_val_ep1"]) <= 20.0
        if not ok:
            t["erro"] = f"MAE da 1a epoca {t['mae_val_ep1']} > 20 pp"
    avisar(f"teste {rotulo}: ok={ok} | {t['seg_por_epoca']} s/epoca | "
           f"pico {t['pico_mem_gb']} GB (reservado {t['pico_res_gb']}) | "
           f"MAE val ep1 {t['mae_val_ep1']} | {t['erro']}")
    t["ok"] = ok
    if not ok:
        FALHAS.append(dict(rotulo=f"teste_{rotulo}", tag=f3.tag_config(cfg),
                           folds=[0], codigo=str(t.get("erro"))))
    return t


def _json(o):
    if isinstance(o, dict):
        return {str(k): _json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def gravar(dec):
    dec["atualizado"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    dec["falhas"] = list(FALHAS)
    with open(PASTA / "investida_decisoes.json", "w", encoding="utf-8") as fo:
        json.dump(_json(dec), fo, ensure_ascii=False, indent=1)


def _decidir(func, *a, reserva=None, rotulo=""):
    try:
        return func(*a)
    except Exception as e:      # a fila nao pode parar por erro de leitura
        avisar(f"ERRO na decisao {rotulo}: {type(e).__name__}: {e}")
        return dict(reserva or {}, erro=f"{type(e).__name__}: {e}")


def esperar(pid):
    def vivo():
        # compara BYTES: o tasklist em pt-BR responde em cp850 quando o PID
        # nao existe, e decodificar em UTF-8 (-X utf8) derrubaria o processo.
        # Em caso de erro, trata como vivo e tenta de novo no ciclo seguinte
        # (nunca comecar a fila em paralelo com outro treino na GPU).
        try:
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                                 capture_output=True).stdout or b""
            return f" {pid} ".encode() in out
        except Exception as e:
            avisar(f"aviso: tasklist falhou ({type(e).__name__}); tentando de "
                   "novo")
            return True
    if vivo():
        avisar(f"aguardando o processo {pid} terminar")
    while vivo():
        time.sleep(20)


def _etapa(nome, func):
    """Executa uma etapa sem deixar uma excecao derrubar a fila inteira."""
    try:
        func()
    except Exception as e:
        avisar(f"ERRO na etapa {nome}: {type(e).__name__}: {e}")
        FALHAS.append(dict(rotulo=f"etapa_{nome}", tag="", folds=[],
                           codigo=f"{type(e).__name__}: {e}"))


def _gravar_seguro(dec):
    try:
        gravar(dec)
    except Exception as e:      # ex.: arquivo aberto em outro programa
        avisar(f"ERRO ao gravar decisoes: {type(e).__name__}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--esperar-pid", type=int, default=None)
    a = p.parse_args()
    if sys.platform == "win32":
        import ctypes   # impede a suspensao por inatividade enquanto roda
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if a.esperar_pid:
        esperar(a.esperar_pid)
    L = Leitor()
    dec = dict(versao_lib=f3.VERSAO, testes={},
               inicio=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"))

    # ---------------- Exp. 1: replicas e controle de informacao (retomada)
    avisar("ETAPA 1: replicas de semente e T=9")

    def e1():
        rodar(dict(R3D, T=9), F3, "e1_T9")
        for s in SEMENTES_E1[1:]:
            rodar(dict(R3D, seed=s), [1], f"e1_s{s}")
        rodar(dict(CTL, seed=1042), F3, "e1_ctl1042")
    _etapa("E1", e1)
    d1 = _decidir(decidir_e1, L, rotulo="E1",
                  reserva=dict(limiar_e3=-0.5, leitura_T="sem dados",
                               sigma_3d_fold1=np.nan,
                               r3d_T19_fold1_por_semente={42: L.mae(R3D, 1)}))
    dec["e1"] = d1
    _gravar_seguro(dec)
    avisar(f"E1: fold 1 {d1.get('leitura_fold1')} | T: {d1.get('leitura_T')} "
           f"| limiar E3 {d1.get('limiar_e3')} | recal: "
           f"{d1.get('previsao_recal')}")

    # ---------------- Exp. 2: BN congelado com estatisticas do dominio
    avisar("ETAPA 2: BN congelado no dominio")

    def e2():
        for rot, cfg in (("e2_r3d_cong", R3D_CONG), ("e2_r21_cong", R21_CONG)):
            t = testar(cfg, rot)
            dec["testes"][rot] = t
            if t["ok"]:
                rodar(cfg, F3, rot)
    _etapa("E2", e2)
    d2 = _decidir(decidir_e2, L, d1, rotulo="E2",
                  reserva=dict(bn_pacote="congelado_dominio"))
    dec["e2"] = d2
    _gravar_seguro(dec)
    avisar(f"E2: pacote com BN {d2.get('bn_pacote')} ({d2.get('motivo')})")

    # ---------------- Exp. 3: pacote estrutural (triagem com k = 2 ou 3)
    avisar("ETAPA 3: pacote estrutural")
    try:
        cands = candidatos_e3(d1, d2)
    except Exception as e:
        avisar(f"ERRO nos candidatos do E3: {type(e).__name__}: {e}")
        cands = {"r3d_pacote": dict(R3D, **PACOTE, bn="congelado_dominio")}

    def e3():
        for nome, cfg in cands.items():
            t = testar(cfg, f"e3_{nome}")
            dec["testes"][f"e3_{nome}"] = t
            if t["ok"]:
                rodar(cfg, F3, f"e3_{nome}")
    _etapa("E3", e3)
    d3 = _decidir(decidir_e3, L, d1, cands, rotulo="E3",
                  reserva=dict(vencedor=None))
    d3["candidatos"] = {n: f3.tag_config(c) for n, c in cands.items()}
    dec["e3"] = d3
    _gravar_seguro(dec)
    venc = d3.get("vencedor")
    avisar(f"E3: vencedor = {venc} | incompletos = {d3.get('incompletos')}")

    # ---------------- Exp. 4: confirmacao (so se o Exp. 3 passar)
    if venc:
        cfg_v = cands[venc]
        avisar(f"ETAPA 4: confirmacao de {venc} em 5 folds com sementes novas")

        def e4():
            for s in SEMENTES_E4:
                rodar(dict(CTL, seed=s), F5, f"e4_ctl_s{s}")
                rodar(dict(cfg_v, seed=s), F5, f"e4_{venc}_s{s}")
            rodar(dict(cfg_v, T=9, seed=SEMENTES_E4[0]), F5, f"e4_{venc}_T9")
        _etapa("E4", e4)
        dec["e4"] = _decidir(decidir_e4, L, cfg_v, rotulo="E4")
        avisar(f"E4: {dec['e4'].get('veredito')} | delta medio "
               f"{dec['e4'].get('delta_medio')}")
    else:
        # sem confirmacao: linha de base descritiva do mosaico em 5 folds
        # (as sementes que a confirmacao usaria), para a etapa seguinte
        avisar("ETAPA extra: controle em 5 folds com 2 sementes (descritivo)")

        def extra():
            for s in SEMENTES_E4:
                rodar(dict(CTL, seed=s), F5, f"extra_ctl_s{s}")
        _etapa("extra", extra)
        dec["extra"] = dict(
            descricao="mosaico em 5 folds, sementes 3042 e 4042 (descritivo)",
            mae_por_fold={s: [L.mae(dict(CTL, seed=s), f) for f in F5]
                          for s in SEMENTES_E4})
    _gravar_seguro(dec)
    avisar("INVESTIDA_FIM")


if __name__ == "__main__":
    main()
