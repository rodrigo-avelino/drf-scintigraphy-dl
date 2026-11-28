# -*- coding: utf-8 -*-
"""Fase 4: tabelas da triagem, da rodada final e da ablacao (descritivo).

Le data/fase4 (folds_fase4.csv e historicos_fase4/pred_*.csv) e, para as
referencias da triagem e da ablacao, data/fase3. As regras, fixadas antes
das execucoes, estao em src/fase4_fila.py. Este script NAO decide nada na fila
local: as decisoes pelas regras fixadas ficam em data/fase4/fase4_decisoes.json,
gravadas por src/fase4_fila.py; aqui so se reportam os numeros e os IC. A
rodada do Colab (data/fase4_colab/ambos, espelho de MyDrive/fase4_colab/ambos)
nao passa pela fila: a decisao e calculada aqui por fase4_fila.decidir_colab
(mesma regra do 3D) e gravada em <pasta do Colab>/fase4_colab_decisao.json.

Uso: .venv\\Scripts\\python -X utf8 src\\fase4_analise.py [--pasta data/fase4]
     [--pasta-colab data/fase4_colab/ambos]
Saidas: <pasta>/fase4_tabelas.md e <pasta>/fase4_final_por_estudo.csv; com
a etapa do 2D colapsado nas decisoes, tambem
<pasta>/fase4_tres_metodos_por_estudo.csv (predicoes dos tres metodos); com a
etapa da validacao interna (em cada fold, o fold seguinte escolhe lr, parada e
melhor epoca), tambem <pasta>/fase4_vi_tres_metodos_por_estudo.csv (com a PDI
por estudo, quando o npz existe), o erro por faixa de LK, a convergencia do
treino com uma sensibilidade definida depois das execucoes e o diagnostico
do otimismo.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fase3_lib as f3  # noqa: E402
import fase4_fila as q  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
N_BOOT, SEMENTE_BOOT = 5000, 0       # como na Fase 3


# ------------------------------------------------------------- leitura
def ler_predicoes(cfg, sementes, folds, fontes):
    """Predicoes por estudo de uma configuracao (uma linha por semente e
    estudo). fontes = [(pasta, nome_base)], a primeira tem prioridade."""
    linhas = []
    for s in sementes:
        tag = f3.tag_config(dict(cfg, seed=s))
        for f in folds:
            for pasta, nome in fontes:
                arq = Path(pasta) / f"historicos_{nome}" / f"pred_{tag}_fold{f}.csv"
                if arq.exists():
                    d = pd.read_csv(arq)
                    d["semente"], d["fold"] = s, f
                    linhas.append(d)
                    break
    if not linhas:
        return None
    d = pd.concat(linhas, ignore_index=True)
    d["coorte"] = d.study.str.split("_").str[0]
    d["faixa"] = pd.cut(d.y, [-np.inf, 35, 65, np.inf], right=False,
                        labels=["LK < 35", "35 a 65", "LK >= 65"])
    if "pred_tta" not in d.columns:
        d["pred_tta"] = np.nan
    return d


def completo(d, sementes, folds):
    return (d is not None and set(d.semente) == set(sementes)
            and set(d.fold) == set(folds))


# ------------------------------------------------------------ metricas
def por_execucao(d, col="pred"):
    """Metricas por (semente, fold): MAE, RMSE, r e R2."""
    out = []
    for (s, f), g in d.groupby(["semente", "fold"]):
        m = f3.metricas(g[col].to_numpy(), g.y.to_numpy())
        out.append(dict(semente=s, fold=f, **m))
    return pd.DataFrame(out)


def calibracao(d, col="pred"):
    """Inclinacao de pred contra y e vieses nas caudas, por semente (todos os
    estudos dos folds juntos); devolve a media entre sementes."""
    res = []
    for _, g in d.groupby("semente"):
        b = np.polyfit(g.y, g[col], 1)[0]
        v35 = (g[col] - g.y)[g.y < 35].mean()
        v65 = (g[col] - g.y)[g.y >= 65].mean()
        res.append((b, v35, v65))
    return np.mean(res, axis=0)


def erro_por_estudo(d, col="pred"):
    """Erro absoluto por estudo, medio entre as sementes."""
    e = d.assign(ea=(d[col] - d.y).abs())
    return e.groupby("study").agg(ea=("ea", "mean"), y=("y", "first"),
                                  fold=("fold", "first"),
                                  coorte=("coorte", "first"),
                                  faixa=("faixa", "first"))


def bootstrap_dif(a, b, n=N_BOOT, semente=SEMENTE_BOOT):
    """IC 95% da diferenca de medias pareada por estudo (a - b)."""
    dif = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(semente)
    idx = rng.integers(0, len(dif), size=(n, len(dif)))
    bs = dif[idx].mean(axis=1)
    return float(dif.mean()), float(np.percentile(bs, 2.5)), \
        float(np.percentile(bs, 97.5))


def bland_altman(d, col="pred"):
    """Vies e limites de 95% por semente (todos os estudos)."""
    out = []
    for s, g in d.groupby("semente"):
        dif = g[col] - g.y
        vies, dp = dif.mean(), dif.std(ddof=1)
        out.append(dict(semente=s, n=len(g), vies=vies, lim_inf=vies - 1.96 * dp,
                        lim_sup=vies + 1.96 * dp))
    return pd.DataFrame(out)


# -------------------------------------------------------------- tabelas
def fmt(x, casas=2):
    return "" if x is None or not np.isfinite(x) else f"{x:.{casas}f}"


def md_tabela(df):
    cab = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    lin = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([cab, sep, *lin])


def linha_metricas(n, d):
    """Metricas de modelo unico de um braco (linha da tabela principal)."""
    ex = por_execucao(d)
    pf = ex.groupby("fold").mae.mean()
    ext = por_execucao(d, "pred_tta")
    b, v35, v65 = calibracao(d)
    return dict(braço=n, MAE=f"{pf.mean():.2f} ± {pf.std(ddof=1):.2f}",
                RMSE=fmt(ex.rmse.mean()), r=fmt(ex.r.mean(), 3),
                R2=fmt(ex.r2.mean(), 3),
                **{"MAE com TTA": fmt(ext.groupby("fold").mae.mean().mean())},
                inclinação=fmt(b, 3), **{"viés LK < 35": fmt(v35),
                                          "viés LK >= 65": fmt(v65)})


def comparar_com(d, ref):
    """Delta braco - referencia por fold (media das sementes) e por estudo
    (IC por bootstrap pareado), com os folds 3 e 4 a parte."""
    dl = (por_execucao(d).groupby("fold").mae.mean()
          - por_execucao(ref).groupby("fold").mae.mean())
    e, er = erro_por_estudo(d), erro_por_estudo(ref)
    md, li, ls = bootstrap_dif(e.ea, er.loc[e.index].ea)
    sub = e.fold.isin([3, 4])
    md34, li34, ls34 = bootstrap_dif(e.ea[sub], er.loc[e.index].ea[sub])
    return {**{f"Δ fold {f}": fmt(v) for f, v in dl.items()},
            "Δ médio": fmt(dl.mean()), "folds vencidos": int((dl < 0).sum()),
            "Δ folds 3 e 4": fmt(dl.loc[[3, 4]].mean()),
            "por estudo (IC 95%)": f"{md:+.2f} ({li:+.2f} a {ls:+.2f})",
            "folds 3 e 4 por estudo (IC)":
                f"{md34:+.2f} ({li34:+.2f} a {ls34:+.2f})"}


def _tabela_ba(preds, extra=None):
    """Bland-Altman (pred - y) por semente, um braco por linha: vies com a
    faixa entre sementes e limites de 95% medios. extra: linhas prontas
    (por exemplo, a PDI, que nao tem sementes) acrescentadas no fim."""
    t = []
    for n, d in preds.items():
        ba = bland_altman(d)
        t.append(dict(braço=n, viés=f"{ba.vies.mean():+.2f} "
                                    f"({ba.vies.min():+.2f} a {ba.vies.max():+.2f})",
                      **{"limite inferior": fmt(ba.lim_inf.mean()),
                         "limite superior": fmt(ba.lim_sup.mean()),
                         "amplitude": fmt((ba.lim_sup - ba.lim_inf).mean())}))
    return md_tabela(pd.DataFrame(t + (extra or [])))


def mae_por_grupo(preds):
    """MAE por coorte e faixa de LK (erro por estudo medio entre sementes)."""
    ep = {n: erro_por_estudo(d) for n, d in preds.items()}
    base = next(iter(ep.values()))
    grupos = [("drsbru", base.coorte == "drsbru"),
              ("drsprg", base.coorte == "drsprg")] + [
        (fx, base.faixa == fx) for fx in ["LK < 35", "35 a 65", "LK >= 65"]]
    return pd.DataFrame([{"grupo": rot, "n": int(m.sum()),
                          **{n: fmt(e.ea.loc[base.index][m].mean())
                             for n, e in ep.items()}} for rot, m in grupos])


def secao_final(fontes, dec, saida_csv, bracos=None,
                titulo="Rodada final (folds 0 a 4, sementes 5042 a 8042)",
                nota="Vereditos pela regra fixada antes das execuções, em "
                     "`fase4_decisoes.json`."):
    """Tabelas da rodada final. Sem `bracos`, os da fila 1 (controle, R3D-18
    e o finalista do caminho A); a chave "controle" e sempre a referencia."""
    if bracos is None:
        fin = dec.get("final", {}).get("finalista_A") or dec.get(
            "triagem", {}).get("finalista")
        bracos = {"controle": q.CTL,
                  "R3D-18 com cabeça espacial e aquecimento": q.R3D_REF}
        if fin:
            bracos[f"mosaico refinado ({fin})"] = q.CANDIDATOS_A[fin]
    preds = {n: ler_predicoes(c, q.SEM_FINAL, q.F5, fontes)
             for n, c in bracos.items()}
    falt = [n for n, d in preds.items() if not completo(d, q.SEM_FINAL, q.F5)]
    txt = [f"## {titulo}", ""]
    if falt:
        txt.append(f"Braços incompletos (fora das tabelas): {', '.join(falt)}.")
        txt.append("")
    preds = {n: d for n, d in preds.items() if n not in falt}
    if "controle" not in preds:
        return txt + ["Sem o controle completo, nada a comparar."]

    # tabela 1: metricas de modelo unico
    arq_f = Path(fontes[0][0]) / f"folds_{fontes[0][1]}.csv"
    rows = pd.read_csv(arq_f) if arq_f.exists() else pd.DataFrame()
    t1 = [linha_metricas(n, d) for n, d in preds.items()]
    if len(rows):
        r = rows[rows.tag.str.startswith("mosaico_vgg19_grade3x3_moderado_bn-"
                                         "treinavel_s5042")]
        pdi = r.drop_duplicates("fold").sort_values("fold")
        if len(pdi) == 5:
            t1.append({"braço": "PDI recalibrada",
                       "MAE": f"{pdi.mae_pdi_recal.mean():.2f} ± "
                              f"{pdi.mae_pdi_recal.std(ddof=1):.2f}"})
            t1.append({"braço": "PDI crua",
                       "MAE": f"{pdi.mae_pdi_crua.mean():.2f} ± "
                              f"{pdi.mae_pdi_crua.std(ddof=1):.2f}"})
    txt += ["MAE de modelo único: média nos 5 folds da média das 4 sementes "
            "em cada fold (± DP entre folds); RMSE, r e R²: média das 20 "
            "execuções; inclinação e vieses: média das 4 sementes, nos 207 "
            "estudos.", "", md_tabela(pd.DataFrame(t1).fillna("")), ""]

    # tabela 2: comparacao com o controle, por fold e por estudo
    ctl = preds["controle"]
    e_ctl = erro_por_estudo(ctl)
    t2, por_estudo = [], {"controle": e_ctl.ea}
    for n, d in preds.items():
        if n == "controle":
            continue
        por_estudo[n] = erro_por_estudo(d).ea
        t2.append(dict(braço=n, **comparar_com(d, ctl)))
    if t2:
        txt += ["Δ = braço menos controle (pp; negativo = erro menor). " + nota,
                "", md_tabela(pd.DataFrame(t2)), ""]

    # tabela 3: coortes e faixas (erro por estudo medio entre sementes)
    t3 = []
    grupos = [("drsbru", e_ctl.coorte == "drsbru"),
              ("drsprg", e_ctl.coorte == "drsprg")] + [
        (fx, e_ctl.faixa == fx) for fx in ["LK < 35", "35 a 65", "LK >= 65"]]
    for rot, m in grupos:
        lin = {"grupo": rot, "n": int(m.sum()),
               "controle": fmt(e_ctl.ea[m].mean())}
        for n, ea in por_estudo.items():
            if n == "controle":
                continue
            a = ea.loc[e_ctl.index][m]
            md, li, ls = bootstrap_dif(a, e_ctl.ea[m])
            lin[n] = f"{a.mean():.2f} (Δ {md:+.2f}; {li:+.2f} a {ls:+.2f})"
        t3.append(lin)
    txt += ["MAE por coorte e faixa (erro por estudo médio entre as sementes; "
            "IC por bootstrap pareado dentro do grupo).", "",
            md_tabela(pd.DataFrame(t3)), ""]

    # tabela 4: ensembles (media das predicoes por estudo)
    def ens(d, sementes, col="pred"):
        g = d[d.semente.isin(sementes)].groupby("study")
        return g[col].mean(), g.y.first()

    t4 = []
    p4c, y = ens(ctl, q.SEM_FINAL)
    base4 = (p4c - y).abs()
    t4.append(dict(ensemble="controle, 4 sementes", MAE=fmt(base4.mean()),
                   **{"Δ para 4 sementes do controle": ""}))
    for n, d in preds.items():
        if n == "controle":
            continue
        p4, _ = ens(d, q.SEM_FINAL)
        ea = (p4 - y.loc[p4.index]).abs()
        t4.append(dict(ensemble=f"{n}, 4 sementes", MAE=fmt(ea.mean()),
                       **{"Δ para 4 sementes do controle": ""}))
        for par in ((5042, 6042), (7042, 8042)):
            pm = (ens(d, par)[0] + ens(ctl, par)[0].loc[ens(d, par)[0].index]) / 2
            ea = (pm - y.loc[pm.index]).abs()
            md, li, ls = bootstrap_dif(ea, base4.loc[ea.index])
            t4.append(dict(ensemble=f"{n} ({par[0]} e {par[1]}) + controle "
                                    f"({par[0]} e {par[1]})",
                           MAE=fmt(ea.mean()),
                           **{"Δ para 4 sementes do controle":
                              f"{md:+.2f} ({li:+.2f} a {ls:+.2f})"}))
    txt += ["Ensembles: média das predições por estudo; a comparação é sempre "
            "com o mesmo número de membros (4).", "",
            md_tabela(pd.DataFrame(t4)), ""]

    # tabela 5: Bland-Altman por semente
    txt += ["Bland-Altman (pred − y), por semente nos 207 estudos; viés com a "
            "faixa entre sementes e limites de 95% médios.", "",
            _tabela_ba(preds), ""]

    # predicoes por estudo (media das sementes) para figuras e conferencia
    pe = pd.DataFrame({n: d.groupby("study").pred.mean()
                       for n, d in preds.items()})
    pe.insert(0, "y", ctl.groupby("study").y.first())
    pe.to_csv(saida_csv)
    return txt


def secao_triagem(fontes, dec, chave="triagem",
                   titulo="Triagem do caminho A (folds 0 a 2, sementes 3042 "
                          "e 4042)", rotulo_ref="controle (referência)"):
    txt = [f"## {titulo}", ""]
    d = dec.get(chave)
    if not d or "configuracoes" not in d:
        return txt + ["Sem registro da triagem."]
    ref = d["controle_ref_por_fold"]
    t = [dict(configuração=rotulo_ref,
              **{f"fold {f}": fmt(v) for f, v in zip(q.F3, ref)},
              **{"Δ médio": "", "vitórias": "", "MAE com TTA": "",
                 "passa": ""})]
    for nome, c in d["configuracoes"].items():
        tta = [v for v in (c.get("mae_tta_por_fold") or []) if v is not None]
        t.append(dict(configuração=nome,
                      **{f"fold {f}": fmt(v) for f, v in zip(q.F3,
                                                             c["mae_por_fold"])},
                      **{"Δ médio": fmt(c["delta_medio"], 4),
                         "vitórias": c["vitorias"],
                         "MAE com TTA": fmt(np.mean(tta)) if tta else "",
                         "passa": "sim" if c["passa"] else "não"}))
    fin = d.get("finalista") or "nenhum"
    return txt + [f"Regra: {d.get('regra')}. Finalista: {fin}. Δ médio com 4 casas, "
                  "como gravado nas decisões e comparado ao limiar.", "",
                  md_tabela(pd.DataFrame(t)), ""]


def secao_ablacao(dec):
    txt = ["## Ablação do caminho B (folds 0 a 2, sementes 3042 e 4042)", ""]
    d = dec.get("ablacao")
    if not d or "leituras" not in d:
        return txt + ["Sem registro da ablação."]
    t = [dict(configuração=n, **{f"fold {f}": fmt(v) for f, v in zip(q.F3, pf)},
              média=fmt(np.mean(pf)) if all(v is not None for v in pf) else "")
         for n, pf in d["mae_por_fold"].items()]
    lt = [dict(leitura=n, **{"com o fator": l["com"], "sem o fator": l["sem"],
                             "efeito médio": fmt(l.get("efeito_medio")),
                             "resultado": l["leitura"]})
          for n, l in d["leituras"].items()]
    return txt + [md_tabela(pd.DataFrame(t)), "",
                  "Efeito = MAE com o fator menos MAE sem o fator.", "",
                  md_tabela(pd.DataFrame(lt)), ""]


def _ler_bracos(bracos, fontes):
    """Predicoes da rodada final de cada braco e lista dos incompletos."""
    preds = {n: ler_predicoes(c, q.SEM_FINAL, q.F5, fontes)
             for n, c in bracos.items()}
    falt = [n for n, d in preds.items() if not completo(d, q.SEM_FINAL, q.F5)]
    return preds, falt


def secao_atribuicao(fontes, dec):
    """Teste de atribuicao (fila 2): T = 9 contra o T = 19 e contra o mosaico 3 x 3."""
    txt = ["## Atribuição com T = 9 (folds 0 a 4, sementes 5042 a 8042)", ""]
    d = dec.get("atribuicao")
    if not d or "leituras" not in d:
        return txt + ["Sem registro da atribuição."]
    n9, n19, nm = "R3D-18 T = 9", "R3D-18 T = 19", "mosaico 3 × 3 (controle)"
    preds, falt = _ler_bracos({n9: q.R3D_T9, n19: q.R3D_REF, nm: q.CTL},
                              fontes)
    if falt:
        return txt + [f"Braços incompletos: {', '.join(falt)}. Leitura "
                      f"registrada: {d.get('interpretacao')}."]
    t1 = [linha_metricas(n, x) for n, x in preds.items()]
    lt = d["leituras"]
    t2 = [dict(comparação=rot, **comparar_com(preds[n9], preds[ref]),
               **{"mesmo sinal": lt[chave].get("folds_com_o_mesmo_sinal", ""),
                  "leitura": lt[chave]["leitura"]})
          for rot, ref, chave in (("T = 9 menos T = 19", n19, "T9_menos_T19"),
                                  ("T = 9 menos mosaico 3 × 3", nm,
                                   "T9_menos_mosaico"))]
    return txt + [
        "Mesmas definições da rodada final. O T = 9 usa a mesma agregação "
        "temporal das células do mosaico 3 × 3.", "",
        md_tabela(pd.DataFrame(t1)), "",
        f"Regra (T = 9, descritiva): {d.get('regra')}. Na coluna \"folds "
        "vencidos\", folds com Δ negativo.", "",
        md_tabela(pd.DataFrame(t2)), "",
        f"**Interpretação pela regra:** {d.get('interpretacao')}.",
        "", "MAE por coorte e faixa (erro por estudo médio entre as sementes).",
        "", md_tabela(mae_por_grupo(preds)), ""]


def secao_final_grades(fontes, dec):
    """Grades nao quadradas do mosaico (fila 2): grade finalista contra o 3 x 3
    e, se ela passar, o 3D contra a grade."""
    txt = ["## Grade finalista na rodada final (folds 0 a 4, sementes 5042 a "
           "8042)", ""]
    d = dec.get("final_grades")
    if not d:
        return txt + ["Sem registro da rodada final das grades."]
    fin = d.get("finalista")
    if not fin:
        return txt + ["Nenhuma grade passou na triagem; a rodada final das "
                      "grades não rodou."]
    ng, nc = f"grade {fin}", "mosaico 3 × 3 (controle)"
    nr = "R3D-18 com cabeça espacial e aquecimento"
    preds, falt = _ler_bracos({nc: q.CTL, ng: q.GRADES[fin], nr: q.R3D_REF},
                              fontes)
    if falt:
        return txt + [f"Braços incompletos: {', '.join(falt)}."]
    t2 = [dict(comparação=f"{ng} menos 3 × 3", **comparar_com(preds[ng],
                                                              preds[nc]),
               veredito=d["grade_contra_3x3"]["veredito"])]
    if "r3d_contra_a_grade" in d:
        t2.append(dict(comparação=f"R3D-18 menos {ng}",
                       **comparar_com(preds[nr], preds[ng]),
                       veredito=d["r3d_contra_a_grade"]["veredito"]))
    return txt + [md_tabela(pd.DataFrame([linha_metricas(n, x)
                                          for n, x in preds.items()])), "",
                  "Vereditos pela regra fixada antes das execuções.", "",
                  md_tabela(pd.DataFrame(t2)), "",
                  "MAE por coorte e faixa (erro por estudo médio entre as "
                  "sementes).", "", md_tabela(mae_por_grupo(preds)), ""]


def secao_colapsado(fontes, dec, saida_csv):
    """Linha de base 2D colapsada (grade 1 x 1, soma dos blocos apos o 1o
    minuto, com a receita do controle) contra o mosaico 3 x 3 e o R3D-18 de
    referencia, na rodada final. Descritiva, sem regra de decisao. Grava as
    predicoes por estudo dos tres metodos (media das sementes) para as
    figuras do artigo."""
    txt = ["## Três métodos: 2D colapsado, mosaico 3 × 3 e R3D-18 (folds 0 a 4, "
           "sementes 5042 a 8042)", ""]
    nc, nm = "2D colapsado (grade 1 × 1)", "mosaico 3 × 3 (controle)"
    nr = "R3D-18 com cabeça espacial e aquecimento"
    preds, falt = _ler_bracos({nc: q.COLAPSADO, nm: q.CTL, nr: q.R3D_REF},
                              fontes)
    if falt:
        return txt + [f"Braços incompletos: {', '.join(falt)}."]
    t2 = [dict(comparação=f"{a} menos {b}", **comparar_com(preds[x], preds[y]))
          for a, x, b, y in (("2D colapsado", nc, "mosaico 3 × 3", nm),
                             ("2D colapsado", nc, "R3D-18", nr),
                             ("R3D-18", nr, "mosaico 3 × 3", nm))]
    reg = dec.get("colapsado", {}).get("comparacoes", {})
    conf = "; ".join(f"{k}: Δ médio {fmt(v.get('delta_medio'))}, "
                     f"{v.get('vitorias')} folds vencidos ({v.get('status')})"
                     for k, v in reg.items())
    curto = {nc: "colapsado_1x1", nm: "mosaico_3x3", nr: "r3d18"}
    info = preds[nm].groupby("study").agg(y=("y", "first"),
                                          coorte=("coorte", "first"),
                                          fold=("fold", "first"))
    info.join(pd.DataFrame({curto[n]: d.groupby("study").pred.mean()
                            for n, d in preds.items()})).to_csv(saida_csv)
    return txt + [
        "Linha de base definida antes das execuções como descritiva, sem regra de decisão. "
        "Mesmas definições da rodada final.", "",
        md_tabela(pd.DataFrame([linha_metricas(n, d)
                                for n, d in preds.items()])), "",
        "Δ = primeiro menos segundo (pp; negativo = erro menor do primeiro); "
        "\"folds vencidos\" = folds em que o primeiro erra menos. O veredito "
        "do R3D-18 contra o mosaico está na rodada final.", "",
        md_tabela(pd.DataFrame(t2)), "",
        "Registro da fila (`fase4_decisoes.json`, chave `colapsado`): "
        f"{conf or 'ausente'}.", "",
        "MAE por coorte e faixa (erro por estudo médio entre as sementes).", "",
        md_tabela(mae_por_grupo(preds)), "",
        "Bland-Altman (pred − y), por semente nos 207 estudos; viés com a "
        "faixa entre sementes e limites de 95% médios.", "",
        _tabela_ba(preds), "",
        f"Predições por estudo (média das sementes): `{Path(saida_csv).name}`.",
        ""]


# --------------------------------------------------- validacao interna
# Execucao que nao convergiu: MAE de treino nunca abaixo deste valor (pp).
# Criterio descritivo, definido depois de ver os historicos de treino (nas 120
# execucoes dos tres metodos, com e sem validacao interna, o maior MAE de
# treino minimo das demais e 6,98 pp e o da excecao, 16,97 pp).
LIMIAR_NAO_CONVERGIU = 10.0
CURTO = {"2D colapsado (grade 1 × 1)": "colapsado_1x1",
         "mosaico 3 × 3": "mosaico_3x3",
         "R3D-18 com cabeça espacial e aquecimento": "r3d18"}


def _bracos_artigo(vi):
    """Os tres metodos do artigo (rotulo -> configuracao), com ou sem a
    validacao interna; a ordem e a das tabelas."""
    if vi:
        b = q.BRACOS_VI
        cfgs = (b["colapsado_1x1_vi"], b["mosaico_3x3_vi"],
                b["r3d_cab_espacial_aq3_vi"])
    else:
        cfgs = (q.COLAPSADO, q.CTL, q.R3D_REF)
    return dict(zip(CURTO, cfgs))


NPZ = ROOT / "data" / "colab_renal_lk.npz"
FAIXAS_LK = ["LK < 35", "35 a 45", "45 a 55", "55 a 65", "LK >= 65"]


def faixas_lk(y):
    """Faixas da LK de referencia nos cortes da estratificacao das particoes
    (f3.FAIXAS_SPLIT: 35, 45, 55 e 65), fixados antes de qualquer resultado;
    nao sao cortes clinicos."""
    return pd.cut(np.asarray(y, float), [-np.inf, *f3.FAIXAS_SPLIT, np.inf],
                  right=False, labels=FAIXAS_LK)


def pdi_por_estudo(npz=NPZ):
    """PDI por estudo real, crua e recalibrada como em f3.referencias_pdi
    (reta nos 4 folds de treino de cada particao fixa, recorte em [0, 100]),
    e o numero de hibridos (so no treino) por faixa de LK. (None, None) sem
    o npz."""
    if not Path(npz).exists():
        return None, None
    with np.load(npz, allow_pickle=False) as z:
        y, study, cru = z["y"], z["study"], z["lk_cru"]
        hib = z["hibrido"].astype(bool)
    partes = []
    for f, (tr, te) in enumerate(f3.montar_folds(dict(study=study, hibrido=hib))):
        a, b = np.polyfit(cru[tr], y[tr], 1)
        partes.append(pd.DataFrame(dict(
            study=study[te].astype(str), fold=f, y=y[te], pdi_crua=cru[te],
            pdi_recal=np.clip(a * cru[te] + b, 0, 100))))
    hibridos = pd.Series(faixas_lk(y[hib])).value_counts().reindex(FAIXAS_LK)
    return pd.concat(partes).set_index("study"), hibridos


PDI_COLS = (("PDI recalibrada", "pdi_recal"), ("PDI crua", "pdi_crua"))


def linha_pdi(rot, p, col):
    """Metricas da PDI (sem sementes) no formato de linha_metricas: MAE,
    RMSE, r e R2 por fold (media dos 5 folds) e, nos 207 estudos, a
    inclinacao e os vieses nas caudas."""
    ex = pd.DataFrame([f3.metricas(g[col].to_numpy(float), g.y.to_numpy(float))
                       for _, g in p.groupby("fold")])
    e = p[col] - p.y
    return {"braço": rot, "MAE": f"{ex.mae.mean():.2f} ± {ex.mae.std(ddof=1):.2f}",
            "RMSE": fmt(ex.rmse.mean()), "r": fmt(ex.r.mean(), 3),
            "R2": fmt(ex.r2.mean(), 3), "MAE com TTA": "",
            "inclinação": fmt(np.polyfit(p.y, p[col], 1)[0], 3),
            "viés LK < 35": fmt(e[p.y < 35].mean()),
            "viés LK >= 65": fmt(e[p.y >= 65].mean())}


def ba_pdi(rot, p, col):
    """Bland-Altman da PDI nos 207 estudos (sem sementes)."""
    dif = p[col] - p.y
    vies, dp = dif.mean(), dif.std(ddof=1)
    return {"braço": rot, "viés": f"{vies:+.2f}",
            "limite inferior": fmt(vies - 1.96 * dp),
            "limite superior": fmt(vies + 1.96 * dp),
            "amplitude": fmt(2 * 1.96 * dp)}


def ler_historicos(cfg, fontes, sementes=q.SEM_FINAL, folds=q.F5):
    """Historico por epoca de cada execucao (uma linha por semente, fold e
    epoca), da primeira fonte que tiver o arquivo."""
    linhas = []
    for s in sementes:
        tag = f3.tag_config(dict(cfg, seed=s))
        for f in folds:
            for pasta, nome in fontes:
                arq = Path(pasta) / f"historicos_{nome}" / f"hist_{tag}_fold{f}.csv"
                if arq.exists():
                    h = pd.read_csv(arq)
                    h["semente"], h["fold"] = s, f
                    linhas.append(h)
                    break
    return pd.concat(linhas, ignore_index=True) if linhas else None


def otimismo_selecao(hist):
    """Por execucao: MAE do fold avaliado (historico, sem recorte) na epoca
    escolhida pela validacao interna (menor MSE do fold seguinte, a regra do
    treino) e na epoca de menor MSE do proprio fold avaliado (a escolha das
    rodadas sem validacao interna, aplicada a mesma trajetoria)."""
    out = []
    for (s, f), h in hist.groupby(["semente", "fold"]):
        h = h.reset_index(drop=True)
        i_sel, i_ref = h.mse_va.idxmin(), h.mse_te.idxmin()
        out.append(dict(semente=s, fold=f, ep_sel=int(h.epoca[i_sel]),
                        ep_ref=int(h.epoca[i_ref]), mae_sel=h.mae_te[i_sel],
                        mae_ref=h.mae_te[i_ref]))
    return pd.DataFrame(out)


def sem_execucoes(d, excluir):
    """Predicoes sem as execucoes (semente, fold) listadas."""
    if not excluir:
        return d
    chave = pd.MultiIndex.from_frame(d[["semente", "fold"]])
    return d[~chave.isin(excluir)]


def _mae_fold(d):
    pf = por_execucao(d).groupby("fold").mae.mean()
    return f"{pf.mean():.2f} ± {pf.std(ddof=1):.2f}"


def secao_validacao_interna(fontes, dec, saida_csv, pdi=None):
    """Os tres metodos do artigo com validacao interna (o fold seguinte,
    (f + 1) % 5, reduz o lr, para o treino e escolhe a melhor epoca; o fold
    avaliado so e previsto no fim). R3D-18 contra o mosaico pela regra fixada
    antes das execucoes (descrita no artigo); o 2D colapsado e descritivo.
    Grava as predicoes por estudo (media das sementes, mais a PDI se houver)
    para as figuras do artigo."""
    txt = ["## Três métodos com validação interna (folds 0 a 4, sementes 5042 a "
           "8042)", ""]
    vi = _bracos_artigo(True)
    nc, nm, nr = vi
    preds, falt = _ler_bracos(vi, fontes)
    if falt:
        return txt + [f"Braços incompletos: {', '.join(falt)}."]
    reg = dec.get("validacao_interna", {}).get("comparacoes", {})

    t1 = [linha_metricas(n, d) for n, d in preds.items()]
    arq_f = Path(fontes[0][0]) / f"folds_{fontes[0][1]}.csv"
    tam = ""
    if arq_f.exists():
        r = pd.read_csv(arq_f)
        r = r[r.tag == f3.tag_config(dict(vi[nm], seed=q.SEM_FINAL[0]))]
        r = r.drop_duplicates("fold", keep="last").sort_values("fold")
        if len(r) == len(q.F5):
            if pdi is None:          # sem o npz: so o MAE gravado no treino
                for rot, col in (("PDI recalibrada", "mae_pdi_recal"),
                                 ("PDI crua", "mae_pdi_crua")):
                    t1.append({"braço": rot, "MAE": f"{r[col].mean():.2f} ± "
                                                   f"{r[col].std(ddof=1):.2f}"})
            tam = "; ".join(f"fold {int(x.fold)}: validação no fold "
                            f"{int(x.fold_validacao)}, {int(x.n_treino_real)} "
                            f"reais no treino e {int(x.n_validacao)} na "
                            "validação" for x in r.itertuples())
    if pdi is not None:              # PDI estudo a estudo: todas as metricas
        t1 += [linha_pdi(rot, pdi, col) for rot, col in PDI_COLS]

    t2 = [dict(comparação=rot, **comparar_com(preds[a], preds[b]),
               veredito=reg.get(chave, {}).get("veredito", "descritivo"))
          for rot, a, b, chave in (
              ("R3D-18 menos mosaico 3 × 3", nr, nm, "r3d_menos_mosaico"),
              ("2D colapsado menos mosaico 3 × 3", nc, nm, ""),
              ("2D colapsado menos R3D-18", nc, nr, ""))]
    conf = "; ".join(f"{k}: Δ médio {fmt(v.get('delta_medio'))}, "
                     f"{v.get('vitorias')} folds vencidos"
                     + (f", {v['veredito']}" if "veredito" in v else
                        f" ({v.get('status')})")
                     for k, v in reg.items())

    info = preds[nm].groupby("study").agg(y=("y", "first"),
                                          coorte=("coorte", "first"),
                                          fold=("fold", "first"))
    tab = info.join(pd.DataFrame({CURTO[n]: d.groupby("study").pred.mean()
                                  for n, d in preds.items()}))
    if pdi is not None:
        tab = tab.join(pdi[["pdi_recal", "pdi_crua"]])
    tab.to_csv(saida_csv)
    return txt + [
        "Validação interna: em cada fold f, o fold (f + 1) mod 5 reduz o lr, "
        "para o treino e escolhe a melhor época; o fold f, de teste, é previsto "
        "com o checkpoint escolhido (o erro nele, registrado a cada época, serve "
        "só para diagnóstico). "
        "A PDI continua recalibrada nos 4 folds de treino (não escolhe nada no "
        "fold avaliado). Mesmas definições da rodada final." +
        (f" Tamanhos: {tam}." if tam else ""), "",
        md_tabela(pd.DataFrame(t1).fillna("")), "",
        "Δ = primeiro menos segundo (pp; negativo = erro menor do primeiro); "
        "\"folds vencidos\" = folds em que o primeiro erra menos. Regra do "
        "R3D-18 contra o mosaico: fixada antes das execuções (Δ médio < −1 pp "
        "ou 5 de 5 folds; caso contrário, empate técnico); o 2D colapsado é "
        "descritivo.", "",
        md_tabela(pd.DataFrame(t2)), "",
        "Registro da fila (`fase4_decisoes.json`, chave `validacao_interna`): "
        f"{conf or 'ausente'}.", "",
        "MAE por coorte e faixa (erro por estudo médio entre as sementes).", "",
        md_tabela(mae_por_grupo(preds)), "",
        "Bland-Altman (pred − y), por semente nos 207 estudos; viés com a "
        "faixa entre sementes e limites de 95% médios."
        + (" A PDI, sem sementes, nos 207 estudos; nas linhas da PDI da tabela "
           "principal, MAE, RMSE, r e R2 são por fold e a inclinação e os vieses "
           "nos 207 estudos." if pdi is not None else ""), "",
        _tabela_ba(preds, [ba_pdi(rot, pdi, col) for rot, col in PDI_COLS]
                   if pdi is not None else None), "",
        f"Predições por estudo (média das sementes): `{Path(saida_csv).name}`.",
        ""]


def _media_ic(x):
    """Media com IC 95% por bootstrap dos estudos (mesmo sorteio de
    bootstrap_dif)."""
    x = np.asarray(x, float)
    m, li, ls = bootstrap_dif(x, np.zeros(len(x)))
    return f"{m:.2f} ({li:.2f} a {ls:.2f})"


def secao_erro_por_faixa(fontes, pdi, hibridos):
    """Erro dos tres metodos com validacao interna e da PDI recalibrada por
    faixa da LK de referencia, com o numero de estudos reais e de hibridos (so
    no treino) em cada faixa. Descritivo, sem regra."""
    txt = ["## Erro por faixa de LK (descritivo)",
           ""]
    vi = _bracos_artigo(True)
    nc, nm, nr = vi
    preds, falt = _ler_bracos(vi, fontes)
    if falt:
        return txt + [f"Braços incompletos: {', '.join(falt)}."]
    ep = {}
    for n, d in preds.items():
        e = d.assign(ea=(d.pred - d.y).abs(), es=d.pred - d.y)
        ep[n] = e.groupby("study").agg(ea=("ea", "mean"), es=("es", "mean"),
                                       y=("y", "first"), fold=("fold", "first"))
    idx = ep[nm].index
    ep = {n: e.loc[idx] for n, e in ep.items()}
    confer = ""
    if pdi is not None:
        p = pdi.loc[idx]
        if (not np.allclose(p.y, ep[nm].y, atol=1e-3)
                or (p.fold.to_numpy() != ep[nm].fold.to_numpy()).any()):
            raise RuntimeError("PDI por estudo nao bate com as predicoes")
        for rot, col in (("PDI recalibrada", "pdi_recal"), ("PDI crua", "pdi_crua")):
            ep[rot] = pd.DataFrame(dict(ea=(p[col] - p.y).abs(), es=p[col] - p.y,
                                        y=p.y, fold=p.fold))
        arq_f = Path(fontes[0][0]) / f"folds_{fontes[0][1]}.csv"
        if arq_f.exists():
            r = pd.read_csv(arq_f)
            r = r[r.tag == f3.tag_config(dict(vi[nm], seed=q.SEM_FINAL[0]))]
            r = r.drop_duplicates("fold", keep="last").set_index("fold")
            calc = ep["PDI recalibrada"].groupby("fold").ea.mean()
            dif = (calc - r.mae_pdi_recal.reindex(calc.index)).abs().max()
            confer = ("Conferência: o MAE da PDI recalibrada por fold, refeito "
                      "aqui estudo a estudo, difere do gravado no treino em no "
                      f"máximo {dif:.4f} pp.")
    fx = pd.Series(faixas_lk(ep[nm].y), index=idx)
    pontas = "pontas juntas (LK < 35 ou >= 65)"
    grupos = [(g, [g]) for g in FAIXAS_LK] + [
        (pontas, [FAIXAS_LK[0], FAIXAS_LK[-1]]), ("todas", FAIXAS_LK)]
    t1, t2, t3 = [], [], []
    for rot, fxs in grupos:
        m = fx.isin(fxs).to_numpy()
        n_est = int(m.sum())
        t1.append({"faixa": rot, "estudos reais": n_est,
                   "híbridos (só no treino)":
                       "" if hibridos is None else int(hibridos[fxs].sum()),
                   **{n: _media_ic(e.ea[m]) for n, e in ep.items()}})
        if rot != pontas:       # os vieses dos dois lados se cancelariam
            t2.append({"faixa": rot, "estudos reais": n_est,
                       **{n: f"{e.es[m].mean():+.2f}" for n, e in ep.items()}})
        lin = {"faixa": rot, "estudos reais": n_est}
        for rc, a, b in (("R3D-18 menos mosaico 3 × 3", nr, nm),
                         ("R3D-18 menos PDI recalibrada", nr, "PDI recalibrada")):
            if b in ep:
                md, li, ls = bootstrap_dif(ep[a].ea[m], ep[b].ea[m])
                lin[rc] = f"{md:+.2f} ({li:+.2f} a {ls:+.2f})"
        t3.append(lin)
    return txt + [
        "Faixas da LK de referência nos cortes da estratificação das partições "
        "(35, 45, 55 e 65), fixados antes de qualquer resultado; não são cortes "
        "clínicos. Erro por estudo médio entre as 4 sementes (três métodos com "
        "validação interna); PDI recalibrada como na tabela principal (reta nos "
        "4 folds de treino de cada partição) e PDI crua (sem ajuste nenhum). "
        "MAE com IC 95% por bootstrap dos "
        "estudos da faixa (5000 reamostragens, semente 0). Híbridos: drshyb, só "
        "no treino, em todas as partições, contados pela LK exata. A linha "
        "\"todas\" é a média por estudo, que difere da média por fold da tabela "
        "principal só pelos tamanhos dos folds.", "",
        md_tabela(pd.DataFrame(t1)), "",
        "Viés (pred − y) por faixa, médio entre os estudos da faixa (positivo = "
        "superestima a LK). Sem a linha das pontas juntas, em que os vieses de "
        "sinais opostos dos dois lados se cancelariam.", "",
        md_tabela(pd.DataFrame(t2)), "",
        "Diferenças pareadas por estudo dentro de cada faixa (primeiro menos "
        "segundo; negativo = erro menor do primeiro), IC 95% por bootstrap. "
        "Descritivas: nenhuma regra foi fixada por faixa.", "",
        md_tabela(pd.DataFrame(t3)), "",
        *([confer, ""] if confer else [])]


def secao_convergencia(fontes):
    """Execucoes com validacao interna que nao convergiram (MAE de treino
    nunca abaixo de LIMIAR_NAO_CONVERGIU) e, se houver, sensibilidade sem
    elas. Definida depois das execucoes: a regra e os numeros principais
    continuam com todas as execucoes."""
    txt = ["## Convergência do treino e sensibilidade (definida depois das execuções)", ""]
    vi = _bracos_artigo(True)
    nc, nm, nr = vi
    t, excl = [], {}
    for n, c in vi.items():
        h = ler_historicos(c, fontes)
        if h is None:
            return txt + [f"Sem históricos de {n}."]
        m = h.groupby(["semente", "fold"]).mae_tr.min()
        ruins = m[m > LIMIAR_NAO_CONVERGIU]
        excl[n] = list(ruins.index)
        bons = m[m <= LIMIAR_NAO_CONVERGIU]
        t.append({"braço": n, "execuções": len(m),
                  "MAE de treino mínimo (mediana; máximo entre as que "
                  "convergiram)": f"{bons.median():.2f}; {bons.max():.2f}",
                  "não convergiram (MAE de treino mínimo)": "; ".join(
                      f"semente {s}, fold {f} ({v:.2f})"
                      for (s, f), v in ruins.items()) or "nenhuma"})
    txt += [f"Critério descritivo, definido depois de ver os históricos: MAE de "
            f"treino nunca abaixo de {LIMIAR_NAO_CONVERGIU:g} pp em nenhuma "
            "época.", "", md_tabela(pd.DataFrame(t)), ""]
    if not any(excl.values()):
        return txt + ["Todas as execuções convergiram.", ""]
    preds, falt = _ler_bracos(vi, fontes)
    if falt:
        return txt + [f"Braços incompletos: {', '.join(falt)}."]
    p2 = {n: sem_execucoes(d, excl[n]) for n, d in preds.items()}
    t1 = [{"braço": n, "MAE com todas": _mae_fold(preds[n]),
           "MAE sem as que não convergiram": _mae_fold(p2[n]),
           "execuções fora": len(excl[n])} for n in vi]
    t2 = []
    for rot, a, b in (("R3D-18 menos mosaico 3 × 3", nr, nm),
                      ("2D colapsado menos mosaico 3 × 3", nc, nm),
                      ("2D colapsado menos R3D-18", nc, nr)):
        c = comparar_com(p2[a], p2[b])
        lin = dict(comparação=rot, **c)
        if a == nr:
            dl = (por_execucao(p2[a]).groupby("fold").mae.mean()
                  - por_execucao(p2[b]).groupby("fold").mae.mean())
            lin["pela regra do 3D"] = (
                "3D vence" if dl.mean() < q.LIMIAR_3D or (dl < 0).all()
                else "empate técnico")
        t2.append(lin)
    return txt + [
        "Sensibilidade: as mesmas métricas sem essas execuções (no fold "
        "afetado, a média é das sementes restantes). Só descritiva; a regra "
        "e os números principais usam todas as execuções.", "",
        md_tabela(pd.DataFrame(t1)), "",
        md_tabela(pd.DataFrame(t2).fillna("")), ""]


def secao_otimismo(fontes):
    """Diagnostico descritivo do otimismo das rodadas sem validacao interna, em
    que o fold avaliado tambem servia de validacao (sem regra): (a) na mesma
    trajetoria com validacao interna, o MAE do fold avaliado na epoca
    escolhida pela validacao interna contra o da epoca de menor MSE do
    proprio fold avaliado; (b) cada metodo com e sem validacao interna (mistura
    o efeito da selecao com o do treino menor, 3 folds contra 4)."""
    txt = ["## Diagnóstico do otimismo (descritivo, sem regra)", ""]
    vi, sem = _bracos_artigo(True), _bracos_artigo(False)
    p_vi, falt = _ler_bracos(vi, fontes)
    p_sem, falt2 = _ler_bracos(sem, fontes)
    if falt or falt2:
        return txt + [f"Braços incompletos: {', '.join(falt + falt2)}."]
    t1, confer = [], []
    for n, c in vi.items():
        o = otimismo_selecao(ler_historicos(c, fontes))
        pf = o.groupby("fold")[["mae_sel", "mae_ref"]].mean()
        dif = o.mae_sel - o.mae_ref
        fin = por_execucao(p_vi[n]).set_index(["semente", "fold"]).mae
        dc = (o.set_index(["semente", "fold"]).mae_sel - fin).abs()
        rec = p_vi[n].groupby(["semente", "fold"]).pred.apply(
            lambda s: bool(((s <= 0) | (s >= 100)).any()))
        confer.append((dc.max(), bool((dc > 1e-3).le(rec.reindex(dc.index)).all())))
        t1.append({"método": n,
                   "na época da validação interna": fmt(pf.mae_sel.mean()),
                   "na época de menor MSE do fold avaliado": fmt(pf.mae_ref.mean()),
                   "diferença": fmt((pf.mae_sel - pf.mae_ref).mean()),
                   "por execução (mín. a máx.)":
                       f"{dif.min():+.2f} a {dif.max():+.2f}",
                   "mesma época": f"{int((o.ep_sel == o.ep_ref).sum())} de {len(o)}",
                   "distância mediana (épocas)":
                       fmt(float(np.median((o.ep_sel - o.ep_ref).abs())), 0)})
    t2 = [dict(método=n, **{"sem validação interna": _mae_fold(p_sem[n]),
                            "com validação interna": _mae_fold(p_vi[n])},
               **comparar_com(p_vi[n], p_sem[n])) for n in vi]
    return txt + [
        "(a) Seleção da época, na mesma trajetória com validação interna: MAE "
        "do fold avaliado pelo histórico (sem recorte em [0, 100]), na época "
        "escolhida pela validação interna e na época de menor MSE do próprio "
        "fold avaliado (a escolha das rodadas anteriores). Média nos 5 folds "
        "da média das 4 sementes. Não inclui o efeito do fold avaliado sobre a "
        "redução do lr e a parada, que nas rodadas anteriores também vinham "
        "dele.", "", md_tabela(pd.DataFrame(t1)), "",
        "Conferência: diferença máxima entre o MAE do histórico na época "
        "escolhida e o MAE final das predições (com recorte), por execução: "
        + ", ".join(f"{x:.3f}" for x, _ in confer) + " pp (na ordem da "
        "tabela); diferenças acima de 0,001 pp só em execuções com alguma "
        "predição recortada em 0 ou 100: "
        + ("sim" if all(ok for _, ok in confer) else "não") + ".", "",
        "(b) Cada método com e sem validação interna. Δ = com menos sem "
        "(negativo = erro menor com validação interna; \"folds vencidos\" = "
        "folds em que a versão com validação interna erra menos). A diferença "
        "mistura a seleção no fold avaliado, o treino menor (3 folds reais em "
        "vez de 4) e as trajetórias diferentes.", "",
        md_tabela(pd.DataFrame(t2)), ""]


def _metricas_csv(df, n, cfg):
    """Metricas de modelo unico pelo CSV por fold (sem as predicoes)."""
    tags = [f3.tag_config(dict(cfg, seed=s)) for s in q.SEM_FINAL]
    d = df[df.tag.isin(tags) & df.fold.isin(q.F5)]
    pf = d.groupby("fold").mae.mean()
    return {"braço": n, "MAE": f"{pf.mean():.2f} ± {pf.std(ddof=1):.2f}",
            "RMSE": fmt(d.rmse.mean()), "r": fmt(d.r.mean(), 3),
            "R2": fmt(d.r2.mean(), 3),
            "MAE com TTA": fmt(d.groupby("fold").mae_tta.mean().mean())}


def secao_colab(pasta, fontes_locais):
    """Rodada final do Colab (A100), R(2+1)D-18 contra o controle do Colab. A
    decisao vem de q.decidir_colab (regra do 3D) e fica gravada em
    <pasta>/fase4_colab_decisao.json. Sem as predicoes por estudo, as metricas
    vem do CSV por fold."""
    titulo = ("R(2+1)D-18 contra o controle do Colab (folds 0 a 4, sementes "
              "5042 a 8042)")
    txt = ["", "# Colab (A100)", ""]
    arq = pasta / f"folds_{q.NOME}.csv"
    if not arq.exists():
        return txt + [f"## {titulo}", "", "Sem resultados do Colab.", ""]
    dec = q.decidir_colab(q.Leitor(fontes=((pasta, q.NOME),)))
    (pasta / "fase4_colab_decisao.json").write_text(
        json.dumps(q._json(dec), ensure_ascii=False, indent=1),
        encoding="utf-8")
    c = dec["r2plus1d_contra_o_controle"]
    regra = ("Regra da Fase 3, fixada antes das execuções: o 3D vence se o "
             "Δ médio ficar abaixo de −1 pp ou se ele vencer nos 5 folds; caso "
             f"contrário, empate técnico. Veredito: **{c['veredito']}** "
             "(`fase4_colab_decisao.json`).")
    nome21 = "R(2+1)D-18 com cabeça espacial e aquecimento"
    bracos = {"controle": q.CTL, nome21: q.R21_COLAB}
    fontes = [(pasta, q.NOME)]
    _, falt = _ler_bracos(bracos, fontes)
    if not falt:
        txt += secao_final(fontes, {}, pasta / "fase4_colab_por_estudo.csv",
                           bracos=bracos, titulo=titulo, nota=regra)
    else:
        df = pd.read_csv(arq).drop_duplicates(["tag", "fold"], keep="last")
        t1 = [_metricas_csv(df, "controle do Colab", q.CTL),
              _metricas_csv(df, nome21, q.R21_COLAB)]
        pdi = df[df.tag == f3.tag_config(dict(q.CTL, seed=q.SEM_FINAL[0]))]
        if len(pdi) == len(q.F5):
            for rot, col in (("PDI recalibrada", "mae_pdi_recal"),
                             ("PDI crua", "mae_pdi_crua")):
                t1.append({"braço": rot, "MAE": f"{pdi[col].mean():.2f} ± "
                                               f"{pdi[col].std(ddof=1):.2f}"})
        t2 = {**{f"Δ fold {f}": fmt(v) for f, v in zip(q.F5, c["deltas"])},
              "Δ médio": fmt(c["delta_medio"]), "folds vencidos": c["vitorias"],
              "Δ folds 3 e 4": fmt(c["delta_folds_3_4"]),
              "veredito": c["veredito"]}
        txt += [f"## {titulo}", "",
                "Métricas pelo CSV por fold: as predições por estudo do Colab "
                "(`historicos_fase4/pred_*.csv`) não estão na pasta, e o IC por "
                "estudo, as coortes, as faixas, a inclinação, os ensembles e o "
                "Bland-Altman ficam pendentes. MAE: média nos 5 folds da média "
                "das 4 sementes em cada fold (± DP entre folds); RMSE, r e R²: "
                "média das 20 execuções.", "",
                md_tabela(pd.DataFrame(t1).fillna("")), "",
                "Δ = R(2+1)D-18 menos o controle do Colab (pp; negativo = erro "
                "menor). " + regra, "",
                md_tabela(pd.DataFrame([{"comparação": "R(2+1)D-18 menos "
                                         "controle do Colab", **t2}])), ""]
    # reprodutibilidade entre maquinas: descritivo, sem pareamento
    loc = q._braco(q.Leitor(fontes=fontes_locais), q.CTL)["mae_por_fold"]
    col = dec["bracos"]["controle_colab"]["mae_por_fold"]
    if np.all(np.isfinite(loc)) and np.all(np.isfinite(col)):
        txt += [f"Controle do Colab: {np.mean(col):.2f} ± "
                f"{np.std(col, ddof=1):.2f}; controle da 4060 na rodada final: "
                f"{np.mean(loc):.2f} ± {np.std(loc, ddof=1):.2f} (mesma "
                "configuração, sementes e folds em máquinas diferentes; só para "
                "conferir a reprodutibilidade, sem pareamento).", ""]
    return txt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pasta", default=str(q.PASTA))
    p.add_argument("--pasta-colab", default=str(q.PASTA_COLAB))
    a = p.parse_args()
    pasta = Path(a.pasta)
    fontes = [(pasta, q.NOME), (q.PASTA_F3, "fase3")]
    arq = pasta / "fase4_decisoes.json"
    dec = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}
    txt = ["# Fase 4: tabelas (geradas por src/fase4_analise.py)", "",
           f"Decisões: `{arq.name}`, atualizado em {dec.get('atualizado')}; "
           f"falhas registradas: {len(dec.get('falhas', []))}.", ""]
    txt += secao_triagem(fontes, dec)
    txt += secao_final(fontes, dec, pasta / "fase4_final_por_estudo.csv")
    txt += secao_ablacao(dec)
    if any(k in dec for k in ("atribuicao", "triagem_grades", "final_grades")):
        txt += ["# Fila 2", ""]
        txt += secao_atribuicao(fontes, dec)
        txt += secao_triagem(
            fontes, dec, chave="triagem_grades",
            titulo="Triagem das grades do mosaico (folds 0 a 2, sementes 3042 "
                   "e 4042)", rotulo_ref="3 × 3 (referência)")
        txt += secao_final_grades(fontes, dec)
    if "colapsado" in dec:
        txt += ["# Artigo: linha de base 2D colapsada", ""]
        txt += secao_colapsado(fontes, dec,
                               pasta / "fase4_tres_metodos_por_estudo.csv")
    if "validacao_interna" in dec:
        pdi, hibridos = pdi_por_estudo()
        txt += ["# Artigo: três métodos com validação interna", ""]
        txt += secao_validacao_interna(
            fontes, dec, pasta / "fase4_vi_tres_metodos_por_estudo.csv", pdi)
        txt += secao_erro_por_faixa(fontes, pdi, hibridos)
        txt += secao_convergencia(fontes)
        txt += secao_otimismo(fontes)
    if Path(a.pasta_colab).exists():
        txt += secao_colab(Path(a.pasta_colab), fontes)
    (pasta / "fase4_tabelas.md").write_text("\n".join(txt), encoding="utf-8")
    print("\n".join(txt))


if __name__ == "__main__":
    main()
