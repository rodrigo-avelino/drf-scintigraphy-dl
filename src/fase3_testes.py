# -*- coding: utf-8 -*-
"""Testes da fase3_lib e das regras das filas (investida 3D e Fase 4).

  python src/fase3_testes.py          (CPU, ~1 a 2 min)
  python src/fase3_testes.py --gpu    (+ treino curto na GPU, lote 4, 1 epoca;
                                       exige data/colab_renal_lk.npz)

Sem a pasta data/, a conferencia das tags ja gravadas (t_tags) e pulada. A
espera por PID (t_regras) usa o tasklist e so e testada no Windows. Alguns
testes baixam os pesos pre-treinados do torchvision na primeira vez.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fase3_investida as inv  # noqa: E402
import fase3_lib as f3  # noqa: E402
import fase4_fila as q4  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CPU = torch.device("cpu")


def t_tags():
    assert f3.tag_config(dict(abordagem="r3d_18")) == \
        "r3d_18_T19_moderado_bn-treinavel"
    assert f3.tag_config(dict(abordagem="r3d_18", seed=1042)) == \
        "r3d_18_T19_moderado_bn-treinavel_s1042"
    t = f3.tag_config(dict(abordagem="r2plus1d_18", bn="congelado_dominio",
                           cabeca="espacial", aquecimento=3, lr=5e-5,
                           micro_lote=8))
    assert t == ("r2plus1d_18_T19_moderado_bn-congelado_dominio_cab-espacial"
                 "_aq3_lr5e-05"), t
    t = f3.tag_config(dict(abordagem="r3d_18", cabeca="espacial",
                           aquecimento=3, sem_stride_t=True, seed=3042))
    assert t == "r3d_18_T19_moderado_bn-treinavel_cab-espacial_aq3_semstride_s3042"
    # 3.6: sufixos novos so fora do padrao
    base = "mosaico_vgg19_grade3x3_moderado_bn-treinavel"
    for cfg, esperado in (
            (dict(aquecimento=3, saida_media=True, seed=3042), "_aq3_sm_s3042"),
            (dict(aquecimento=3, agendamento="cosseno"), "_aq3_cos"),
            (dict(saida_media=True), "_sm"),
            (dict(agendamento="plato", saida_media=False), "")):
        t = f3.tag_config(dict(abordagem="mosaico_vgg19", **cfg))
        assert t == base + esperado, t
    assert f3.tag_config(dict(abordagem="mosaico_vgg19", grid=(2, 3))) == \
        "mosaico_vgg19_grade2x3_moderado_bn-treinavel"
    assert f3.tag_config(dict(abordagem="r3d_18", aquecimento=3,
                              saida_media=True, seed=3042)) == \
        "r3d_18_T19_moderado_bn-treinavel_aq3_sm_s3042"
    # toda tag ja gravada continua sendo gerada a partir das colunas do CSV
    # (colunas vazias = valor padrao; lr, cabeca, aquecimento e sem_stride_t
    # so existem a partir da 3.5.0)
    arq = ROOT / "data" / "fase3" / "folds_fase3.csv"
    if not arq.exists():        # resultados locais da Fase 3, fora do repositorio
        print("  (t_tags: sem data/fase3/folds_fase3.csv; tags gravadas nao conferidas)")
        return
    df = pd.read_csv(arq)
    for _, r in df.drop_duplicates("tag").iterrows():
        cfg = dict(abordagem=r.abordagem, aug=r.aug, bn=r.bn)
        if f3.ABORDAGENS[r.abordagem]["entrada"] == "video":
            cfg["T"] = int(r["T"])
        else:
            cfg["grid"] = tuple(int(v) for v in str(r.grid).split("x"))
        if pd.notna(r.get("semente")):
            cfg["seed"] = int(r["semente"])
        if pd.notna(r.get("lr")):
            cfg["lr"] = float(r["lr"])
        if isinstance(r.get("cabeca"), str):
            cfg["cabeca"] = r.cabeca
        if pd.notna(r.get("aquecimento")):
            cfg["aquecimento"] = int(r["aquecimento"])
        cfg["sem_stride_t"] = str(r.get("sem_stride_t")).lower() == "true"
        assert f3.tag_config(cfg) == r.tag, (cfg, r.tag)


def t_agrupar():
    xb = np.ones((2, 20, 4, 4), np.uint16)
    assert f3.agrupar_tempo(xb, 9)[:, 0].max() == 3     # 1o quadro: 3 min
    for n in (0, 20, 38):
        try:
            f3.agrupar_tempo(xb, n)
            raise AssertionError(f"aceitou {n}")
        except ValueError:
            pass


def t_cabeca():
    cab = f3.CabecaEspacial(512, 7, 7)
    cab.iniciar_vies(0.5)
    x = torch.randn(3, 512 * 49)
    assert torch.allclose(cab(x), torch.full((3, 1), 0.5))
    assert cab(x.to(torch.bfloat16)).dtype == torch.float32
    feat = torch.randn(2, 512, 1, 7, 7)            # saida do avgpool (1,7,7)
    assert torch.equal(feat.flatten(1).reshape(-1, 512, 7, 7), feat[:, :, 0])


def _forma_layer4(m, x):
    formas = {}
    h = m.layer4.register_forward_hook(          # o gancho devolve None
        lambda mod, i, o: formas.__setitem__("l4", tuple(o.shape)))
    with torch.no_grad():
        saida = m(x)
    h.remove()
    return formas["l4"], saida


def t_modelos():
    x = torch.randn(2, 3, 19, 112, 112)
    for nome in ("r3d_18", "r2plus1d_18"):
        m = f3.construir_modelo(nome, cabeca="espacial").eval()
        m.fc.iniciar_vies(0.42)
        forma, y = _forma_layer4(m, x)
        assert forma == (2, 512, 3, 7, 7), forma
        assert y.shape == (2, 1) and torch.allclose(y, torch.full((2, 1), 0.42))
    forma, _ = _forma_layer4(f3.construir_modelo("r3d_18").eval(), x[:1])
    assert forma == (1, 512, 3, 7, 7), forma
    forma, _ = _forma_layer4(
        f3.construir_modelo("r3d_18", sem_stride_t=True).eval(), x[:1])
    assert forma == (1, 512, 19, 7, 7), forma
    for args in (dict(nome="r2plus1d_18", sem_stride_t=True),
                 dict(nome="mosaico_vgg19", cabeca="espacial")):
        try:
            f3.construir_modelo(**args)
            raise AssertionError(f"aceitou {args}")
        except ValueError:
            pass


def t_micro_lote():
    torch.manual_seed(0)
    m = f3.construir_modelo("r3d_18")
    f3.modo_treino(m, "congelado")
    x, y = torch.randn(6, 3, 4, 32, 32), torch.rand(6)
    crit = nn.MSELoss()
    grads = []
    for ml in (None, 4):                   # 4 + 2: pedacos desiguais
        m.zero_grad(set_to_none=True)
        p, s = f3._gradientes(m, x, y, crit, ml, CPU, "bf16")
        grads.append(([q.grad.clone() for q in m.parameters()], p, s))
    (g1, p1, s1), (g2, p2, s2) = grads
    assert abs(p1 - p2) < 1e-6 * max(1.0, abs(p1)), (p1, p2)
    assert torch.allclose(s1, s2, atol=1e-6)
    rel = max(float((a - b).norm() / (a.norm() + 1e-12)) for a, b in zip(g1, g2))
    assert rel < 1e-4, rel
    cfg = dict(f3.PADRAO, micro_lote=8, bn="treinavel")
    try:
        f3._checar_micro_lote(cfg, m)
        raise AssertionError("aceitou micro-lote com BN treinavel")
    except ValueError:
        pass
    f3._checar_micro_lote(dict(cfg, bn="congelado_dominio"), m)


def t_bn_dominio():
    torch.manual_seed(0)
    m = f3.construir_modelo("r3d_18")
    bns = [b for b in m.modules()
           if isinstance(b, nn.modules.batchnorm._BatchNorm)]
    antes = bns[0].running_mean.clone()
    ds = torch.utils.data.TensorDataset(torch.randn(8, 3, 4, 32, 32) * 3 + 1,
                                        torch.zeros(8))
    f3.recalibrar_bn(m, torch.utils.data.DataLoader(ds, batch_size=4), CPU,
                     "bf16")
    depois = [b.running_mean.clone() for b in bns]
    assert not torch.allclose(antes, depois[0])
    f3.modo_treino(m, "congelado_dominio")
    assert m.training and all(not b.training for b in bns)
    opt = torch.optim.Adam(m.parameters(), 1e-3)
    m(torch.randn(2, 3, 4, 32, 32)).sum().backward()
    opt.step()
    assert all(torch.equal(b.running_mean, d) for b, d in zip(bns, depois))
    f3.modo_treino(m, "treinavel")
    assert all(b.training for b in bns)


def t_aquecimento():
    lrs = [f3._lr_aquecimento(1e-4, ep, i, 7, 3) for ep in range(3)
           for i in range(7)]
    assert abs(lrs[0] - 1e-4 / 21) < 1e-15 and abs(lrs[-1] - 1e-4) < 1e-15
    assert all(a < b for a, b in zip(lrs, lrs[1:]))


# ------------------------------------------------------------- lib 3.6
def t_cosseno():
    lrs = [f3._lr_cosseno(1e-4, ep, i, 7, 3, 120) for ep in range(3, 120)
           for i in range(7)]
    assert abs(lrs[0] - 1e-4) < 1e-15 and 0 < lrs[-1] < 1e-7, lrs[-1]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))
    # emenda com o aquecimento: o ultimo passo dele ja chega a lr_base
    assert abs(f3._lr_aquecimento(1e-4, 2, 6, 7, 3) - lrs[0]) < 1e-15
    assert abs(f3._lr_cosseno(1e-4, 0, 0, 7, 0, 120) - 1e-4) < 1e-15


def t_grade():
    for g in ((1, 1), (2, 2), (3, 3), (4, 4)):          # quadradas: 224 x 224
        assert f3.tamanho_mosaico(g) == (224, 224), g
    assert f3.tamanho_mosaico((2, 3)) == (183, 274)
    assert f3.tamanho_mosaico((3, 2)) == (274, 183)
    assert f3.tamanho_mosaico((3, 4)) == (194, 259)
    assert f3.tamanho_mosaico((4, 3)) == (259, 194)
    for g in ((2, 3), (3, 4)):                           # mesma area (+-1%)
        h, w = f3.tamanho_mosaico(g)
        assert abs(h * w / 224 ** 2 - 1) < 0.01 and abs(h / g[0] - w / g[1]) < 1
    assert f3.descrever_entrada(dict(abordagem="mosaico_vgg19")) == "224x224"
    assert f3.descrever_entrada(dict(abordagem="r3d_18")) == "19x112x112"


def t_tta_espelho():
    rng = np.random.default_rng(0)
    y = np.array([30.0, 70.0])
    aug = f3.PRESETS_AUG["moderado"]
    for grade in ((3, 3), (2, 3)):
        n = grade[0] * grade[1]
        C = rng.poisson(5, size=(2, n, 8, 8)).astype(np.float32)
        ds = f3.MosaicoDataset(C, y, grade, aug, treino=False)
        esp = f3.MosaicoDataset(C, y, grade, aug, treino=False, espelhar=True)
        mao = f3.MosaicoDataset(C[..., ::-1].copy(), y, grade, aug,
                                treino=False)          # cada celula espelhada
        estado = torch.get_rng_state()
        xe, ye = esp[0]
        assert torch.equal(torch.get_rng_state(), estado)   # sem sorteio
        xm, _ = mao[0]
        x0, y0 = ds[0]
        assert torch.equal(xe, xm) and not torch.equal(x0, xe)
        assert abs(float(ye) - 0.70) < 1e-6 and abs(float(y0) - 0.30) < 1e-6
        assert tuple(xe.shape) == (3, *f3.tamanho_mosaico(grade)), xe.shape
    V = rng.poisson(5, size=(2, 19, 16, 16)).astype(np.float32)
    dv = f3.VideoDataset(V, y, aug, False, 16, "kinetics", espelhar=True)
    mv = f3.VideoDataset(V[..., ::-1].copy(), y, aug, False, 16, "kinetics")
    assert torch.equal(dv[1][0], mv[1][0]) and abs(float(dv[1][1]) - 0.30) < 1e-6
    cfg = dict(abordagem="mosaico_vgg19", grid=(2, 3))
    assert isinstance(f3.criar_dataset_espelhado(cfg, C, y), f3.MosaicoDataset)
    # combinacao: media de p e 1 - p', recorte depois de combinar
    p = f3.combinar_tta(np.array([0.30, 1.20, -0.10]),
                        np.array([0.70, 0.10, 1.05]))
    assert np.allclose(p, [30.0, 100.0, 0.0]), p


def t_saida_media():
    torch.manual_seed(0)
    f3.construir_modelo("mosaico_vgg19")
    estado = torch.get_rng_state()
    torch.manual_seed(0)
    m = f3.construir_modelo("mosaico_vgg19", saida_media=True)
    assert torch.equal(torch.get_rng_state(), estado)   # mesma sequencia
    cam = f3.camada_saida(m, "mosaico_vgg19")
    assert float(cam.weight.detach().abs().sum()) == 0.0
    with torch.no_grad():
        cam.bias.fill_(0.42)
        m.eval()
        for forma in ((224, 224), f3.tamanho_mosaico((2, 3))):
            assert torch.allclose(m(torch.randn(2, 3, *forma)),
                                  torch.full((2, 1), 0.42))
        r = f3.construir_modelo("r3d_18", saida_media=True).eval()
        f3.camada_saida(r, "r3d_18").bias.fill_(0.37)
        assert torch.allclose(r(torch.randn(1, 3, 4, 32, 32)),
                              torch.full((1, 1), 0.37))
    for args in (dict(nome="r3d_18", cabeca="espacial", saida_media=True),
                 dict(nome="s3d", saida_media=True)):
        try:
            f3.construir_modelo(**args)
            raise AssertionError(f"aceitou {args}")
        except ValueError:
            pass


# ------------------------------------------------- regras da Fase 4
def _gravar_f4(pasta, nome, cfg, fold, mae, mae_tta=np.nan, versao="3.6.0"):
    arq = pasta / f"folds_{nome}.csv"
    lin = pd.DataFrame([dict(tag=f3.tag_config(cfg), fold=fold, mae=mae,
                             mae_tta=mae_tta, particao=f3.PARTICAO_ID,
                             versao=versao)])
    if arq.exists():
        lin = pd.concat([pd.read_csv(arq), lin], ignore_index=True)
    lin.to_csv(arq, index=False)


def t_regras_fase4():
    with tempfile.TemporaryDirectory() as tmp:
        p3, p4 = Path(tmp) / "f3", Path(tmp) / "f4"
        p3.mkdir()
        p4.mkdir()
        L = q4.Leitor(fontes=((p4, "fase4"), (p3, "fase3")))
        # referencia da triagem gravada pela 3.5.0 na pasta da Fase 3
        ctl = {0: (4.775, 4.179), 1: (5.369, 5.427), 2: (3.727, 3.365)}
        for f, (a, b) in ctl.items():
            _gravar_f4(p3, "fase3", dict(q4.CTL, seed=3042), f, a, versao="3.5.0")
            _gravar_f4(p3, "fase3", dict(q4.CTL, seed=4042), f, b, versao="3.5.0")
        ref = {f: np.mean(v) for f, v in ctl.items()}
        # A1 -0,30 (3 vitorias): passa; A2 -0,40 com 1 vitoria: nao passa;
        # A3 exatamente -0,25: passa (fronteira); A4 sem o fold 2: nao passa
        deltas = {"A1_aq3": (-0.3, -0.3, -0.3), "A2_sm": (-1.5, 0.15, 0.15),
                  "A3_aq3_sm": (-0.25, -0.25, -0.25),
                  "A4_aq3_cos": (-2.0, -2.0, None)}
        for nome, dl in deltas.items():
            for f, d in zip(q4.F3, dl):
                if d is None:
                    continue
                for s in q4.SEM_TRIAGEM:
                    _gravar_f4(p4, "fase4", dict(q4.CANDIDATOS_A[nome], seed=s),
                               f, ref[f] + d)
        d = q4.decidir_triagem(L)
        c = d["configuracoes"]
        assert c["A1_aq3"]["passa"] and c["A3_aq3_sm"]["passa"], c
        assert not c["A2_sm"]["passa"] and c["A2_sm"]["vitorias"] == 1
        assert not c["A4_aq3_cos"]["passa"]
        assert c["A4_aq3_cos"]["status"].startswith("incompleto")
        assert d["finalista"] == "A1_aq3" and d["passaram"] == ["A1_aq3",
                                                                 "A3_aq3_sm"]
        assert d["fontes_controle"] == ["fase3 3.5.0"], d["fontes_controle"]

        # rodada final: A com -0,30 e 4 vitorias passa; 3D com -0,36 e 4
        # vitorias fica em empate tecnico
        cfg_a = q4.CANDIDATOS_A["A1_aq3"]
        for f in q4.F5:
            for s in q4.SEM_FINAL:
                _gravar_f4(p4, "fase4", dict(q4.CTL, seed=s), f, 5.0, 4.9)
                _gravar_f4(p4, "fase4", dict(cfg_a, seed=s), f,
                           4.6 if f < 4 else 5.1)
                _gravar_f4(p4, "fase4", dict(q4.R3D_REF, seed=s), f,
                           4.5 if f < 4 else 5.2)
        dfin = q4.decidir_final(L, cfg_a)
        cp = dfin["comparacoes_com_o_controle"]
        assert cp["finalista_A"]["veredito"].startswith("mosaico refinado"), cp
        assert cp["r3d_cab_espacial_aq3"]["veredito"] == "empate tecnico", cp
        assert abs(dfin["bracos"]["controle"]["mae_tta_medio"] - 4.9) < 1e-9
        assert abs(cp["finalista_A"]["delta_folds_3_4"] + 0.15) < 1e-9
        # 3D vence nos 5 folds; A cai para 3 vitorias: controle segue
        for s in q4.SEM_FINAL:
            _gravar_f4(p4, "fase4", dict(q4.R3D_REF, seed=s), 4, 4.9)
            _gravar_f4(p4, "fase4", dict(cfg_a, seed=s), 3, 5.2)
        cp = q4.decidir_final(L, cfg_a)["comparacoes_com_o_controle"]
        assert cp["r3d_cab_espacial_aq3"]["veredito"] == "3D vence", cp
        assert cp["finalista_A"]["veredito"] == "controle segue como referencia"
        cp = q4.decidir_final(L, q4.CANDIDATOS_A["A2_sm"])[
            "comparacoes_com_o_controle"]
        assert cp["finalista_A"]["veredito"] == "incompleto"

        # ablacao: referencia na Fase 3; efeito = com o fator - sem o fator
        refb = {0: 3.77, 1: 5.44, 2: 3.87}
        ef_aq, ef_cab, ef_sm = (-0.3, -0.3, 0.1), (-0.5,) * 3, (-0.6, -0.6, 0.3)
        for f in q4.F3:
            b2 = refb[f] - ef_cab[f]
            for s in q4.SEM_TRIAGEM:
                _gravar_f4(p3, "fase3", dict(q4.R3D_REF, seed=s), f, refb[f],
                           versao="3.5.0")
                g = lambda nome, v: _gravar_f4(
                    p4, "fase4", dict(q4.ABLACAO_B[nome], seed=s), f, v)
                g("B1_cab_espacial_sem_aq", refb[f] - ef_aq[f])
                g("B2_cab_padrao_sm_aq3", b2)
                g("B3_r3d_exploratorio", b2 - ef_sm[f])
        lt = q4.decidir_ablacao(L)["leituras"]
        assert lt["aquecimento_dada_a_cabeca_espacial"]["leitura"] == \
            "nao detectado", lt
        assert lt["cabeca_espacial_em_si"]["leitura"].startswith(
            "detectado: o fator reduz"), lt
        assert lt["saida_media_e_aquecimento_com_media_global"][
            "leitura"].startswith("detectado: o fator reduz"), lt

        # recontrole: linha da Fase 4 tem prioridade sobre a da Fase 3
        _gravar_f4(p4, "fase4", dict(q4.CTL, seed=3042), 0, 9.9)
        assert L.mae(dict(q4.CTL, seed=3042), 0) == 9.9
    assert q4.args_cli(dict(q4.CTL, aquecimento=3, saida_media=True,
                            agendamento="cosseno", grid=(2, 3), seed=5042)) \
        == ["--abordagens", "mosaico_vgg19", "--seed", "5042",
            "--aquecimento", "3", "--agendamento", "cosseno",
            "--saida-media", "--grid", "2", "3"]
    cmd = q4.cmd_rodar(dict(q4.R3D_REF, seed=5042), q4.F5)
    assert cmd[-4:] == ["--saida", str(q4.PASTA), "--nome-base", "fase4"]


def _gravar_sementes(pasta, cfg, valores, sementes, nome="fase4", folds=None):
    """Mesmo MAE em todas as sementes de cada fold (valores por fold)."""
    folds = q4.F5 if folds is None else folds
    for f, v in zip(folds, valores):
        if v is None:
            continue
        for s in sementes:
            _gravar_f4(pasta, nome, dict(cfg, seed=s), f, v, v - 0.1)


def t_regras_fila2():
    # tags e linha de comando das configuracoes da fila 2 (R3D_T9 e GRADES)
    assert f3.tag_config(dict(q4.R3D_T9, seed=5042)) == \
        "r3d_18_T9_moderado_bn-treinavel_cab-espacial_aq3_s5042"
    assert f3.tag_config(dict(q4.GRADES["G_4x3"], seed=3042)) == \
        "mosaico_vgg19_grade4x3_moderado_bn-treinavel_s3042"
    assert list(q4.GRADES) == ["G_2x3", "G_3x2", "G_3x4", "G_4x3"]
    assert q4.args_cli(dict(q4.R3D_T9, seed=5042)) == [
        "--abordagens", "r3d_18", "--T", "9", "--seed", "5042",
        "--cabeca", "espacial", "--aquecimento", "3"]
    assert q4.args_cli(dict(q4.GRADES["G_3x2"], seed=4042)) == [
        "--abordagens", "mosaico_vgg19", "--seed", "4042", "--grid", "3", "2"]
    assert q4._min_fold(q4.R3D_T9) == q4.MIN_FOLD["r3d_cab_T9"]
    assert q4._min_fold(q4.R3D_REF) == q4.MIN_FOLD["r3d_cab"]

    with tempfile.TemporaryDirectory() as tmp:
        p3, p4 = Path(tmp) / "f3", Path(tmp) / "f4"
        p3.mkdir()
        p4.mkdir()
        L = q4.Leitor(fontes=((p4, "fase4"), (p3, "fase3")))
        # triagem das grades: mesma referencia 3 x 3 gravada na Fase 3
        ctl = (4.477, 5.398, 3.546)
        _gravar_sementes(p3, q4.CTL, ctl, q4.SEM_TRIAGEM, "fase3", q4.F3)
        deltas = {"G_2x3": (-0.3, -0.3, -0.3), "G_3x2": (-0.5, 0.1, 0.1),
                  "G_3x4": (-0.25, -0.25, -0.25), "G_4x3": (-2.0, -2.0, None)}
        for nome, dl in deltas.items():
            _gravar_sementes(p4, q4.GRADES[nome],
                             [None if d is None else c + d
                              for c, d in zip(ctl, dl)], q4.SEM_TRIAGEM,
                             folds=q4.F3)
        d = q4.decidir_triagem(L, q4.GRADES)
        c = d["configuracoes"]
        assert c["G_2x3"]["passa"] and c["G_3x4"]["passa"], c
        assert not c["G_3x2"]["passa"] and c["G_3x2"]["vitorias"] == 1
        assert c["G_4x3"]["status"].startswith("incompleto")
        assert d["finalista"] == "G_2x3" and d["passaram"] == ["G_2x3",
                                                                "G_3x4"]
        assert set(q4.decidir_triagem(L)["configuracoes"]) == set(
            q4.CANDIDATOS_A)        # o padrao continua sendo o caminho A

        # rodada final da grade: -0,30 com 4 vitorias passa; o 3D vence a
        # grade nos 5 folds; com 4 vitorias e -0,3 fica em empate tecnico
        g = q4.GRADES["G_2x3"]
        _gravar_sementes(p4, q4.CTL, [5.0] * 5, q4.SEM_FINAL)
        _gravar_sementes(p4, g, [4.6, 4.6, 4.6, 4.6, 5.1], q4.SEM_FINAL)
        _gravar_sementes(p4, q4.R3D_REF, [4.2] * 5, q4.SEM_FINAL)
        dg = q4.decidir_final_grade(L, g)
        assert dg["grade_contra_3x3"]["veredito"].startswith("grade passa"), dg
        assert dg["grade_contra_3x3"]["vitorias"] == 4
        assert dg["r3d_contra_a_grade"]["veredito"] == "3D vence", dg
        assert abs(dg["bracos"]["grade"]["mae_tta_medio"] - 4.6) < 1e-9
        _gravar_sementes(p4, q4.R3D_REF, [4.2, 4.2, 4.2, 4.2, 5.1],
                         q4.SEM_FINAL)
        dg = q4.decidir_final_grade(L, g)
        assert dg["r3d_contra_a_grade"]["veredito"] == "empate tecnico", dg
        # grade reprovada: sem comparacao do 3D com ela
        _gravar_sementes(p4, g, [4.6, 4.6, 5.1, 5.1, 5.1], q4.SEM_FINAL)
        dg = q4.decidir_final_grade(L, g)
        assert dg["grade_contra_3x3"]["veredito"] == "3x3 segue como referencia"
        assert "r3d_contra_a_grade" not in dg
        assert q4.decidir_final_grade(L, q4.GRADES["G_3x2"])[
            "grade_contra_3x3"]["veredito"] == "incompleto"

        # atribuicao: T = 19 a 4,2 e mosaico a 4,5 em todos os folds
        _gravar_sementes(p4, q4.R3D_REF, [4.2] * 5, q4.SEM_FINAL)

        def atrib(t9):
            _gravar_sementes(p4, q4.R3D_T9, t9, q4.SEM_FINAL)
            return q4.decidir_atribuicao(L)

        _gravar_sementes(p4, q4.CTL, [4.5] * 5, q4.SEM_FINAL)
        r = atrib([4.25, 4.15, 4.2, 4.3, 4.1])        # T = 9 igual ao T = 19
        assert r["leituras"]["T9_menos_T19"]["leitura"] == "nao detectado"
        assert r["leituras"]["T9_menos_mosaico"]["leitura"].startswith(
            "detectado: T = 9 erra menos"), r["leituras"]
        assert r["interpretacao"].startswith("ganho do 3D mantido"), r
        r = atrib([4.55, 4.45, 4.5, 4.6, 4.4])        # T = 9 igual ao mosaico
        assert r["leituras"]["T9_menos_T19"]["leitura"].startswith(
            "detectado: T = 9 erra mais")
        assert r["interpretacao"] == "ganho do 3D dependente dos 19 quadros", r
        r = atrib([4.35] * 5)                           # a meio caminho
        assert r["interpretacao"] == "inconclusivo", r
        # fronteira: -0,25 com 4 de 5 folds no mesmo sinal e detectado
        r = atrib([4.15, 4.15, 4.15, 4.15, 4.65])
        lm = r["leituras"]["T9_menos_mosaico"]
        assert lm["detectado"] and lm["folds_com_o_mesmo_sinal"] == 4, lm
        # 3 de 5 no mesmo sinal: nao detectado, mesmo com media grande
        r = atrib([3.5, 3.5, 3.5, 4.7, 4.7])
        assert not r["leituras"]["T9_menos_mosaico"]["detectado"]
        # T = 9 muito melhor que o T = 19: o ganho continua mantido
        r = atrib([3.8] * 5)
        assert r["interpretacao"].startswith("ganho do 3D mantido"), r
        # fold faltante: incompleto
        arq = p4 / "folds_fase4.csv"
        df = pd.read_csv(arq)
        tag9 = f3.tag_config(dict(q4.R3D_T9, seed=8042))
        df[~((df.tag == tag9) & (df.fold == 4))].to_csv(arq, index=False)
        r = q4.decidir_atribuicao(L)
        assert r["interpretacao"] == "incompleto", r
        assert r["leituras"]["T9_menos_T19"]["leitura"] == "incompleto"


def t_regras_colab():
    # tag do R(2+1)D-18 do notebook do Colab
    assert f3.tag_config(dict(q4.R21_COLAB, seed=5042)) == \
        "r2plus1d_18_T19_moderado_bn-treinavel_cab-espacial_aq3_lr5e-05_s5042"
    sem = q4.SEM_FINAL
    with tempfile.TemporaryDirectory() as tmp:
        pc, p4 = Path(tmp) / "colab", Path(tmp) / "f4"
        pc.mkdir()
        p4.mkdir()
        # controle da 4060 bem pior: nao pode entrar na comparacao do Colab
        _gravar_sementes(p4, q4.CTL, [9.0] * 5, sem)
        _gravar_sementes(pc, q4.CTL, [4.5] * 5, sem)
        Lc = q4.Leitor(fontes=((pc, "fase4"),))

        def colab(r21):
            arq = pc / "folds_fase4.csv"
            df = pd.read_csv(arq)
            tag = f3.tag_config(q4.R21_COLAB)
            df[~df.tag.str.startswith(tag)].to_csv(arq, index=False)
            _gravar_sementes(pc, q4.R21_COLAB, r21, sem)
            return q4.decidir_colab(Lc)["r2plus1d_contra_o_controle"]

        c = colab([4.4] * 5)                      # 5 de 5, delta pequeno
        assert c["veredito"] == "3D vence" and c["vitorias"] == 5, c
        c = colab([3.0, 3.0, 3.0, 3.0, 4.6])      # delta -1,18, 4 de 5
        assert c["veredito"] == "3D vence", c
        c = colab([3.5, 3.5, 3.5, 3.5, 4.6])      # delta -0,78, 4 de 5
        assert c["veredito"] == "empate tecnico", c
        c = colab([4.9, 4.8, 5.9, 4.7, 3.9])      # 3D pior: empate tecnico
        assert c["veredito"] == "empate tecnico" and c["vitorias"] == 1, c
        assert abs(c["delta_medio"] - 0.34) < 1e-9, c
        c = colab([4.4, 4.4, 4.4, 4.4, None])     # fold faltante
        assert c["veredito"] == "incompleto", c


def t_regras_colapsado():
    # linha de base 2D colapsada do artigo: grade 1 x 1 com a receita do controle
    assert f3.tag_config(dict(q4.COLAPSADO, seed=5042)) == \
        "mosaico_vgg19_grade1x1_moderado_bn-treinavel_s5042"
    assert q4.args_cli(dict(q4.COLAPSADO, seed=6042)) == [
        "--abordagens", "mosaico_vgg19", "--seed", "6042", "--grid", "1", "1"]
    assert q4._min_fold(q4.COLAPSADO) == q4.MIN_FOLD["mosaico"]
    sem = q4.SEM_FINAL
    with tempfile.TemporaryDirectory() as tmp:
        p4 = Path(tmp) / "f4"
        p4.mkdir()
        L = q4.Leitor(fontes=((p4, "fase4"),))
        _gravar_sementes(p4, q4.CTL, [4.4, 5.4, 3.9, 4.6, 3.7], sem)
        _gravar_sementes(p4, q4.R3D_REF, [3.9, 5.1, 3.6, 4.5, 3.5], sem)
        _gravar_sementes(p4, q4.COLAPSADO, [7.4, 8.4, 6.9, None, 6.7], sem)
        r = q4.decidir_colapsado(L)["comparacoes"]
        assert r["colapsado_menos_mosaico_3x3"]["status"] == "incompleto", r
        _gravar_sementes(p4, q4.COLAPSADO, [None, None, None, 7.6, None], sem,
                         folds=q4.F5)
        r = q4.decidir_colapsado(L)["comparacoes"]
        c = r["colapsado_menos_mosaico_3x3"]
        assert c["status"] == "completo" and c["vitorias"] == 0, c
        assert abs(c["delta_medio"] - 3.0) < 1e-9, c
        c = r["colapsado_menos_r3d_cab_espacial_aq3"]
        assert abs(c["delta_medio"] - 3.28) < 1e-9, c


def t_validacao_interna():
    # 3.7 (validacao interna): tags com _vi so quando ligada; as antigas
    # continuam iguais (t_tags confere as gravadas)
    vi = q4.BRACOS_VI
    assert list(vi) == ["mosaico_3x3_vi", "r3d_cab_espacial_aq3_vi",
                        "colapsado_1x1_vi"]
    assert f3.tag_config(dict(vi["mosaico_3x3_vi"], seed=5042)) == \
        "mosaico_vgg19_grade3x3_moderado_bn-treinavel_vi_s5042"
    assert f3.tag_config(dict(vi["r3d_cab_espacial_aq3_vi"], seed=6042)) == \
        "r3d_18_T19_moderado_bn-treinavel_cab-espacial_aq3_vi_s6042"
    assert f3.tag_config(dict(vi["colapsado_1x1_vi"], seed=7042)) == \
        "mosaico_vgg19_grade1x1_moderado_bn-treinavel_vi_s7042"
    assert f3.tag_config(dict(q4.CTL, seed=5042, validacao_interna=False)) \
        == "mosaico_vgg19_grade3x3_moderado_bn-treinavel_s5042"
    assert q4.args_cli(dict(vi["colapsado_1x1_vi"], seed=8042)) == [
        "--abordagens", "mosaico_vgg19", "--seed", "8042",
        "--validacao-interna", "--grid", "1", "1"]
    assert q4.args_cli(dict(vi["r3d_cab_espacial_aq3_vi"], seed=5042)) == [
        "--abordagens", "r3d_18", "--seed", "5042", "--cabeca", "espacial",
        "--aquecimento", "3", "--validacao-interna"]

    # separacao: o fold seguinte vira validacao; o fold avaliado e os
    # hibridos ficam fora dela; o treino mantem a ordem original
    nomes = [s for v in f3.PARTICOES.values() for s in v.split()]
    nomes = list(np.random.default_rng(0).permutation(nomes)) + [
        f"drshyb_{i:03d}_H" for i in range(6)]
    dados = dict(study=np.array(nomes),
                 hibrido=np.array([n.startswith("drshyb") for n in nomes]))
    folds = f3.montar_folds(dados)
    hib = set(np.where(dados["hibrido"])[0])
    for f in range(5):
        tr, te = folds[f]
        tr2, sel = f3.separar_validacao_interna(folds, f)
        assert set(sel) == set(folds[(f + 1) % 5][1])
        assert not (set(sel) & set(te) or set(tr2) & set(sel)
                    or set(tr2) & set(te))
        assert set(tr2) | set(sel) == set(tr)
        assert not (set(sel) | set(tr2) | set(te)) & hib
        assert list(tr2) == [i for i in tr if i not in set(sel)]
        assert len(tr2) + len(sel) + len(te) == 207
    try:
        f3.separar_validacao_interna([folds[0], folds[0]], 0)
        raise AssertionError("aceitou validacao igual ao fold avaliado")
    except RuntimeError:
        pass

    # regra do 3D (igual a da rodada final) e o colapsado descritivo
    sem = q4.SEM_FINAL
    with tempfile.TemporaryDirectory() as tmp:
        p4 = Path(tmp) / "f4"
        p4.mkdir()
        L = q4.Leitor(fontes=((p4, "fase4"),))
        # a rodada final sem validacao interna nao entra nesta decisao
        _gravar_sementes(p4, q4.CTL, [9.0] * 5, sem)
        _gravar_sementes(p4, q4.R3D_REF, [1.0] * 5, sem)

        def regra(r3d, mosaico=(4.5,) * 5, colapsado=(7.0,) * 5):
            arq = p4 / "folds_fase4.csv"
            df = pd.read_csv(arq)
            df[~df.tag.str.contains("_vi_")].to_csv(arq, index=False)
            _gravar_sementes(p4, vi["mosaico_3x3_vi"], list(mosaico), sem)
            _gravar_sementes(p4, vi["r3d_cab_espacial_aq3_vi"], list(r3d), sem)
            _gravar_sementes(p4, vi["colapsado_1x1_vi"], list(colapsado), sem)
            return q4.decidir_validacao_interna(L)["comparacoes"]

        c = regra([4.4] * 5)["r3d_menos_mosaico"]            # 5 de 5
        assert c["veredito"] == "3D vence" and c["vitorias"] == 5, c
        c = regra([3.0, 3.0, 3.0, 3.0, 4.6])["r3d_menos_mosaico"]   # -1,18
        assert c["veredito"] == "3D vence", c
        c = regra([3.5, 3.5, 3.5, 3.5, 4.6])["r3d_menos_mosaico"]   # -0,78
        assert c["veredito"] == "empate tecnico", c
        c = regra([4.4, 4.4, 4.4, 4.4, None])["r3d_menos_mosaico"]
        assert c["veredito"] == "incompleto", c
        r = regra([4.2] * 5)
        c = r["colapsado_menos_mosaico_3x3_vi"]
        assert c["status"] == "completo" and abs(c["delta_medio"] - 2.5) < 1e-9
        c = r["colapsado_menos_r3d_cab_espacial_aq3_vi"]
        assert abs(c["delta_medio"] - 2.8) < 1e-9 and c["vitorias"] == 0, c
        r = regra([4.2] * 5, colapsado=(7.0, 7.0, None, 7.0, 7.0))
        assert r["colapsado_menos_mosaico_3x3_vi"]["status"] == "incompleto"


def t_validacao_interna_treino():
    # treino curto em CPU com modelo minimo: a escolha usa a validacao
    # interna, a predicao final e do fold avaliado, o historico traz o fold
    # avaliado por epoca, e medir o fold avaliado nao muda o treino (o mesmo
    # treino com a mesma validacao, sem fold avaliado separado, da as mesmas
    # perdas por epoca)
    rng = np.random.default_rng(1)
    n = 30
    rep = rng.poisson(5, size=(n, 9, 8, 8)).astype(np.float32)
    y = rng.uniform(20, 80, n).astype(np.float32)
    hib = np.zeros(n, bool)
    hib[-4:] = True
    tr, sel, te = np.arange(0, 14), np.arange(14, 20), np.arange(20, 26)
    original = f3.construir_modelo
    f3.construir_modelo = lambda *a, **k: nn.Sequential(
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 1))
    try:
        cfg = dict(f3.PADRAO, abordagem="mosaico_vgg19", batch=4)
        r_vi = f3.treinar_fold(dict(cfg, validacao_interna=True), rep, y, hib,
                               tr, te, 0, CPU, max_epocas=4, verbose=False,
                               est_sel=sel)
        r_antigo = f3.treinar_fold(cfg, rep, y, hib, tr, sel, 0, CPU,
                                   max_epocas=4, verbose=False)
    finally:
        f3.construir_modelo = original
    h, h0 = r_vi["hist"], r_antigo["hist"]
    assert {"mse_te", "mae_te"} <= set(h.columns) and len(h) == 4
    assert "mse_te" not in h0.columns
    assert len(r_vi["preds"]) == len(te) and len(r_vi["preds_tta"]) == len(te)
    assert len(r_antigo["preds"]) == len(sel)
    assert r_vi["melhor_epoca"] == int(h.mse_va.idxmin()) + 1
    assert np.allclose(h.mse_tr, h0.mse_tr) and np.allclose(h.mse_va,
                                                            h0.mse_va)


# ------------------------------------------------- regras do orquestrador
def _gravar_execucao(pasta, cfg, fold, mae, nivel=None, recal=np.nan):
    """Linha em folds_fase3.csv + historico com nivel tipico controlado."""
    arq = pasta / "folds_fase3.csv"
    lin = pd.DataFrame([dict(tag=f3.tag_config(cfg), fold=fold, mae=mae,
                             mae_bn_recal=(mae + recal) if np.isfinite(recal)
                             else np.nan, particao=f3.PARTICAO_ID)])
    if arq.exists():
        lin = pd.concat([pd.read_csv(arq), lin], ignore_index=True)
    lin.to_csv(arq, index=False)
    nv = mae if nivel is None else nivel
    h = pd.DataFrame(dict(epoca=range(1, 31), mse_va=np.linspace(1, 2, 30),
                          mae_va=[nv] * 30))
    h.loc[0, "mse_va"] = 0.5                       # melhor epoca = 1
    (pasta / "historicos_fase3").mkdir(exist_ok=True)
    h.to_csv(pasta / "historicos_fase3"
             / f"hist_{f3.tag_config(cfg)}_fold{fold}.csv", index=False)


def _cenario(pasta, t9, f1_novas, ctl1042, recal=(0.7, 0.8)):
    R3D, CTL = inv.R3D, inv.CTL
    for f, v in zip(inv.F3, (4.45, 6.58, 4.04)):
        _gravar_execucao(pasta, R3D, f, v)
    for s, v, rc in zip((1042, 2042), f1_novas, recal):
        _gravar_execucao(pasta, dict(R3D, seed=s), 1, v, recal=rc)
    for f, v in zip(inv.F3, t9):
        _gravar_execucao(pasta, dict(R3D, T=9), f, v)
    for f, (a, b) in enumerate(zip((4.43, 4.90, 4.33), ctl1042)):
        _gravar_execucao(pasta, CTL, f, a)
        _gravar_execucao(pasta, dict(CTL, seed=1042), f, b)


def t_regras():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _cenario(p, t9=(6.6, 7.0, 5.5), f1_novas=(5.0, 5.2),
                 ctl1042=(4.6, 4.8, 4.2))
        L = inv.Leitor(p)
        d1 = inv.decidir_e1(L)
        assert d1["leitura_T"] == "T importa", d1
        assert d1["leitura_fold1"].startswith("acaso"), d1
        assert abs(d1["sigma_3d_fold1"] - np.std([6.58, 5.0, 5.2], ddof=1)) < 1e-9
        assert abs(d1["limiar_e3"] + 0.8 * np.sqrt(
            d1["sigma_3d_fold1"] ** 2 + d1["sigma_controle"] ** 2 / 2)) < 1e-6
        assert d1["previsao_recal"].startswith("confirmada")
        # dado faltando: E2 mantem o padrao, E3 e E4 ficam "incompleto"
        d2i = inv.decidir_e2(L, d1)
        assert d2i["bn_pacote"] == "congelado_dominio", d2i
        assert d2i["status"] == "incompleto"
        ci = inv.candidatos_e3(d1, d2i)
        d3i = inv.decidir_e3(L, d1, ci)
        assert d3i["vencedor"] is None and set(d3i["incompletos"]) == set(ci)
        d4i = inv.decidir_e4(L, ci["r3d_pacote"])
        assert d4i["veredito"] == "incompleto" and len(d4i["faltantes"]) == 20
        # E2: congelado bem pior -> recuo para BN treinavel, sem R(2+1)D
        for f, v in zip(inv.F3, (7.0, 8.0, 7.5)):
            _gravar_execucao(p, inv.R3D_CONG, f, v)
        d2 = inv.decidir_e2(L, d1)
        assert d2["bn_pacote"] == "treinavel", d2
        c = inv.candidatos_e3(d1, d2)
        assert set(c) == {"r3d_pacote", "r3d_pacote_sem_stride"}, c
        assert c["r3d_pacote"]["bn"] == "treinavel"

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp)
        _cenario(p, t9=(4.5, 5.6, 4.3), f1_novas=(4.6, 4.9),
                 ctl1042=(4.5, 4.7, 4.4), recal=(0.2, 0.9))
        L = inv.Leitor(p)
        d1 = inv.decidir_e1(L)
        assert d1["leitura_T"] == "indiferente", d1
        assert d1["previsao_recal"] == "nao confirmada"
        sig = d1["sigma_3d_fold1"]
        assert sig > 0.8 and d1["limiar_e3"] < -0.5
        for f, v in zip(inv.F3, (4.5, 5.2, 4.3)):
            _gravar_execucao(p, inv.R3D_CONG, f, v)
        d2 = inv.decidir_e2(L, d1)
        assert d2["bn_pacote"] == "congelado_dominio", d2
        c = inv.candidatos_e3(d1, d2)
        assert set(c) == {"r3d_pacote", "r2plus1d_pacote"}, c
        assert c["r2plus1d_pacote"]["micro_lote"] == 8
        ref = {f: np.mean(v) for f, v in
               zip(inv.F3, ((4.43, 4.5), (4.90, 4.7), (4.33, 4.4)))}
        lim = d1["limiar_e3"]
        # R3D passa (media bem abaixo do limiar, 3 vitorias); R(2+1)D nao
        for f in inv.F3:
            _gravar_execucao(p, c["r3d_pacote"], f, ref[f] + lim - 0.3)
            _gravar_execucao(p, c["r2plus1d_pacote"], f,
                             ref[f] + (-2.0 if f == 0 else 0.2))
        d3 = inv.decidir_e3(L, d1, c)
        assert d3["vencedor"] == "r3d_pacote", d3
        assert not d3["configuracoes"]["r2plus1d_pacote"]["passa"]
        # E4: 4 de 5 vitorias e media -0,6 -> empate; 5 de 5 -> vence
        v = c["r3d_pacote"]
        for f in inv.F5:
            for s in inv.SEMENTES_E4:
                _gravar_execucao(p, dict(inv.CTL, seed=s), f, 5.0)
                _gravar_execucao(p, dict(v, seed=s), f,
                                 4.0 if f < 4 else 5.6)
        d4 = inv.decidir_e4(L, v)
        assert d4["vitorias"] == 4 and d4["veredito"] == "empate tecnico", d4
        _gravar_execucao(p, dict(v, seed=3042), 4, 4.9)
        _gravar_execucao(p, dict(v, seed=4042), 4, 4.9)
        assert inv.decidir_e4(L, v)["veredito"] == "3D vence"
    with tempfile.TemporaryDirectory() as tmp:     # fronteira exata: 5,3
        p = Path(tmp)
        _cenario(p, t9=(4.5, 6.5, 4.0), f1_novas=(5.2, 5.4),
                 ctl1042=(4.5, 4.7, 4.4))
        d1 = inv.decidir_e1(inv.Leitor(p))
        assert d1["leitura_fold1"].startswith("acaso"), d1["leitura_fold1"]
    # espera por PID inexistente com -X utf8 (tasklist em cp850): sem erro.
    # O tasklist so existe no Windows; em outros sistemas o teste e pulado.
    if sys.platform == "win32":
        t0 = __import__("time").time()
        inv.esperar(999999)
        assert __import__("time").time() - t0 < 15
    assert inv.args_cli(dict(inv.R21_CONG, **inv.PACOTE, lr=5e-5)) == [
        "--abordagens", "r2plus1d_18", "--bn", "congelado_dominio",
        "--lr", "5e-05", "--cabeca", "espacial", "--aquecimento", "3",
        "--micro-lote", "8"]


# --------------------------------------------- treino curto (GPU real)
def t_gpu():
    dev = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    dados = f3.carregar_dados(ROOT / "data" / "colab_renal_lk.npz")
    folds = f3.montar_folds(dados)
    est_tr, est_va = folds[0]
    cfgs = [
        dict(abordagem="r3d_18", bn="congelado_dominio", cabeca="espacial",
             aquecimento=3, micro_lote=2),
        dict(abordagem="r2plus1d_18", bn="congelado_dominio",
             cabeca="espacial", aquecimento=3, lr=5e-5, micro_lote=2),
        dict(abordagem="r3d_18", cabeca="espacial", aquecimento=3,
             sem_stride_t=True),
        dict(abordagem="r3d_18"),
        # 3.6: saida na media + cosseno no mosaico; grade nao quadrada
        dict(abordagem="mosaico_vgg19", aquecimento=1, saida_media=True,
             agendamento="cosseno"),
        dict(abordagem="mosaico_vgg19", grid=(2, 3)),
    ]
    for c in cfgs:
        cfg = dict(f3.PADRAO, **c, batch=4)
        rep = f3.preparar_representacao(dados, cfg)
        r = f3.treinar_fold(cfg, rep, dados["y"], dados["hibrido"], est_tr,
                            est_va, 0, dev, max_epocas=1, verbose=False)
        h = r["hist"].iloc[0]
        real = dados["y"][est_va]
        print(f"  gpu {f3.tag_config(cfg)}: MAE val {h.mae_va:.2f} | TTA "
              f"{f3.metricas(r['preds_tta'], real)['mae']:.2f} | "
              f"{h.segundos:.0f} s | pico {r['pico_mem_gb']:.2f} GB | "
              f"recal {r['preds_recal'] is not None}")
        assert np.isfinite(r["preds"]).all() and h.mae_va < 40
        assert np.isfinite(r["preds_tta"]).all()
        del r
        torch.cuda.empty_cache()


if __name__ == "__main__":
    testes = [t_tags, t_agrupar, t_cabeca, t_aquecimento, t_cosseno, t_grade,
              t_tta_espelho, t_micro_lote, t_bn_dominio, t_regras,
              t_regras_fase4, t_regras_fila2, t_regras_colab,
              t_regras_colapsado, t_validacao_interna,
              t_validacao_interna_treino, t_saida_media, t_modelos]
    if "--gpu" in sys.argv:
        testes.append(t_gpu)
    for t in testes:
        t()
        print(f"ok  {t.__name__}", flush=True)
    print("TODOS OS TESTES PASSARAM")
