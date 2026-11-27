# -*- coding: utf-8 -*-
"""Fase 3: biblioteca comum dos experimentos espaco-temporais.

Uma unica fonte de codigo para os dois ambientes:
  - local (RTX 4060): importada por src/fase3_rodar.py;
  - Colab (A100): embutida nos notebooks do Colab por
    src/gerar_notebook_fase3.py.
Mudancas de protocolo sao feitas AQUI e regeneradas nos dois lados.
"""
import contextlib
import copy
import datetime
import math
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Dataset, Subset, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import video as vmodels

VERSAO = "3.7.0"   # 3.1: particoes fixas; 3.2: recal. de BN embaralhada;
                   # 3.3: momentum de BN do S3D padronizado em 0,1;
                   # 3.3.1: semente na tag (_s<semente> se != 42) e T <= 19;
                   # 3.4: BN congelado com estatisticas do dominio e
                   #      micro-lotes com acumulacao de gradiente;
                   # 3.5: cabeca espacial, aquecimento do lr, R3D sem stride
                   #      temporal e trava de tempo por epoca;
                   # 3.6: TTA pela simetria (sempre registrada, sem mudar o
                   #      treino), saida iniciada na media, agendamento em
                   #      cosseno, grades nao quadradas com celulas quadradas
                   #      e nome-base dos arquivos de saida (Fase 4).
                   # 3.7: validacao interna opcional (artigo): o fold seguinte
                   #      comanda lr, parada e melhor epoca; o fold avaliado
                   #      fica fora de toda escolha.
                   # Os valores padrao reproduzem exatamente as versoes 3.3.x.

# --------------------------------------------------------------- protocolo
PADRAO = dict(
    abordagem="r2plus1d_18",
    T=19,                       # quadros do video: 19 = 1 min cada; 9 = 2 min
    grid=(3, 3),                # so para a abordagem "mosaico_vgg19"
    aug="moderado",
    lr=1e-4,
    wd=1e-4,
    batch=32,
    max_epocas=120,
    paciencia_es=25,
    paciencia_lr=8,
    alpha_sampler=0.5,
    faixas_sampler=(35, 45, 55, 65),
    usar_drshyb=True,
    precisao="bf16",            # igual nas duas maquinas (BN depende do batch)
    tam=112,                    # lado do quadro na entrada dos modelos de video
    bn="treinavel",             # "treinavel" | "congelado" (estatisticas do
                                # Kinetics) | "congelado_dominio" (reestimadas
                                # no treino real do fold e fixas no treino)
    seed=42,
    cabeca="padrao",            # "padrao" (media global) | "espacial"
    aquecimento=0,              # epocas de aquecimento linear do lr
    micro_lote=None,            # acumulacao de gradiente (exige BN fixo)
    sem_stride_t=False,         # R3D sem stride temporal nos layers 2 a 4
    limite_seg_epoca=None,      # aborta se a mediana das epocas 2 a 4 passar
    saida_media=False,          # camada final zerada e vies na media do alvo
                                # (mosaico e cabeca padrao dos modelos de video)
    agendamento="plato",        # "plato": ReduceLROnPlateau + parada antecipada;
                                # "cosseno": cosseno ate zero em max_epocas,
                                # sem parada antecipada
    validacao_interna=False,    # 3.7: True = o fold seguinte, (f + 1) % 5, sai
                                # do treino e escolhe lr, parada e melhor epoca;
                                # o fold f so e previsto no fim (e medido por
                                # epoca so para diagnostico). False = o fold
                                # avaliado faz essas escolhas (ate a 3.6)
)

REGIMES_BN = ("treinavel", "congelado", "congelado_dominio")
AGENDAMENTOS = {"plato": "", "cosseno": "_cos"}     # sufixo na tag
SAIDA_MEDIA_OK = ("mosaico_vgg19", "r3d_18", "mc3_18", "r2plus1d_18")
AREA_MOSAICO = 224          # lado do mosaico quadrado; grades nao quadradas
                            # mantem a mesma area com celulas quadradas

PRESETS_AUG = {
    "nenhum":   dict(transl=0.00, flip=False, thinning=False),
    "leve":     dict(transl=0.05, flip=False, thinning=False),
    "moderado": dict(transl=0.08, flip=True,  thinning=False),
    "pesado":   dict(transl=0.12, flip=True,  thinning=True),
}

FAIXAS_SPLIT = (35, 45, 55, 65)

NORMAS = {
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    "kinetics": ((0.43216, 0.394666, 0.37645), (0.22803, 0.22145, 0.216989)),
    "kinetics_pv": ((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
}

ABORDAGENS = {
    "r3d_18":        dict(entrada="video", norma="kinetics",
                          descricao="ResNet 3D: convolucao 3D em todas as camadas"),
    "mc3_18":        dict(entrada="video", norma="kinetics",
                          descricao="3D so nas camadas iniciais"),
    "r2plus1d_18":   dict(entrada="video", norma="kinetics",
                          descricao="R(2+1)D: espaco e tempo fatorados"),
    "s3d":           dict(entrada="video", norma="kinetics",
                          descricao="S3D: 3D separavel e leve"),
    "x3d_s":         dict(entrada="video", norma="kinetics_pv",
                          descricao="X3D: escalonamento estilo EfficientNet"),
    "vgg19_lstm":    dict(entrada="video", norma="imagenet",
                          descricao="VGG19 por quadro + LSTM bidirecional"),
    "vgg19_media":   dict(entrada="video", norma="imagenet",
                          descricao="VGG19 por quadro + media temporal (ablacao)"),
    "mosaico_vgg19": dict(entrada="mosaico", norma="imagenet",
                          descricao="controle: campeao da Fase 2"),
}


def semear(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tag_config(cfg):
    cfg = {**PADRAO, **cfg}
    if ABORDAGENS[cfg["abordagem"]]["entrada"] == "mosaico":
        rep = f"grade{cfg['grid'][0]}x{cfg['grid'][1]}"
    else:
        rep = f"T{cfg['T']}"
    tag = f"{cfg['abordagem']}_{rep}_{cfg['aug']}_bn-{cfg['bn']}"
    # opcoes da 3.4/3.5 so entram na tag fora do padrao: tags antigas intactas
    if cfg["cabeca"] != PADRAO["cabeca"]:
        tag += f"_cab-{cfg['cabeca']}"
    if cfg["aquecimento"]:
        tag += f"_aq{cfg['aquecimento']}"
    if cfg["lr"] != PADRAO["lr"]:
        tag += f"_lr{cfg['lr']:g}"
    if cfg["sem_stride_t"]:
        tag += "_semstride"
    if cfg["saida_media"]:
        tag += "_sm"
    tag += AGENDAMENTOS[cfg["agendamento"]]
    if cfg["validacao_interna"]:
        tag += "_vi"
    if cfg["seed"] != PADRAO["seed"]:
        tag += f"_s{cfg['seed']}"         # replicas nao colidem com a semente 42
    return tag


# -------------------------------------------------------------------- dados
def carregar_dados(caminho_npz):
    d = np.load(caminho_npz, allow_pickle=False)
    req = ["X_blocos", "y", "study", "grupo", "lk_cru", "hibrido"]
    faltam = [k for k in req if k not in d.files]
    if faltam:
        raise RuntimeError(f"npz sem {faltam}: gere de novo com "
                           "src/make_colab_dataset.py")
    return {k: d[k] for k in req}


# Particoes de validacao FIXAS. O sorteio do StratifiedGroupKFold mudou entre
# versoes do scikit-learn (a 1.9.1 local nao reproduz as particoes da Fase 2,
# geradas no Colab), entao as da Fase 3 ficam gravadas aqui: todas as
# maquinas usam exatamente os mesmos pacientes em cada fold. Geradas com
# sortear_folds() no scikit-learn 1.9.1 (seed 42).
PARTICAO_ID = "fase3-sk191"
PARTICOES = {
    0: (
        "drsbru_004 drsbru_007 drsbru_010 drsbru_014 drsbru_018 drsbru_023 "
        "drsbru_035 drsbru_038 drsbru_048 drsbru_065 drsbru_066 drsbru_076 "
        "drsbru_079 drsbru_080 drsbru_083 drsbru_086 drsbru_088 drsbru_093 "
        "drsbru_094 drsbru_099 drsprg_002 drsprg_014 drsprg_016 drsprg_017 "
        "drsprg_029 drsprg_031 drsprg_034 drsprg_056 drsprg_061 drsprg_068 "
        "drsprg_070 drsprg_071 drsprg_076 drsprg_078 drsprg_079 drsprg_090 "
        "drsprg_092 drsprg_095 drsprg_097 drsprg_100 drsprg_105 drsprg_110"
    ),
    1: (
        "drsbru_002 drsbru_003 drsbru_015 drsbru_016 drsbru_022 drsbru_028 "
        "drsbru_031 drsbru_034 drsbru_039 drsbru_052 drsbru_055 drsbru_059 "
        "drsbru_060 drsbru_069 drsbru_081 drsbru_082 drsbru_087 drsbru_091 "
        "drsbru_095 drsbru_096 drsbru_098 drsprg_001 drsprg_007 drsprg_008 "
        "drsprg_024 drsprg_026 drsprg_044 drsprg_053 drsprg_058 drsprg_069 "
        "drsprg_072 drsprg_080 drsprg_082 drsprg_083 drsprg_087 drsprg_093 "
        "drsprg_098 drsprg_099 drsprg_101 drsprg_104 drsprg_108 drsprg_115"
    ),
    2: (
        "drsbru_008 drsbru_011 drsbru_012 drsbru_013 drsbru_017 drsbru_025 "
        "drsbru_027 drsbru_030 drsbru_033 drsbru_045 drsbru_049 drsbru_057 "
        "drsbru_058 drsbru_070 drsbru_071 drsbru_072 drsbru_074 drsbru_077 "
        "drsbru_089 drsbru_090 drsprg_009 drsprg_013 drsprg_015 drsprg_022 "
        "drsprg_023 drsprg_025 drsprg_028 drsprg_032 drsprg_040 drsprg_041 "
        "drsprg_048 drsprg_060 drsprg_062 drsprg_063 drsprg_065 drsprg_084 "
        "drsprg_085 drsprg_089 drsprg_094 drsprg_096 drsprg_106"
    ),
    3: (
        "drsbru_006 drsbru_026 drsbru_036 drsbru_040 drsbru_041 drsbru_043 "
        "drsbru_051 drsbru_053 drsbru_054 drsbru_056 drsbru_061 drsbru_062 "
        "drsbru_063 drsbru_064 drsbru_075 drsbru_078 drsbru_084 drsbru_085 "
        "drsbru_097 drsbru_100 drsprg_004 drsprg_010 drsprg_012 drsprg_019 "
        "drsprg_035 drsprg_038 drsprg_042 drsprg_045 drsprg_046 drsprg_047 "
        "drsprg_049 drsprg_054 drsprg_055 drsprg_057 drsprg_064 drsprg_066 "
        "drsprg_067 drsprg_074 drsprg_086 drsprg_088 drsprg_111 drsprg_114"
    ),
    4: (
        "drsbru_001 drsbru_005 drsbru_009 drsbru_019 drsbru_020 drsbru_021 "
        "drsbru_024 drsbru_029 drsbru_032 drsbru_037 drsbru_042 drsbru_044 "
        "drsbru_046 drsbru_047 drsbru_050 drsbru_067 drsbru_068 drsbru_073 "
        "drsbru_092 drsprg_005 drsprg_006 drsprg_011 drsprg_018 drsprg_020 "
        "drsprg_021 drsprg_027 drsprg_037 drsprg_039 drsprg_050 drsprg_052 "
        "drsprg_059 drsprg_073 drsprg_075 drsprg_077 drsprg_081 drsprg_091 "
        "drsprg_102 drsprg_107 drsprg_112 drsprg_113"
    ),
}


def montar_folds(dados):
    """Particoes fixas por paciente, so entre os reais (iguais em toda maquina)."""
    study, hib = dados["study"], dados["hibrido"]
    reais = np.where(~hib)[0]
    fold_de = {s: f for f, nomes in PARTICOES.items() for s in nomes.split()}
    faltam = [str(s) for s in study[reais] if str(s) not in fold_de]
    if faltam or len(fold_de) != len(reais):
        raise RuntimeError(f"particoes nao batem com o npz (faltam {faltam[:5]})")
    fold_arr = np.array([fold_de[str(s)] for s in study[reais]])
    return [(reais[fold_arr != f], reais[fold_arr == f])
            for f in sorted(PARTICOES)]


def separar_validacao_interna(folds, f):
    """3.7: validacao interna do fold f. O fold seguinte, (f + 1) % k, sai do
    treino e passa a escolher o lr, a parada e a melhor epoca; o fold f fica
    intocado ate a predicao final. Devolve (treino, validacao), com o treino
    na mesma ordem de folds[f][0]."""
    est_tr, est_te = folds[f]
    est_sel = folds[(f + 1) % len(folds)][1]
    if np.intersect1d(est_sel, est_te).size or not np.isin(est_sel, est_tr).all():
        raise RuntimeError(f"validacao interna do fold {f} invalida")
    return est_tr[~np.isin(est_tr, est_sel)], est_sel


def sortear_folds(dados, n_splits=5, seed=42):
    """Sorteio original (dependente da versao do scikit-learn); referencia."""
    y, grupo, study, hib = (dados["y"], dados["grupo"], dados["study"],
                            dados["hibrido"])
    faixas = np.digitize(y, FAIXAS_SPLIT)
    reais = np.where(~hib)[0]
    estrato = np.array([f"{grupo[i]}_{faixas[i]}" for i in reais])
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    return [(reais[tr], reais[va]) for tr, va in
            sgkf.split(np.zeros(len(reais)), estrato, groups=study[reais])]


def agrupar_tempo(XB, n_passos):
    """Blocos de 60 s (n, 20, H, W) -> (n, n_passos, H, W).

    Descarta o 1o minuto (fase vascular) e soma os 19 blocos restantes em
    n_passos quadros consecutivos: 19 -> 1 min cada; 9 -> ~2 min cada
    (a mesma agregacao das celulas do mosaico 3x3).
    """
    blocos = XB[:, 1:20].astype(np.float32)
    if not 1 <= n_passos <= blocos.shape[1]:
        # array_split com mais partes que blocos gera partes vazias, que
        # virariam quadros pretos; quadros mais finos exigem um npz novo
        raise ValueError(f"n_passos={n_passos}: o npz tem {blocos.shape[1]} "
                         "blocos de 60 s apos o 1o minuto")
    partes = np.array_split(np.arange(blocos.shape[1]), n_passos)
    return np.stack([blocos[:, p].sum(axis=1) for p in partes], axis=1)


def preparar_representacao(dados, cfg):
    cfg = {**PADRAO, **cfg}
    if ABORDAGENS[cfg["abordagem"]]["entrada"] == "mosaico":
        return agrupar_tempo(dados["X_blocos"], cfg["grid"][0] * cfg["grid"][1])
    return agrupar_tempo(dados["X_blocos"], cfg["T"])


def _rng_amostra():
    # gerador por amostra derivado do RNG do torch (semeado por worker)
    return np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))


class VideoDataset(Dataset):
    """Tensor (3, T, tam, tam) por estudo e alvo normalizado (y/100).

    Augmentation adaptado ao video: translacao identica em todos os quadros,
    espelhamento horizontal do tensor inteiro com troca do alvo (em 3D nao
    inverte o tempo) e thinning com o MESMO p em todos os quadros (a dose e
    do exame inteiro). Normalizacao log1p + escala GLOBAL ao tensor.
    espelhar=True (so na avaliacao, para a TTA): espelha SEMPRE, com o mesmo
    espelhamento do augmentation, sem consumir o gerador aleatorio.
    """

    def __init__(self, V, y, aug, treino, tam, norma, espelhar=False):
        self.V, self.y, self.aug, self.treino = V, y, aug, treino
        self.espelhar = espelhar
        self.resize = transforms.Resize((tam, tam), antialias=True)
        self.transl = (transforms.RandomAffine(
            degrees=0, translate=(aug["transl"], aug["transl"]), fill=0.0)
            if treino and aug["transl"] > 0 else None)
        media, desvio = NORMAS[norma]
        self.media = torch.tensor(media).view(3, 1, 1, 1)
        self.desvio = torch.tensor(desvio).view(3, 1, 1, 1)

    def __len__(self):
        return len(self.V)

    def __getitem__(self, i):
        v = self.V[i]
        alvo = float(self.y[i])
        if self.treino and self.aug["thinning"] and torch.rand(1).item() < 0.5:
            rng = _rng_amostra()
            p = rng.uniform(0.5, 0.9)
            v = rng.binomial(v.astype(np.int64), p).astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(v))          # (T, H, W)
        if self.transl is not None:
            t = self.transl(t)
        # com espelhar=False a condicao e a mesma das versoes anteriores (o
        # torch.rand so e chamado no treino com flip): sequencia aleatoria igual
        if self.espelhar or (self.treino and self.aug["flip"]
                             and torch.rand(1).item() < 0.5):
            t = torch.flip(t, dims=[2])
            alvo = 100.0 - alvo
        t = torch.log1p(t)
        t = t / (t.max() + 1e-8)
        t = self.resize(t)
        t = t.unsqueeze(0).expand(3, -1, -1, -1)
        t = (t - self.media) / self.desvio
        return t.contiguous(), torch.tensor(alvo / 100.0, dtype=torch.float32)


def tamanho_mosaico(grid, area=AREA_MOSAICO):
    """(altura, largura) da entrada do mosaico com celulas quadradas.

    Lado da celula = area / raiz(g0 * g1): a area da imagem fica igual a do
    mosaico 224 x 224 em qualquer grade. Grades quadradas (1x1, 2x2, 3x3,
    4x4) dao exatamente 224 x 224, como nas versoes anteriores; nas nao
    quadradas (3.6), por exemplo 2x3 -> 183 x 274, as celulas nao sao
    achatadas pelo redimensionamento.
    """
    g0, g1 = int(grid[0]), int(grid[1])
    lado = area / math.sqrt(g0 * g1)
    return int(round(g0 * lado)), int(round(g1 * lado))


class MosaicoDataset(Dataset):
    """Mosaico temporal da Fase 2 (controle), identico ao protocolo de la.

    Celulas em ordem temporal, linha a linha (grade g0 linhas x g1 colunas).
    espelhar=True (so na avaliacao, para a TTA): espelha SEMPRE cada celula,
    como no augmentation, sem consumir o gerador aleatorio.
    """

    def __init__(self, C, y, grid, aug, treino, espelhar=False):
        self.C, self.y, self.grid, self.aug, self.treino = C, y, grid, aug, treino
        self.espelhar = espelhar
        self.resize = transforms.Resize(tamanho_mosaico(grid), antialias=True)
        self.transl = (transforms.RandomAffine(
            degrees=0, translate=(aug["transl"], aug["transl"]), fill=0.0)
            if treino and aug["transl"] > 0 else None)
        self.norm = transforms.Normalize(*NORMAS["imagenet"])

    def __len__(self):
        return len(self.C)

    def __getitem__(self, i):
        cel = self.C[i]
        alvo = float(self.y[i])
        if self.treino and self.aug["thinning"] and torch.rand(1).item() < 0.5:
            rng = _rng_amostra()
            p = rng.uniform(0.5, 0.9)
            cel = rng.binomial(cel.astype(np.int64), p).astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(cel))
        if self.transl is not None:
            t = self.transl(t)
        if self.espelhar or (self.treino and self.aug["flip"]
                             and torch.rand(1).item() < 0.5):
            t = torch.flip(t, dims=[2])          # espelha CADA celula
            alvo = 100.0 - alvo
        t = torch.log1p(t)
        t = t / (t.max() + 1e-8)
        g0, g1 = self.grid
        linhas = [torch.cat(tuple(t[r * g1:(r + 1) * g1]), dim=1)
                  for r in range(g0)]
        img = torch.cat(linhas, dim=0).unsqueeze(0).repeat(3, 1, 1)
        return (self.norm(self.resize(img)),
                torch.tensor(alvo / 100.0, dtype=torch.float32))


def criar_datasets(cfg, rep, y):
    cfg = {**PADRAO, **cfg}
    info = ABORDAGENS[cfg["abordagem"]]
    aug = PRESETS_AUG[cfg["aug"]]
    if info["entrada"] == "mosaico":
        return (MosaicoDataset(rep, y, cfg["grid"], aug, treino=True),
                MosaicoDataset(rep, y, cfg["grid"], aug, treino=False))
    return (VideoDataset(rep, y, aug, True, cfg["tam"], info["norma"]),
            VideoDataset(rep, y, aug, False, cfg["tam"], info["norma"]))


def criar_dataset_espelhado(cfg, rep, y):
    """Avaliacao com o espelho de cada estudo (TTA pela simetria)."""
    cfg = {**PADRAO, **cfg}
    info = ABORDAGENS[cfg["abordagem"]]
    aug = PRESETS_AUG[cfg["aug"]]
    if info["entrada"] == "mosaico":
        return MosaicoDataset(rep, y, cfg["grid"], aug, treino=False,
                              espelhar=True)
    return VideoDataset(rep, y, aug, False, cfg["tam"], info["norma"],
                        espelhar=True)


def descrever_entrada(cfg):
    cfg = {**PADRAO, **cfg}
    if ABORDAGENS[cfg["abordagem"]]["entrada"] == "mosaico":
        return "x".join(str(v) for v in tamanho_mosaico(cfg["grid"]))
    return f"{cfg['T']}x{cfg['tam']}x{cfg['tam']}"


# ------------------------------------------------------------------ modelos
class CNNTemporal(nn.Module):
    """Encoder VGG19 (ImageNet, sem BatchNorm) quadro a quadro + agregacao."""

    def __init__(self, agregador="lstm", oculto=256):
        super().__init__()
        self.encoder = models.vgg19(weights="DEFAULT").features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.agregador = agregador
        if agregador == "lstm":
            self.rnn = nn.LSTM(512, oculto, batch_first=True,
                               bidirectional=True)
            self.cabeca = nn.Linear(2 * oculto, 1)
        else:
            self.cabeca = nn.Linear(512, 1)

    def forward(self, x):                                  # (B, 3, T, H, W)
        b, c, t, h, w = x.shape
        q = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        f = self.pool(self.encoder(q)).flatten(1).view(b, t, -1)
        if self.agregador == "lstm":
            with torch.autocast(device_type=x.device.type, enabled=False):
                saida, _ = self.rnn(f.float())
            z = saida.mean(dim=1)
        else:
            z = f.mean(dim=1)
        return self.cabeca(z)


class CabecaEspacial(nn.Module):
    """Cabeca que preserva o mapa espacial do layer4 (media so no tempo).

    Recebe o tensor ja achatado pelo forward do VideoResNet (B, C*h*w, com o
    avgpool trocado por AdaptiveAvgPool3d((1, h, w))), reduz os canais com
    conv 1x1 e regride com uma Linear sobre o mapa h x w, mantendo a
    disposicao esquerda-direita. A Linear final comeca zerada, e o vies e
    posto na media do alvo (iniciar_vies). Roda em fp32, fora do autocast.
    """

    def __init__(self, canais=512, h=7, w=7, reduzidos=64, p_drop=0.5):
        super().__init__()
        self.canais, self.h, self.w = canais, h, w
        self.reduz = nn.Conv2d(canais, reduzidos, kernel_size=1)
        self.drop = nn.Dropout(p_drop)
        self.saida = nn.Linear(reduzidos * h * w, 1)
        nn.init.zeros_(self.saida.weight)
        nn.init.zeros_(self.saida.bias)

    @torch.no_grad()
    def iniciar_vies(self, valor):
        self.saida.bias.fill_(float(valor))

    def forward(self, x):
        with torch.autocast(device_type=x.device.type, enabled=False):
            z = x.float().reshape(-1, self.canais, self.h, self.w)
            z = torch.relu(self.reduz(z))
            return self.saida(self.drop(z.flatten(1)))


def _remover_stride_temporal(m, nome):
    """R3D-18 sem stride temporal nos layers 2 a 4 (pesos continuam validos)."""
    if nome != "r3d_18":
        raise ValueError("sem_stride_t implementado apenas para o r3d_18")
    for camada in (m.layer2, m.layer3, m.layer4):
        bloco = camada[0]
        for conv in (bloco.conv1[0], bloco.downsample[0]):
            conv.stride = (1,) + tuple(conv.stride[1:])


def _trocar_avgpool3d(modulo):
    for nome, filho in modulo.named_children():
        if isinstance(filho, nn.AvgPool3d):
            setattr(modulo, nome, nn.AdaptiveAvgPool3d(1))
        else:
            _trocar_avgpool3d(filho)


def _zerar_saida(camada):
    """Saida iniciada na media (3.6): pesos zerados; o vies e posto na media
    do alvo de treino em treinar_fold. O construtor da Linear ja consumiu o
    gerador aleatorio como antes, entao a sequencia aleatoria nao muda."""
    with torch.no_grad():
        camada.weight.zero_()
        camada.bias.zero_()


def camada_saida(modelo, nome):
    """Ultima camada linear (cabeca padrao) do mosaico e das ResNets 3D."""
    if nome == "mosaico_vgg19":
        return modelo.classifier[6]
    if nome in ("r3d_18", "mc3_18", "r2plus1d_18"):
        return modelo.fc
    raise ValueError(f"saida_media nao implementada para {nome}")


def construir_modelo(nome, cabeca="padrao", sem_stride_t=False,
                     saida_media=False):
    if saida_media and (nome not in SAIDA_MEDIA_OK or cabeca != "padrao"):
        # a cabeca espacial ja nasce com a saida na media
        raise ValueError(f"saida_media nao se aplica a {nome} com cabeca "
                         f"{cabeca}")
    if nome in ("r3d_18", "mc3_18", "r2plus1d_18"):
        m = getattr(vmodels, nome)(weights="DEFAULT")
        if sem_stride_t:
            _remover_stride_temporal(m, nome)
        if cabeca == "espacial":
            m.avgpool = nn.AdaptiveAvgPool3d((1, 7, 7))
            m.fc = CabecaEspacial(m.fc.in_features, 7, 7)
        elif cabeca == "padrao":
            m.fc = nn.Linear(m.fc.in_features, 1)
            if saida_media:
                _zerar_saida(m.fc)
        else:
            raise ValueError(f"cabeca desconhecida: {cabeca}")
        return m
    if cabeca != "padrao" or sem_stride_t:
        raise ValueError(f"cabeca/sem_stride_t nao implementados para {nome}")
    if nome == "s3d":
        m = vmodels.s3d(weights="DEFAULT")
        m.avgpool = nn.AdaptiveAvgPool3d(1)      # aceita qualquer T e tamanho
        m.classifier[-1] = nn.Conv3d(m.classifier[-1].in_channels, 1,
                                     kernel_size=1)
        # O S3D do torchvision herda do TensorFlow o momentum de BN 0,001 (os
        # demais usam 0,1): com ~7 batches por epoca as estatisticas acumuladas
        # quase nao saem das do Kinetics e a avaliacao desaba. Padronizado.
        for b in m.modules():
            if isinstance(b, nn.modules.batchnorm._BatchNorm):
                b.momentum = 0.1
        return m
    if nome == "x3d_s":
        try:
            from pytorchvideo.models.hub import x3d_s as _x3d_s
        except Exception as e:
            raise RuntimeError("x3d_s requer o pacote pytorchvideo") from e
        m = _x3d_s(pretrained=True)
        cabeca = m.blocks[-1]
        _trocar_avgpool3d(cabeca)
        cabeca.proj = nn.Linear(cabeca.proj.in_features, 1)
        cabeca.activation = None
        return m
    if nome == "vgg19_lstm":
        return CNNTemporal("lstm")
    if nome == "vgg19_media":
        return CNNTemporal("media")
    if nome == "mosaico_vgg19":
        m = models.vgg19(weights="DEFAULT")
        m.classifier[6] = nn.Linear(m.classifier[6].in_features, 1)
        if saida_media:
            _zerar_saida(m.classifier[6])
        return m
    raise ValueError(f"abordagem desconhecida: {nome}")


def tem_bn(modelo):
    return any(isinstance(m, nn.modules.batchnorm._BatchNorm)
               for m in modelo.modules())


def modo_treino(modelo, regime_bn):
    """BN congelado: estatisticas fixas (eval), gamma e beta seguem treinaveis.

    Vale para "congelado" (estatisticas do pre-treino) e "congelado_dominio"
    (estatisticas reestimadas no treino real do fold antes do 1o passo).
    """
    modelo.train()
    if regime_bn.startswith("congelado"):
        for m in modelo.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def _autocast(dispositivo, precisao):
    if precisao == "bf16" and dispositivo.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


@torch.no_grad()
def avaliar(modelo, loader, dispositivo, precisao):
    modelo.eval()
    crit = nn.MSELoss(reduction="sum")
    soma_mse, soma_mae, n, preds = 0.0, 0.0, 0, []
    for x, alvo in loader:
        x = x.to(dispositivo, non_blocking=True)
        alvo = alvo.to(dispositivo, non_blocking=True)
        with _autocast(dispositivo, precisao):
            saida = modelo(x).squeeze(1)
        saida = saida.float()
        soma_mse += crit(saida, alvo).item()
        soma_mae += (saida - alvo).abs().sum().item()
        n += len(alvo)
        preds.append(saida.cpu().numpy())
    return soma_mse / n, soma_mae / n, np.concatenate(preds)


@torch.no_grad()
def recalibrar_bn(modelo, loader_natural, dispositivo, precisao):
    """Reestima as estatisticas de BN no treino em distribuicao natural."""
    bns = [m for m in modelo.modules()
           if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None                      # media cumulativa exata
    modelo.eval()
    for bn in bns:
        bn.train()
    for x, _ in loader_natural:
        with _autocast(dispositivo, precisao):
            modelo(x.to(dispositivo, non_blocking=True))
    modelo.eval()


# -------------------------------------------------------------------- treino
def _lr_aquecimento(lr_base, ep, passo, passos_por_epoca, epocas_aq):
    """Aquecimento linear por passo: chega a lr_base no ultimo passo."""
    return lr_base * (ep * passos_por_epoca + passo + 1) / (
        epocas_aq * passos_por_epoca)


def _lr_cosseno(lr_base, ep, passo, passos_por_epoca, epocas_aq,
                epocas_total):
    """Cosseno por passo depois do aquecimento: lr_base no 1o passo e perto
    de zero no ultimo passo da ultima epoca."""
    total = max(1, (epocas_total - epocas_aq) * passos_por_epoca)
    t = (ep - epocas_aq) * passos_por_epoca + passo
    return lr_base * 0.5 * (1.0 + math.cos(math.pi * t / total))


def combinar_tta(p_bruta, p_esp):
    """TTA pela simetria, com o alvo normalizado (y/100): media de p(x) e de
    1 - p(espelho de x), com o recorte em [0, 100] depois da combinacao."""
    return np.clip((p_bruta + 1.0 - p_esp) / 2.0 * 100.0, 0, 100)


def _checar_micro_lote(cfg, modelo):
    ml = cfg["micro_lote"]
    if ml is None:
        return
    if ml < 1:
        raise ValueError(f"micro_lote invalido: {ml}")
    if ml < cfg["batch"] and cfg["bn"] == "treinavel" and tem_bn(modelo):
        # com BN em modo de treino, cada pedaco teria as proprias estatisticas
        raise ValueError("micro-lotes exigem BN congelado (ou modelo sem BN)")


def _gradientes(modelo, x, alvo, crit, micro_lote, dispositivo, precisao):
    """Forward e backward de um lote (acumula em .grad, sem opt.step).

    Com micro_lote, o lote e dividido em pedacos e a perda de cada pedaco e
    ponderada por len(pedaco)/len(lote): a soma dos gradientes e a do lote
    inteiro (exata quando o BN esta fixo). Devolve (perda media, saidas).
    """
    nb = len(alvo)
    if not micro_lote or micro_lote >= nb:
        with _autocast(dispositivo, precisao):
            saida = modelo(x).squeeze(1)
        saida = saida.float()
        perda = crit(saida, alvo)
        perda.backward()
        return perda.item(), saida.detach()
    perda_tot, partes = 0.0, []
    for i0 in range(0, nb, micro_lote):
        xs, ys = x[i0:i0 + micro_lote], alvo[i0:i0 + micro_lote]
        with _autocast(dispositivo, precisao):
            s = modelo(xs).squeeze(1)
        s = s.float()
        perda = crit(s, ys) * (len(ys) / nb)
        perda.backward()
        perda_tot += perda.item()
        partes.append(s.detach())
    return perda_tot, torch.cat(partes)


def treinar_fold(cfg, rep, y, hibrido, est_tr, est_va, fold, dispositivo,
                 num_workers=0, max_epocas=None, verbose=True, est_sel=None):
    """Treina no treino real (est_tr) mais os hibridos e preve est_va.

    Sem est_sel (ate a 3.6), est_va tambem e a validacao que reduz o lr, para
    o treino e escolhe a melhor epoca. Com est_sel (3.7, validacao interna),
    essas escolhas usam est_sel, e est_va so e previsto no fim, com o
    checkpoint escolhido; o MAE de est_va por epoca fica no historico
    (mse_te, mae_te) so para diagnostico.
    """
    cfg = {**PADRAO, **cfg}
    if cfg["bn"] not in REGIMES_BN:
        raise ValueError(f"regime de BN desconhecido: {cfg['bn']}")
    if cfg["agendamento"] not in AGENDAMENTOS:
        raise ValueError(f"agendamento desconhecido: {cfg['agendamento']}")
    semear(cfg["seed"] + fold)
    ds_tr, ds_av = criar_datasets(cfg, rep, y)

    est_tr_cnn = (np.concatenate([est_tr, np.where(hibrido)[0]])
                  if cfg["usar_drshyb"] else est_tr)
    fx = np.digitize(y[est_tr_cnn], cfg["faixas_sampler"])
    freq = {f: int((fx == f).sum()) for f in np.unique(fx)}
    pesos = torch.as_tensor([(1.0 / freq[f]) ** cfg["alpha_sampler"]
                             for f in fx], dtype=torch.double)
    gerador = torch.Generator().manual_seed(cfg["seed"] + fold)
    sampler = WeightedRandomSampler(pesos, len(pesos), replacement=True,
                                    generator=gerador)
    pin = dispositivo.type == "cuda"
    tr_loader = DataLoader(Subset(ds_tr, est_tr_cnn), batch_size=cfg["batch"],
                           sampler=sampler, num_workers=num_workers,
                           pin_memory=pin)
    va_loader = DataLoader(Subset(ds_av, est_va if est_sel is None else est_sel),
                           batch_size=cfg["batch"],
                           shuffle=False, num_workers=num_workers,
                           pin_memory=pin)
    te_loader = None
    if est_sel is not None:
        # fold avaliado medido por epoca so para diagnostico, com gerador
        # proprio: nao consome o gerador global, que comanda o augmentation
        te_loader = DataLoader(Subset(ds_av, est_va), batch_size=cfg["batch"],
                               shuffle=False, num_workers=num_workers,
                               pin_memory=pin,
                               generator=torch.Generator().manual_seed(
                                   cfg["seed"] + fold))
    aval_loader = va_loader if te_loader is None else te_loader

    modelo = construir_modelo(cfg["abordagem"], cabeca=cfg["cabeca"],
                              sem_stride_t=cfg["sem_stride_t"],
                              saida_media=cfg["saida_media"]).to(dispositivo)
    _checar_micro_lote(cfg, modelo)
    if cfg["cabeca"] == "espacial":
        modelo.fc.iniciar_vies(float(np.mean(y[est_tr_cnn])) / 100.0)
    elif cfg["saida_media"]:
        with torch.no_grad():
            camada_saida(modelo, cfg["abordagem"]).bias.fill_(
                float(np.mean(y[est_tr_cnn])) / 100.0)
    if cfg["bn"] == "congelado_dominio":
        # estatisticas do dominio: treino REAL do fold, sem augmentation, sem
        # reamostragem e sem hibridos, em lotes embaralhados (drop_last evita
        # que o ultimo lote pequeno pese na media cumulativa); depois ficam
        # fixas durante todo o treino (modo_treino as mantem em eval)
        base_bn = DataLoader(Subset(ds_av, est_tr), batch_size=cfg["batch"],
                             shuffle=True, drop_last=True,
                             num_workers=num_workers, pin_memory=pin,
                             generator=torch.Generator().manual_seed(
                                 cfg["seed"] + fold))
        recalibrar_bn(modelo, base_bn, dispositivo, cfg["precisao"])
    opt = optim.Adam(modelo.parameters(), lr=cfg["lr"],
                     weight_decay=cfg["wd"])
    cosseno = cfg["agendamento"] == "cosseno"
    sch = None if cosseno else optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=cfg["paciencia_lr"])
    crit = nn.MSELoss()
    if dispositivo.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dispositivo)

    passos = len(tr_loader)
    aq = int(cfg["aquecimento"] or 0)
    n_epocas = max_epocas or cfg["max_epocas"]
    hist, melhor, melhor_estado, sem = [], float("inf"), None, 0
    t0 = time.time()
    for ep in range(n_epocas):
        te = time.time()
        modo_treino(modelo, cfg["bn"])
        soma_mse, soma_mae, n = 0.0, 0.0, 0
        for i, (x, alvo) in enumerate(tr_loader):
            if ep < aq:
                for g in opt.param_groups:
                    g["lr"] = _lr_aquecimento(cfg["lr"], ep, i, passos, aq)
            elif cosseno:
                for g in opt.param_groups:
                    g["lr"] = _lr_cosseno(cfg["lr"], ep, i, passos, aq,
                                          n_epocas)
            x = x.to(dispositivo, non_blocking=True)
            alvo = alvo.to(dispositivo, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            perda_val, saida = _gradientes(modelo, x, alvo, crit,
                                           cfg["micro_lote"], dispositivo,
                                           cfg["precisao"])
            opt.step()
            soma_mse += perda_val * len(alvo)
            soma_mae += (saida - alvo).abs().sum().item()
            n += len(alvo)
        mse_va, mae_va, _ = avaliar(modelo, va_loader, dispositivo,
                                    cfg["precisao"])
        diag = {}
        if te_loader is not None:
            mse_te, mae_te, _ = avaliar(modelo, te_loader, dispositivo,
                                        cfg["precisao"])
            diag = dict(mse_te=mse_te, mae_te=mae_te * 100)
        if sch is not None and ep >= aq:   # o plato so conta depois do
            sch.step(mse_va)               # aquecimento
        hist.append(dict(epoca=ep + 1, mse_tr=soma_mse / n,
                         mae_tr=soma_mae / n * 100, mse_va=mse_va,
                         mae_va=mae_va * 100, lr=opt.param_groups[0]["lr"],
                         segundos=time.time() - te, **diag))
        lim = cfg["limite_seg_epoca"]
        if lim and len(hist) == 4:
            med = float(np.median([h["segundos"] for h in hist[1:4]]))
            if med > lim:
                raise RuntimeError(f"{med:.0f} s por epoca (limite {lim:g} s): "
                                   "provavel transbordo da VRAM")
        if mse_va < melhor:
            melhor, sem = mse_va, 0
            melhor_estado = {k: v.detach().to("cpu", copy=True)
                             for k, v in modelo.state_dict().items()}
        else:
            sem += 1
        if verbose:
            print(f"  fold {fold} | ep {ep + 1:3d} | MAE treino "
                  f"{soma_mae / n * 100:5.2f} | MAE val {mae_va * 100:5.2f} | "
                  f"{time.time() - te:4.0f}s")
        if not cosseno and sem >= cfg["paciencia_es"]:
            break                    # no cosseno o orcamento de epocas e fixo

    modelo.load_state_dict(melhor_estado)
    _, _, p_bruta = avaliar(modelo, aval_loader, dispositivo, cfg["precisao"])
    preds = np.clip(p_bruta * 100.0, 0, 100)

    # TTA pela simetria (3.6): no mesmo checkpoint, preve o espelho de cada
    # estudo (alvo trocado por 100 - y) e combina p com 1 - p' antes do
    # recorte. Nao muda o treino nem a escolha da melhor epoca.
    esp_loader = DataLoader(Subset(criar_dataset_espelhado(cfg, rep, y),
                                   est_va),
                            batch_size=cfg["batch"], shuffle=False,
                            num_workers=num_workers, pin_memory=pin)
    _, _, p_esp = avaliar(modelo, esp_loader, dispositivo, cfg["precisao"])
    preds_tta = combinar_tta(p_bruta, p_esp)

    preds_recal = None
    if tem_bn(modelo):
        # embaralhado: no npz os estudos estao agrupados (pediatricos antes dos
        # adultos), e batches homogeneos subestimam a variancia da populacao
        natural = DataLoader(Subset(ds_av, est_tr), batch_size=cfg["batch"],
                             shuffle=True, num_workers=num_workers,
                             pin_memory=pin,
                             generator=torch.Generator().manual_seed(
                                 cfg["seed"] + fold))
        copia = copy.deepcopy(modelo)
        recalibrar_bn(copia, natural, dispositivo, cfg["precisao"])
        _, _, p2 = avaliar(copia, aval_loader, dispositivo, cfg["precisao"])
        preds_recal = np.clip(p2 * 100.0, 0, 100)
        del copia

    pico = (torch.cuda.max_memory_allocated(dispositivo) / 1e9
            if dispositivo.type == "cuda" else float("nan"))
    pico_res = (torch.cuda.max_memory_reserved(dispositivo) / 1e9
                if dispositivo.type == "cuda" else float("nan"))
    dfh = pd.DataFrame(hist)
    return dict(preds=preds, preds_recal=preds_recal, preds_tta=preds_tta,
                hist=dfh,
                melhor_epoca=int(dfh.mse_va.idxmin()) + 1, epocas=len(dfh),
                minutos=(time.time() - t0) / 60, pico_mem_gb=pico,
                pico_res_gb=pico_res)


# ------------------------------------------------------------------ metricas
def metricas(pred, real):
    err = pred - real
    r = float(np.corrcoef(pred, real)[0, 1]) if np.std(pred) > 0 else np.nan
    return dict(mae=float(np.abs(err).mean()),
                rmse=float(np.sqrt((err ** 2).mean())), r=r,
                r2=float(1 - (err ** 2).sum()
                         / ((real - real.mean()) ** 2).sum()))


def referencias_pdi(lk_cru, y, est_tr, est_va):
    crua = metricas(lk_cru[est_va], y[est_va])
    a, b = np.polyfit(lk_cru[est_tr], y[est_tr], 1)
    recal = metricas(np.clip(a * lk_cru[est_va] + b, 0, 100), y[est_va])
    return crua, recal


# ------------------------------------------------------------------- runner
def _anexar(arq, linha, chave=None):
    novo = pd.DataFrame([linha])
    if os.path.exists(arq):
        antigo = pd.read_csv(arq)
        if chave:
            mascara = np.ones(len(antigo), bool)
            for k in chave:
                mascara &= (antigo[k].astype(str) == str(linha[k])).to_numpy()
            antigo = antigo[~mascara]
        novo = pd.concat([antigo, novo], ignore_index=True)
    novo.to_csv(arq, index=False)


def descrever_hardware():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "cpu"


def arquivos_saida(pasta, nome_base="fase3"):
    """(CSV por fold, CSV de resumo, pasta de historicos) de uma fila."""
    return (os.path.join(pasta, f"folds_{nome_base}.csv"),
            os.path.join(pasta, f"resultados_{nome_base}.csv"),
            os.path.join(pasta, f"historicos_{nome_base}"))


def rodar_configuracao(cfg, dados, folds, folds_usar, pasta, dispositivo,
                       num_workers=0, verbose=True, nome_base="fase3"):
    """Treina/avalia uma configuracao nas particoes pedidas, com retomada.

    Grava por fold (folds_<nome_base>.csv, historicos e predicoes por estudo)
    e, quando todas as particoes pedidas existem, o resumo
    (resultados_<nome_base>.csv). nome_base="fase3" mantem os nomes antigos.
    """
    cfg = {**PADRAO, **cfg}
    tag = tag_config(cfg)
    info = ABORDAGENS[cfg["abordagem"]]
    os.makedirs(pasta, exist_ok=True)
    arq_f, arq_r, dir_h = arquivos_saida(pasta, nome_base)
    os.makedirs(dir_h, exist_ok=True)
    hardware = descrever_hardware()

    feitos = set()
    if os.path.exists(arq_f):
        df = pd.read_csv(arq_f)
        if "particao" in df.columns:          # linhas de outra particao nao contam
            df = df[df.particao == PARTICAO_ID]
            feitos = set(df.loc[df.tag == tag, "fold"].astype(int))

    rep = None
    for f in folds_usar:
        if f in feitos:
            print(f"-- {tag} fold {f}: ja registrado, pulando")
            continue
        if rep is None:
            rep = preparar_representacao(dados, cfg)
        print(f"\n===== {tag} | fold {f} | {hardware} =====")
        est_tr, est_va = folds[f]
        est_tr_rede, est_sel = (separar_validacao_interna(folds, f)
                                if cfg["validacao_interna"] else (est_tr, None))
        r = treinar_fold(cfg, rep, dados["y"], dados["hibrido"], est_tr_rede,
                         est_va, f, dispositivo, num_workers=num_workers,
                         verbose=verbose, est_sel=est_sel)
        real = dados["y"][est_va]
        m = metricas(r["preds"], real)
        mt = metricas(r["preds_tta"], real)
        mr = (metricas(r["preds_recal"], real)
              if r["preds_recal"] is not None else None)
        crua, recal = referencias_pdi(dados["lk_cru"], dados["y"], est_tr,
                                      est_va)
        linha = dict(
            quando=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            tag=tag, abordagem=cfg["abordagem"],
            T=cfg["T"] if info["entrada"] == "video" else np.nan,
            grid=(f"{cfg['grid'][0]}x{cfg['grid'][1]}"
                  if info["entrada"] == "mosaico" else ""),
            aug=cfg["aug"], bn=cfg["bn"], semente=cfg["seed"], fold=f,
            mae=round(m["mae"], 3), rmse=round(m["rmse"], 3),
            r=round(m["r"], 4), r2=round(m["r2"], 4),
            mae_bn_recal=round(mr["mae"], 3) if mr else np.nan,
            rmse_bn_recal=round(mr["rmse"], 3) if mr else np.nan,
            mae_pdi_recal=round(recal["mae"], 3),
            mae_pdi_crua=round(crua["mae"], 3),
            melhor_epoca=r["melhor_epoca"], epocas=r["epocas"],
            minutos=round(r["minutos"], 2),
            pico_mem_gb=round(r["pico_mem_gb"], 2),
            pico_res_gb=round(r["pico_res_gb"], 2),
            hardware=hardware, precisao=cfg["precisao"], batch=cfg["batch"],
            lr=cfg["lr"], cabeca=cfg["cabeca"], aquecimento=cfg["aquecimento"],
            micro_lote=cfg["micro_lote"] or 0,
            sem_stride_t=bool(cfg["sem_stride_t"]),
            particao=PARTICAO_ID, versao=VERSAO,
            # 3.6: TTA pela simetria, saida na media, agendamento e entrada
            mae_tta=round(mt["mae"], 3), rmse_tta=round(mt["rmse"], 3),
            r_tta=round(mt["r"], 4), r2_tta=round(mt["r2"], 4),
            saida_media=bool(cfg["saida_media"]),
            agendamento=cfg["agendamento"], entrada=descrever_entrada(cfg),
            # 3.7: fold que escolheu lr, parada e melhor epoca e tamanhos
            validacao_interna=bool(cfg["validacao_interna"]),
            fold_validacao=(int((f + 1) % len(folds))
                            if cfg["validacao_interna"] else int(f)),
            n_treino_real=int(len(est_tr_rede)),
            n_validacao=int(len(est_va if est_sel is None else est_sel)))
        _anexar(arq_f, linha)
        r["hist"].to_csv(os.path.join(dir_h, f"hist_{tag}_fold{f}.csv"),
                         index=False)
        pd.DataFrame(dict(
            study=dados["study"][est_va], y=real, pred=r["preds"],
            pred_bn_recal=(r["preds_recal"] if r["preds_recal"] is not None
                           else np.nan),
            pred_tta=r["preds_tta"])).to_csv(
            os.path.join(dir_h, f"pred_{tag}_fold{f}.csv"), index=False)
        print(f">>> {tag} fold {f}: MAE {m['mae']:.2f} | TTA {mt['mae']:.2f} "
              f"| RMSE {m['rmse']:.2f} | r {m['r']:.3f} | PDI recal "
              f"{recal['mae']:.2f} | {r['minutos']:.1f} min | pico "
              f"{r['pico_mem_gb']:.1f} GB")
        if dispositivo.type == "cuda":
            torch.cuda.empty_cache()

    df = pd.read_csv(arq_f)
    d = df[(df.tag == tag) & (df.fold.isin(folds_usar))
           & (df.get("particao", pd.Series(index=df.index, dtype=str))
              == PARTICAO_ID)]
    if len(d) < len(folds_usar):
        return None
    resumo = dict(
        quando=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        tag=tag, abordagem=cfg["abordagem"], T=d["T"].iloc[0],
        grid=d["grid"].iloc[0], aug=cfg["aug"], bn=cfg["bn"],
        semente=cfg["seed"], n_folds=len(folds_usar),
        folds=",".join(str(f) for f in sorted(folds_usar)),
        mae=round(d.mae.mean(), 2), dp_mae=round(d.mae.std(ddof=0), 2),
        rmse=round(d.rmse.mean(), 2), r=round(d.r.mean(), 3),
        r2=round(d.r2.mean(), 3),
        mae_bn_recal=round(d.mae_bn_recal.mean(), 2),
        mae_pdi_recal=round(d.mae_pdi_recal.mean(), 2),
        minutos_por_fold=round(d.minutos.mean(), 1),
        pico_mem_gb=round(d.pico_mem_gb.max(), 2),
        hardware=";".join(sorted(set(d.hardware.astype(str)))),
        lr=cfg["lr"], cabeca=cfg["cabeca"], aquecimento=cfg["aquecimento"],
        micro_lote=cfg["micro_lote"] or 0,
        sem_stride_t=bool(cfg["sem_stride_t"]),
        particao=PARTICAO_ID, versao=VERSAO,
        mae_tta=(round(d.mae_tta.mean(), 2) if "mae_tta" in d.columns
                 else np.nan),
        saida_media=bool(cfg["saida_media"]),
        agendamento=cfg["agendamento"], entrada=descrever_entrada(cfg),
        validacao_interna=bool(cfg["validacao_interna"]))
    _anexar(arq_r, resumo, chave=("tag", "n_folds"))
    return resumo


def teste_rapido(abordagens, dados, folds, dispositivo, T=19, epocas=2,
                 num_workers=0, extra=None):
    """Treina poucas epocas no fold 0 para medir tempo e memoria (sem gravar)."""
    linhas = []
    est_tr, est_va = folds[0]
    for nome in abordagens:
        cfg = {**PADRAO, "abordagem": nome, "T": T, **(extra or {})}
        try:
            rep = preparar_representacao(dados, cfg)
            est_tr_rede, est_sel = (separar_validacao_interna(folds, 0)
                                    if cfg["validacao_interna"]
                                    else (est_tr, None))
            r = treinar_fold(cfg, rep, dados["y"], dados["hibrido"],
                             est_tr_rede, est_va, 0, dispositivo,
                             num_workers=num_workers, max_epocas=epocas,
                             verbose=False, est_sel=est_sel)
            real = dados["y"][est_va]
            linhas.append(dict(abordagem=nome, tag=tag_config(cfg), ok=True,
                               seg_por_epoca=round(r["hist"].segundos.iloc[-1], 1),
                               pico_mem_gb=round(r["pico_mem_gb"], 2),
                               pico_res_gb=round(r["pico_res_gb"], 2),
                               mae_val_ep1=round(r["hist"].mae_va.iloc[0], 2),
                               mae_val_ultima=round(r["hist"].mae_va.iloc[-1], 2),
                               mae_final=round(metricas(r["preds"], real)["mae"], 2),
                               mae_tta=round(metricas(r["preds_tta"], real)["mae"], 2),
                               entrada=descrever_entrada(cfg),
                               erro=""))
        except Exception as e:
            linhas.append(dict(abordagem=nome, tag=tag_config(cfg), ok=False,
                               seg_por_epoca=np.nan, pico_mem_gb=np.nan,
                               pico_res_gb=np.nan, mae_val_ep1=np.nan,
                               mae_val_ultima=np.nan, mae_final=np.nan,
                               mae_tta=np.nan, entrada="",
                               erro=f"{type(e).__name__}: {str(e)[:160]}"))
        if dispositivo.type == "cuda":
            torch.cuda.empty_cache()
        print(linhas[-1])
    return pd.DataFrame(linhas)
