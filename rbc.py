from __future__ import annotations
import os
import re
import json
import math
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import MinMaxScaler


@dataclass
class RBCConfig:
    numeric_weight: float = 0.5  # peso da parte numérica na similaridade final
    categorical_weight: float = 0.5  # peso da parte categórica
    k_neighbors: int = 10
    max_rows: int = 250_000  # amostra máx. para performance
    drop_col_threshold: float = 0.98  # dropa colunas “quase constantes”
    max_na_frac: float = 0.8  # dropa colunas com NA acima desse percentual


def find_main_csv(path: str) -> str:
    """
    Varre a pasta baixada do Kaggle e seleciona:
    1) um arquivo 'main' conhecido, se houver,
    2) ou o maior CSV (por tamanho) como fallback.
    """
    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    if not csvs:
        raise FileNotFoundError("Nenhum CSV encontrado na pasta do dataset.")

    # Heurística: preferir nomes típicos
    preferred = [c for c in csvs if re.search(r"(full|main|global|covid|cases|time|country)", os.path.basename(c), re.I)]
    if preferred:
        # dentre os preferidos, pegue o maior por tamanho
        preferred.sort(key=lambda p: os.path.getsize(p), reverse=True)
        return preferred[0]

    # fallback: pegar o maior
    csvs.sort(key=lambda p: os.path.getsize(p), reverse=True)
    return csvs[0]


def load_dataset(csv_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    # tenta um load direto; se falhar por separador, tenta auto detect
    try:
        df = pd.read_csv(csv_path, low_memory=False)
    except Exception:
        df = pd.read_csv(csv_path, low_memory=False, sep=None, engine="python")
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


def basic_clean(df: pd.DataFrame, cfg: RBCConfig) -> pd.DataFrame:
    # remove colunas 100% nulas
    df = df.dropna(axis=1, how="all")

    # remove colunas com muita ausência
    na_frac = df.isna().mean()
    df = df.loc[:, na_frac <= cfg.max_na_frac].copy()

    # dropa colunas quase constantes
    nunique = df.nunique(dropna=True)
    keep = nunique[nunique > 1].index
    const_cols = [c for c in df.columns if c not in keep]
    df = df[keep].copy()

    # remove colunas obviamente identificadoras (heurística)
    id_like = [c for c in df.columns if re.search(r"(id|uid|uuid|hash)", c, re.I)]
    df = df.drop(columns=list(set(id_like)), errors="ignore")

    return df


def split_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    # tipagem simples: numérico vs categórico (object / bool / category)
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    return numeric_cols, cat_cols


def impute_and_scale(
    df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, str], MinMaxScaler]:
    df = df.copy()

    # imputação numérica: mediana
    medians: Dict[str, float] = {}
    for c in numeric_cols:
        med = df[c].median(skipna=True)
        medians[c] = float(med) if not math.isnan(med) else 0.0
        df[c] = df[c].fillna(medians[c])

    # imputação categórica: moda
    modes: Dict[str, str] = {}
    for c in cat_cols:
        if df[c].dropna().empty:
            modes[c] = ""
            df[c] = df[c].fillna("")
        else:
            mode_val = df[c].mode(dropna=True).iloc[0]
            modes[c] = str(mode_val)
            df[c] = df[c].astype(str).fillna(modes[c])

    # escala min-max nos numéricos
    scaler = MinMaxScaler()
    if numeric_cols:
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df, medians, modes, scaler


def jaccard_similarity(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    uni = len(a | b)
    return inter / uni if uni else 1.0


def compute_similarity_row(
    case_row: pd.Series,
    df: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    num_weight: float,
    cat_weight: float,
    treat_categorical_as_sets: bool = False
) -> np.ndarray:
    # NUMÉRICO: Manhattan normalizada (colunas já em [0,1])
    num_sim = 0.0
    if numeric_cols:
        # distância média por linha
        diffs = np.abs(df[numeric_cols].values - case_row[numeric_cols].values)
        # similaridade = 1 - distância média
        num_sim = 1.0 - diffs.mean(axis=1)

    # CATEGÓRICO:
    #  - padrão: match exato (1/0) por coluna e média
    #  - opcional: Jaccard se "parecer" lista (ex: "A;B;C")
    cat_sim = 0.0
    if cat_cols:
        if treat_categorical_as_sets:
            def to_set(x):
                if isinstance(x, str) and ";" in x:
                    return set(p.strip() for p in x.split(";") if p.strip())
                # caso não tenha separador, vira singleton
                return {str(x)}
            case_sets = [to_set(case_row[c]) for c in cat_cols]
            df_sets = [[to_set(v) for v in df[c].values] for c in cat_cols]

            sims = []
            for i in range(len(df)):
                per_col = []
                for j, c in enumerate(cat_cols):
                    per_col.append(jaccard_similarity(case_sets[j], df_sets[j][i]))
                sims.append(np.mean(per_col))
            cat_sim = np.array(sims)
        else:
            matches = []
            for c in cat_cols:
                matches.append((df[c].values == case_row[c]).astype(float))
            cat_sim = np.mean(np.vstack(matches), axis=0)

    # COMBINAÇÃO HÍBRIDA
    if len(numeric_cols) == 0:
        final = cat_sim
    elif len(cat_cols) == 0:
        final = num_sim
    else:
        final = num_weight * num_sim + cat_weight * cat_sim

    # garante faixa [0,1]
    return np.clip(final, 0.0, 1.0)


def top_k_similar(
    new_case: pd.Series,
    df_proc: pd.DataFrame,
    numeric_cols: List[str],
    cat_cols: List[str],
    cfg: RBCConfig,
    treat_categorical_as_sets: bool = False
) -> pd.DataFrame:
    sims = compute_similarity_row(
        new_case, df_proc, numeric_cols, cat_cols,
        cfg.numeric_weight, cfg.categorical_weight,
        treat_categorical_as_sets=treat_categorical_as_sets
    )
    out = df_proc.copy()
    out["__similarity__"] = sims
    out = out.sort_values("__similarity__", ascending=False).head(cfg.k_neighbors)
    return out
