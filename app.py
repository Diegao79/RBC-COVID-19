import os
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import kagglehub

from rbc import (
    RBCConfig, find_main_csv, load_dataset, basic_clean,
    split_types, impute_and_scale, top_k_similar
)

# =========================
# Configuração da página
# =========================
st.set_page_config(page_title="RBC – COVID-19 (2025)", layout="wide")

# =========================
# Parâmetros fixos (sem sidebar)
# =========================
K_NEIGHBORS = 10
NUM_WEIGHT = 0.5          # peso dos atributos numéricos
CAT_WEIGHT = 1.0 - NUM_WEIGHT
MAX_ROWS   = 250_000
TREAT_SETS = False        # geralmente não há listas no mesmo campo nessa base

cfg = RBCConfig(
    k_neighbors=K_NEIGHBORS,
    numeric_weight=NUM_WEIGHT,
    categorical_weight=CAT_WEIGHT,
    max_rows=MAX_ROWS
)

# =========================
# Tradução & “humanização” (somente exibição)
# =========================
OVERRIDES_PT = {
    # Identidade/locais/tempo
    "location": "País/Local",
    "country": "País",
    "state": "Estado",
    "province": "Província/Estado",
    "region": "Região",
    "continent": "Continente",
    "iso_code": "Código ISO",
    "date": "Data",
    "population": "População",

    # Casos e óbitos
    "total_cases": "Casos Totais",
    "new_cases": "Novos Casos (dia)",
    "new_cases_smoothed": "Novos Casos (média móvel)",
    "total_deaths": "Óbitos Totais",
    "new_deaths": "Novos Óbitos (dia)",
    "new_deaths_smoothed": "Novos Óbitos (média móvel)",
    "case_fatality_rate": "Letalidade (%)",

    # Testes
    "total_tests": "Testes Totais",
    "new_tests": "Novos Testes (dia)",
    "positive_rate": "Taxa de Positividade",
    "tests_units": "Unidade de Testes",
    "tests_per_case": "Testes por Caso",
    "tests_per_1m": "Testes por 1M hab.",
    "tests_per_million": "Testes por 1M hab.",
    "tests_units_code": "Cód. Unidade de Testes",

    # Internações
    "icu_patients": "Pacientes em UTI",
    "hosp_patients": "Pacientes Hospitalizados",
    "weekly_icu_admissions": "Internações em UTI (semana)",
    "weekly_hosp_admissions": "Internações Hospitalares (semana)",

    # Vacinas
    "total_vaccinations": "Doses Totais Aplicadas",
    "people_vaccinated": "Pessoas com 1ª dose",
    "people_fully_vaccinated": "Pessoas com esquema completo",
    "new_vaccinations": "Novas Vacinações (dia)",
    "total_boosters": "Doses de Reforço Totais",
    "people_vaccinated_per_hundred": "1ª dose (% população)",
    "people_fully_vaccinated_per_hundred": "Esquema completo (% população)",
    "total_vaccinations_per_hundred": "Doses Totais (% população)",

    # Indicadores
    "reproduction_rate": "Taxa de Reprodução (Rt)",
    "stringency_index": "Índice de Rigor (NPIs)",
    "excess_mortality": "Excesso de Mortalidade (%)",
    "life_expectancy": "Expectativa de Vida (anos)",
    "median_age": "Idade Mediana",
    "gdp_per_capita": "PIB per capita (US$)",
    "human_development_index": "IDH",
}

def _tidy_token(tok: str) -> str:
    tok = tok.replace("_per_million", " por 1M hab.")
    tok = tok.replace("_per_hundred", " (% população)")
    tok = tok.replace("_smoothed", " (média móvel)")
    tok = tok.replace("_", " ")
    return tok

def auto_pt(col: str) -> str:
    base = _tidy_token(col)
    base = " ".join(w.capitalize() for w in base.split())
    return base

def traduz(col: str) -> str:
    return OVERRIDES_PT.get(col, auto_pt(col))

def traduz_cols(cols):
    return {c: traduz(c) for c in cols}

# =========================
# Título / fluxo
# =========================
st.title("🧠 RBC – COVID-19 (2025)")
st.markdown("""
**Como usar (3 passos):**
1. **A base é carregada automaticamente** e mostramos um resumo.
2. **Escolha os atributos** e **preencha o novo caso** (um país/dia hipotético).
3. Clique em **Encontrar casos semelhantes** para ver o **Top 10**.

> Para comparar **nível da pandemia**, inclua **Casos/Óbitos/Internações/Testes** (numéricos).  
> Para comparar **perfil geográfico**, inclua **Continente/País/Região** (categóricos).
""")

# =========================
# 1) Download e carga do dataset
# =========================
with st.spinner("📥 Baixando a base do Kaggle (primeira vez pode demorar)..."):
    try:
        dataset_path = kagglehub.dataset_download("palakjain9/latest-covid-19-data-2025")
    except Exception as e:
        st.error(f"Falha ao baixar a base via kagglehub: {e}")
        st.stop()
st.success(f"Dataset baixado em: {dataset_path}")

# Seleciona CSV principal
try:
    csv_path = find_main_csv(dataset_path)
    st.caption(f"Arquivo de dados identificado: **{os.path.basename(csv_path)}**")
except Exception as e:
    st.error(str(e))
    st.stop()

# Carrega e pré-processa
with st.spinner("🧼 Carregando e preparando os dados..."):
    raw = load_dataset(csv_path, max_rows=MAX_ROWS)
    st.markdown(f"**Linhas:** {len(raw):,} &nbsp;&nbsp; **Colunas:** {len(raw.columns)}")

    df = basic_clean(raw, cfg)
    numeric_cols, cat_cols = split_types(df)
    df_proc, medians, modes, scaler = impute_and_scale(df, numeric_cols, cat_cols)

# =========================
# 2) Atributos & Novo Caso
# =========================
st.markdown("### 🧩 Passo 2 — Selecione os atributos e preencha o novo caso")

# Sugestões úteis (ajuste automático conforme existem no CSV)
suggest_num = [c for c in [
    "total_cases", "new_cases", "new_cases_smoothed",
    "total_deaths", "new_deaths", "new_deaths_smoothed",
    "total_tests", "new_tests", "positive_rate",
    "icu_patients", "hosp_patients",
    "total_vaccinations", "people_vaccinated", "people_fully_vaccinated",
    "new_vaccinations",
    "reproduction_rate", "stringency_index",
] if c in numeric_cols]

suggest_cat = [c for c in [
    "date", "location", "continent", "region", "country", "state", "province",
] if c in cat_cols]

colA, colB = st.columns(2)

with colA:
    selected_numeric = st.multiselect(
        "Atributos **numéricos** (dinâmica/nível da pandemia)",
        options=numeric_cols,
        default=suggest_num[: min(8, len(suggest_num))] or numeric_cols[: min(8, len(numeric_cols))],
        format_func=traduz
    )

with colB:
    selected_categorical = st.multiselect(
        "Atributos **categóricos** (perfil geográfico/temporal)",
        options=cat_cols,
        default=suggest_cat[: min(6, len(suggest_cat))] or cat_cols[: min(6, len(cat_cols))],
        format_func=traduz
    )

if not selected_numeric and not selected_categorical:
    st.warning("Selecione pelo menos um atributo numérico ou categórico.")
    st.stop()

st.divider()
st.markdown("#### 🧪 Preencha o **novo caso**")

def mk_numeric_widget(col):
    col_min = float(df[col].min(skipna=True)) if col in df else 0.0
    col_max = float(df[col].max(skipna=True)) if col in df else 1.0
    default = float(df[col].median()) if col in df else 0.0
    step = (col_max - col_min) / 100 or 0.1
    label = f"{traduz(col)}  (intervalo: {col_min:.2f} – {col_max:.2f})"
    if not np.isfinite(col_min) or not np.isfinite(col_max) or col_min == col_max:
        label = traduz(col)
    return st.number_input(label, value=default, step=step)

def mk_cat_widget(col):
    uniq = df[col].dropna().astype(str).unique().tolist()[:500]
    default = modes.get(col, uniq[0] if uniq else "")
    return st.selectbox(traduz(col), options=uniq if uniq else [""],
                        index=uniq.index(default) if default in uniq else 0)

left, right = st.columns(2)
new_numeric, new_categorical = {}, {}

with left:
    st.markdown("**Atributos numéricos**")
    for c in selected_numeric:
        new_numeric[c] = mk_numeric_widget(c)

with right:
    st.markdown("**Atributos categóricos**")
    for c in selected_categorical:
        new_categorical[c] = mk_cat_widget(c)

# Monta Series do novo caso
new_case_full = {c: np.nan for c in df.columns}
for c, v in new_numeric.items():
    new_case_full[c] = v
for c, v in new_categorical.items():
    new_case_full[c] = v
new_case = pd.Series(new_case_full)

# Imputação consistente
for c in selected_numeric:
    if pd.isna(new_case[c]):
        new_case[c] = medians.get(c, 0.0)
for c in selected_categorical:
    if pd.isna(new_case[c]):
        new_case[c] = modes.get(c, "")
    new_case[c] = str(new_case[c])

# Escala consistente (coluna a coluna) — evita erro de feature names
def scale_value(col_name: str, raw_value: float) -> float:
    if col_name not in numeric_cols:
        return float(raw_value)
    idx = numeric_cols.index(col_name)
    sc = float(scaler.scale_[idx])
    mn = float(scaler.min_[idx])
    return float(raw_value) * sc + mn

for col in selected_numeric:
    new_case[col] = scale_value(col, float(new_case[col]))

# =========================
# 3) Execução do RBC
# =========================
st.divider()
cols_for_similarity = list(dict.fromkeys(selected_numeric + selected_categorical))
df_sim = df_proc[cols_for_similarity].copy()
new_case_sim = new_case[cols_for_similarity]

if st.button("🔍 Encontrar casos semelhantes"):
    with st.spinner("Calculando similaridade..."):
        if (cfg.numeric_weight + cfg.categorical_weight) == 0:
            cfg.numeric_weight, cfg.categorical_weight = 0.5, 0.5

        out = top_k_similar(
            new_case_sim, df_sim,
            selected_numeric, selected_categorical,
            cfg, treat_categorical_as_sets=TREAT_SETS
        )

    st.success(f"Top {cfg.k_neighbors} casos mais semelhantes")

    # --------------------------
    # Toggle para ver normalizado
    # --------------------------
    mostrar_normalizado = st.checkbox("Mostrar valores normalizados (0–1)", value=False)

    # --------------------------
    # Exibição principal (valores reais por padrão)
    # --------------------------
    prefer_order = [c for c in [
        "date", "continent", "location", "region", "country", "state", "province",
        "total_cases", "new_cases", "new_cases_smoothed",
        "total_deaths", "new_deaths", "new_deaths_smoothed",
        "total_tests", "new_tests", "positive_rate",
        "icu_patients", "hosp_patients",
        "total_vaccinations", "people_vaccinated", "people_fully_vaccinated",
        "new_vaccinations",
        "reproduction_rate", "stringency_index",
        "population",
    ] if c in out.columns or c in df.columns]

    cols_base = list(dict.fromkeys(prefer_order + cols_for_similarity))

    # ---- Funções de formatação PT-BR
    def _fmt_generic(x):
        if pd.isna(x):
            return ""
        # inteiro (ou float com .0)
        if isinstance(x, (int, np.integer)) or (isinstance(x, (float, np.floating)) and float(x).is_integer()):
            s = f"{int(x):,}"
            return s.replace(",", ".")
        # decimal com 2 casas
        s = f"{float(x):,.2f}"
        return s.replace(",", "X").replace(".", ",").replace("X", ".")

    def fmt_percent(x):
        if pd.isna(x):
            return ""
        return f"{x*100:.1f}%".replace(".", ",")

    # conjunto de nomes traduzidos das colunas numéricas, para sabermos o que formatar
    numeric_translated = {traduz(c) for c in numeric_cols}

    if mostrar_normalizado:
        # Mostrar diretamente a visão processada (0–1)
        base_norm = df_proc.loc[out.index, [c for c in cols_base if c in df_proc.columns]].copy()
        base_norm["__similarity__"] = out["__similarity__"].values
        rename_map = traduz_cols(base_norm.columns)
        rename_map["__similarity__"] = "Similaridade"
        shown = base_norm.rename(columns=rename_map)
        # Similaridade em %
        shown["Similaridade"] = shown["Similaridade"].map(fmt_percent)
        st.subheader("📋 Top 10 (normalizado 0–1)")
        st.dataframe(shown.reset_index(drop=True), use_container_width=True)
    else:
        # Mostrar os valores reais do df original
        base_view = df.loc[out.index, [c for c in cols_base if c in df.columns]].copy()
        base_view["__similarity__"] = out["__similarity__"].values

        rename_map = traduz_cols(base_view.columns)
        rename_map["__similarity__"] = "Similaridade"
        base_view = base_view.rename(columns=rename_map)

        # Formata
        formatted = base_view.copy()
        for col in formatted.columns:
            if col == "Similaridade":
                formatted[col] = formatted[col].map(fmt_percent)
            elif col in numeric_translated:
                formatted[col] = formatted[col].map(_fmt_generic)
            # categóricos ficam como estão

        st.subheader("📋 Top 10 casos mais semelhantes")
        st.dataframe(formatted.reset_index(drop=True), use_container_width=True)

    # Gráfico de barras da Similaridade (%)
    st.subheader("📈 Similaridade (percentual)")
    plot_df = out.reset_index(drop=True).copy()
    plot_df["similaridade_%"] = plot_df["__similarity__"] * 100
    fig = px.bar(
        plot_df,
        y="similaridade_%", x=plot_df.index.astype(str),
        labels={"x": "Caso", "similaridade_%": "Similaridade (%)"},
        title="Ranking de Similaridade"
    )
    st.plotly_chart(fig, use_container_width=True)

    # Exporta CSV (valores reais + Similaridade %)
    export_df = df.loc[out.index, [c for c in cols_base if c in df.columns]].copy()
    export_df["Similaridade (%)"] = out["__similarity__"].values * 100
    st.download_button(
        label="⬇️ Baixar resultados (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name="rbc_covid_casos_semelhantes.csv",
        mime="text/csv"
    )
