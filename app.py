"""
Aplicação Streamlit da Global Solution: classificação de roubo ou furto.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras

PASTA = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Segurança Pública Inteligente | MLP",
    page_icon="🔎",
    layout="centered",
)


@st.cache_resource(show_spinner="Carregando modelo de classificação...")
def carregar_arquivos():
    """Carrega uma única vez o modelo, o pré-processador e os metadados."""
    with (PASTA / "metadados.json").open(encoding="utf-8") as arquivo:
        metadados = json.load(arquivo)
    preprocessador = joblib.load(PASTA / "preprocessador.joblib")
    modelo = keras.models.load_model(PASTA / "modelo_mlp.keras", compile=False)
    return metadados, preprocessador, modelo


try:
    metadados, preprocessador, modelo = carregar_arquivos()
except Exception as erro:
    st.error("Não foi possível carregar os arquivos treinados.")
    st.code(f"{type(erro).__name__}: {erro}")
    st.info("Verifique os três arquivos na pasta e instale as versões indicadas em requirements.txt.")
    st.stop()

st.title("Segurança Pública Inteligente")
st.write("**Classificação de ocorrências de roubo ou furto na cidade de São Paulo**")
st.caption("Projeto • Global Solution 2026 • Rede neural MLP")

st.info(
    "Esta ferramenta classifica o **tipo de uma ocorrência hipotética ou já registrada**. "
    "Ela **não prevê** a probabilidade de um crime acontecer em um bairro, dia ou horário."
)

st.subheader("Características da ocorrência")
st.write("Preencha os campos abaixo para consultar a classificação estimada pela rede neural.")

# Mantém as categorias exatamente como foram utilizadas no treinamento.
# Categorias fora das listas mais frequentes são transformadas em 'Outros'.
bairros = sorted(set(metadados["bairros_mantidos"]) | {"Outros"})
locais = sorted(set(metadados["locais_mantidos"]) | {"Outros"})

with st.form("formulario_ocorrencia"):
    bairro = st.selectbox("Bairro", bairros, index=bairros.index("PINHEIROS") if "PINHEIROS" in bairros else 0)
    dia = st.selectbox("Dia da semana", metadados["dias_semana"])
    periodo = st.selectbox("Período do dia", metadados["periodos"])
    local = st.selectbox("Tipo de local", locais, index=locais.index("Via Pública") if "Via Pública" in locais else 0)
    enviado = st.form_submit_button("Classificar ocorrência", type="primary", use_container_width=True)

if enviado:
    # Recria a mesma ordem e os mesmos nomes de colunas usados no Colab.
    entrada = pd.DataFrame([{ 
        "BAIRRO": bairro,
        "dia_semana": dia,
        "periodo": periodo,
        "DESCR_TIPOLOCAL": local,
    }])[metadados["atributos"]]

    entrada["BAIRRO"] = entrada["BAIRRO"].where(
        entrada["BAIRRO"].isin(metadados["bairros_mantidos"]), "Outros"
    )
    entrada["DESCR_TIPOLOCAL"] = entrada["DESCR_TIPOLOCAL"].where(
        entrada["DESCR_TIPOLOCAL"].isin(metadados["locais_mantidos"]), "Outros"
    )

    try:
        entrada_processada = preprocessador.transform(entrada).astype("float32")
        # Verificação extra para garantir a compatibilidade entre artefatos.
        dimensao_esperada = modelo.input_shape[-1]
        if entrada_processada.shape[1] != dimensao_esperada:
            raise ValueError(
                f"O modelo espera {dimensao_esperada} atributos, "
                f"mas o pré-processador gerou {entrada_processada.shape[1]}."
            )
        prob_roubo = float(np.asarray(modelo.predict(entrada_processada, verbose=0)).reshape(-1)[0])
        prob_roubo = float(np.clip(prob_roubo, 0, 1))
        prob_furto = 1 - prob_roubo
        limiar = float(metadados.get("limiar", 0.5))
        classe = "Roubo" if prob_roubo >= limiar else "Furto"
    except Exception as erro:
        st.error("Ocorreu um erro durante a classificação.")
        st.code(f"{type(erro).__name__}: {erro}")
        st.stop()

    st.divider()
    st.subheader("Resultado da classificação")
    st.metric("Classe estimada", classe)
    col1, col2 = st.columns(2)
    col1.metric("Roubo", f"{prob_roubo * 100:.1f}%")
    col2.metric("Furto", f"{prob_furto * 100:.1f}%")
    st.progress(prob_roubo, text="Pontuação da classe roubo")
    st.caption(
        "Os percentuais representam as saídas do modelo para as duas classes, "
        "não uma previsão de ocorrência futura."
    )
    with st.expander("Ver dados usados na classificação"):
        st.dataframe(entrada, hide_index=True, use_container_width=True)

st.divider()
st.caption(
    f"Modelo treinado: {metadados.get('modelo_escolhido', 'MLP')} • "
    "Fonte: Secretaria da Segurança Pública do Estado de São Paulo (SSP-SP) • "
)
