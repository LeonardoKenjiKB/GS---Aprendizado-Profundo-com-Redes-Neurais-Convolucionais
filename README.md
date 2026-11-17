# Segurança Pública Inteligente em São Paulo — Global Solution 2026

Aplicação acadêmica que usa uma rede neural **MLP** para classificar características de uma ocorrência como **roubo** ou **furto**, com dados públicos da SSP-SP.

> **Importante:** a aplicação não prevê se um crime vai acontecer, nem o risco de crimes em um local ou horário. A classificação é condicionada à hipótese de uma ocorrência já ter acontecido.

## Arquivos

- `app.py`: formulário e inferência com Streamlit.
- `modelo_mlp.keras`: rede neural escolhida no notebook (E3).
- `preprocessador.joblib`: transformador categórico ajustado no treinamento.
- `metadados.json`: atributos, categorias, rótulos e limiar.
- `GS_Aprendizado_Profundo_com_Redes_Neurais_Convolucionais.ipynb`: notebook com preparação, treinos e avaliação.
- `requirements.txt`: dependências para reproduzir a execução.

## Executar localmente

Use **Python 3.11 ou 3.12**. Abra o terminal do VS Code na pasta do projeto e execute:

```bash
python -m venv .venv
```

Ative o ambiente virtual:

- **Windows (PowerShell):** `.venv\Scripts\Activate.ps1`
- **Windows (Prompt de Comando):** `.venv\Scripts\activate.bat`
- **Windows (Git Bash):** `source .venv/Scripts/activate`
- **macOS/Linux:** `source .venv/bin/activate`

Depois instale as bibliotecas e execute o site:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

O Streamlit fornecerá um endereço local, geralmente `http://localhost:8501`.

## Publicação

1. Crie um repositório no GitHub e envie os arquivos do projeto, **sem** a pasta `.venv`.
2. Entre em https://share.streamlit.io/ e escolha **Create app**.
3. Selecione seu repositório, branch e `app.py` como arquivo principal.
4. Configure Python 3.11 ou 3.12 nas opções avançadas, quando disponíveis.
5. Depois do deploy, abra o endereço gerado e teste o formulário.
6. Adicione o link público da aplicação neste README e envie-o ao professor.

**Link público da aplicação:** _preencher após a publicação_.

## Dados e metodologia

- Fonte: [SSP-SP — Estatísticas e consultas](https://www.ssp.sp.gov.br/estatistica/consultas).
- Base: celulares subtraídos, arquivo disponibilizado em 2026.
- Recorte: registros da cidade de São Paulo classificados como furto (art. 155) ou roubo (art. 157).
- Amostragem: 20.000 registros, após os filtros e o tratamento descritos no notebook.
- Entradas: bairro, dia da semana, período do dia e tipo de local.
- Pré-processamento: agrupamento das categorias raras em `Outros` e codificação one-hot.
- Modelo: MLP E3, selecionada pelo F1-score para a classe roubo na validação.
- Na avaliação final do notebook, a acurácia foi de aproximadamente 71,6%; confira o notebook para as métricas detalhadas e limitações.

## Limitações

A rede possui desempenho moderado, especialmente em recall da classe roubo. Os resultados dependem das características e limitações dos registros públicos e não se aplicam à previsão de crimes futuros. As saídas do modelo não devem ser interpretadas como probabilidades calibradas de risco.

