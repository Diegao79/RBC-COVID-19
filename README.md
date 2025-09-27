# 🦠 Raciocínio Baseado em Casos (RBC) – COVID-19

Projeto acadêmico utilizando **Raciocínio Baseado em Casos (RBC)** para analisar dados reais da COVID-19.  
A aplicação foi desenvolvida em **Python** com interface interativa em **Streamlit**.

---

## 🚀 Funcionalidades
- Upload automático do dataset COVID-19 via Kaggle.
- Pré-processamento dos dados:
  - Remoção de colunas irrelevantes/nulas
  - Imputação de valores faltantes
  - Normalização numérica [0,1]
  - Codificação categórica
- Definição de atributos relevantes para análise.
- Implementação do RBC:
  - Similaridade Numérica: distância Manhattan normalizada
  - Similaridade Categórica: correspondência exata
  - Combinação com pesos (50% numérico, 50% categórico)
- Interface gráfica em **Streamlit**:
  - Formulário para inserir um novo caso
  - Escolha de atributos para comparação
  - Exibição de casos mais semelhantes em **tabela** e **gráfico**
  - Opção para baixar resultados em CSV

---

## 🛠️ Tecnologias Utilizadas
- [Python 3.10+](https://www.python.org/)
  
- [Streamlit](https://streamlit.io/)
  
- [Pandas](https://pandas.pydata.org/)
  
- [NumPy](https://numpy.org/)
  
- [Scikit-Learn](https://scikit-learn.org/)
  
- [Plotly](https://plotly.com/python/)
  
- [KaggleHub](https://github.com/Kaggle/kagglehub)

---

## 📂 Estrutura do Projeto
RBC_COVID/
│── app.py # Interface Streamlit
│── rbc.py # Funções de pré-processamento e RBC
│── requirements.txt # Dependências do projeto
└── README.md # Este arquivo

---

## ⚙️ Como Executar

### 1. Clone o repositório
```bash
git clone 
cd RBC_COVID
2. Crie e ative o ambiente virtual
No Windows (PowerShell):

powershell
Copiar código
python -m venv .venv
.venv\Scripts\Activate.ps1

3. Instale as dependências
pip install -r requirements.txt

4. Execute a aplicação

streamlit run app.py
O sistema abrirá em:
👉 http://localhost:8501

📊 Exemplos de Uso
Comparar Brasil com outros países em julho/2025 → Moçambique e Gana (~75% de similaridade).

Comparar América do Sul em novos casos → Chile, Argentina e Peru aparecem entre os mais semelhantes.

👨‍💻 Autores
Diego Rafael Muller e Guilherme Massinhani de Souza – Desenvolvimento e documentação
