# Detecção de Fraudes em Cartões de Crédito com Machine Learning

Projeto desenvolvido como parte do desafio prático de Machine Learning e Detecção de Anomalias da **Digital Innovation One (DIO)**.

---

## 1. O Problema e o Desafio do Desbalanceamento

Transações fraudulentas são eventos raros no sistema financeiro, mas com potencial de causar prejuízos milionários e perda de confiança por parte dos clientes.

O dataset utilizado contém transações realizadas por cartões de crédito em setembro de 2013 por portadores de cartões europeus:
* **Total de transações:** 284.807
* **Transações legítimas (Classe 0):** 284.315 (99,827%)
* **Transações fraudulentas (Classe 1):** 492 (0,173%)

### Por que a acurácia é uma métrica enganosa?
Em conjuntos com desbalanceamento severo (apenas 0,17% de fraudes), um classificador trivial que responda *"não é fraude"* para 100% dos casos atingirá **99,83% de acurácia**, mas terá **0% de recall**, deixando passar todas as fraudes e sendo completamente inútil para o negócio.

Por essa razão, o projeto foca nas métricas que realmente importam:
* **Recall (Sensibilidade):** Percentual de fraudes reais que o modelo conseguiu capturar. (Quanto maior o recall, menor o prejuízo financeiro com fraudes não detectadas).
* **Precisão:** Percentual de acerto quando o modelo dispara um alarme de fraude. (Quanto maior a precisão, menor o número de clientes legítimos com cartões bloqueados por engano).
* **F1-Score:** Média harmônica entre Precisão e Recall, servindo como indicador de equilíbrio.
* **PR-AUC (Precision-Recall AUC):** Área sob a curva Precisão-Recall, considerada a métrica mais confiável e robusta para avaliação em cenários de classes ultrarraras.

---

## 2. Engenharia de Features e Prevenção de *Data Leakage*

1. **Transformação Logarítmica (`Amount`):**
   A variável de valor da transação apresenta forte assimetria (*heavy-tailed distribution*). Foi aplicada a transformação `np.log1p(df["Amount"])` para estabilizar a variância e aproximar os dados de uma distribuição normal.

2. **Prevenção Rigorosa de Vazamento de Dados (*Data Leakage*):**
   * A divisão entre treino e teste (`train_test_split`) com estratificação (`stratify=y`, 80/20) foi realizada **antes** de qualquer ajuste de escalonamento.
   * O `StandardScaler` foi ajustado (`fit_transform`) **exclusivamente sobre os dados de treino**. O conjunto de teste recebeu apenas a transformação (`transform`), garantindo que média e desvio padrão do teste não influenciassem o aprendizado dos modelos.
   * A variável `Time` também foi padronizada para evitar distorções de escala em algoritmos lineares.

3. **SMOTE vs. Calibração de Limiar (*Threshold Tuning*):**
   * O teste com **SMOTE** (oversampling sintético) elevou o recall, mas provocou uma queda abrupta na precisão (cerca de 6% na Regressão Logística, gerando mais de 1.500 falsos positivos).
   * Constatou-se empiricamente que manter a **distribuição natural dos dados** e **calibrar o limiar de decisão (*threshold tuning*)** produz um modelo com probabilidades calibradas e desempenho significativamente superior.

---

## 3. Comparação de Modelos

Foram implementadas funções modulares (`encontrar_melhor_limiar` e `avaliar_modelo`) para testar os limiares de decisão (de 0.10 a 0.90) e avaliar o desempenho em cada algoritmo.

### Tabela Comparativa Consolidada

| Modelo | Limiar Ótimo | Precisão (Fraude) | Recall (Fraude) | F1-Score | PR-AUC | AUC-ROC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Regressão Logística** *(Baseline)* | 0.10 | 72,22% | 79,59% | 0.7573 | 0.7388 | 0.9573 |
| **Random Forest** | 0.50 | 83,51% | 82,65% | 0.8308 | 0.8214 | **0.9840** |
| **XGBoost** *(Campeão)* | **0.20** | **91,21%** | **84,69%** | **0.8783** | **0.8707** | 0.9683 |

### Análise de Negócio dos Resultados:
* **Regressão Logística:** Serviu como um baseline robusto. Com o limiar ajustado em 0.10, atingiu F1 de 0.7573.
* **Random Forest (`class_weight='balanced_subsample'`):** Saltou para F1 de 0.8308 com limiar natural de 0.50, equilibrando recall (82,65%) e precisão (83,51%).
* **XGBoost (Campeão):** Foi o algoritmo mais eficiente em capturar padrões complexos. No limiar ótimo de **0.20**, atingiu **84,69% de recall** e impressionantes **91,21% de precisão**, com **F1-Score de 0.8783** e **PR-AUC de 0.8707**. Ou seja, detecta ~85% das fraudes com menos de 9% de falsos positivos.

---

## 4. Limiar de Decisão e Interpretabilidade com SHAP

### Limiar Escolhido
Para o modelo vencedor (**XGBoost**), o limiar adotado foi **0.20**:
* No limiar padrão de 0.50, o modelo era conservador (Recall 80,61% e Precisão 92,94%).
* Reduzindo o ponto de corte para **0.20**, o modelo ganha **+4% de recall (atingindo 84,69%)**, mantendo a precisão acima de **91%**, atingindo o ponto ótimo da curva F1.

### O que o SHAP revelou?
Utilizando o `shap.TreeExplainer` no XGBoost:
1. **Visão Global (`summary_plot`):**
   * As variáveis que exercem o maior impacto na decisão de fraude são: **`V14`**, **`V17`**, **`V12`**, **`V10`** e **`V4`**.
   * Valores fortemente negativos em `V14`, `V12` e `V17` empurram drasticamente a probabilidade para **Fraude**.
   * Valores elevados em `V4` e `V11` também estão associados a transações fraudulentas.
2. **Visão Local (`waterfall_plot`):**
   * Ao decompor uma fraude real individual, foi possível verificar com exatidão a contribuição aditiva de cada variável (ex: como `V14` com valor negativo empurrou o *log-odds* da predição base até a classificação final de fraude).

---

## 5. Diferenciais Implementados em Relação ao Pipeline Base

1. **Prevenção de Data Leakage:** Separação estrita de treino/teste antes do `StandardScaler`, além da padronização e tratamento de `Amount` e `Time`.
2. **Funções Reutilizáveis:** Criação de `encontrar_melhor_limiar()` e `avaliar_modelo()`, tornando o notebook modular, sem repetição de código (DRY) e padronizando as análises gráficas (Matriz de Confusão, ROC Curve e PR Curve lado a lado).
3. **Threshold Tuning Dinâmico:** Implementação de varredura automatizada para seleção do ponto de corte ótimo baseado no F1-Score da classe minoritária.
4. **XGBoost com Hiperparâmetros Ajustados:** Otimização de taxa de aprendizado e profundidade (`learning_rate=0.04`, `max_depth=10`), superando com folga os outros algoritmos.
5. **Explicabilidade Completa com SHAP:** Inclusão tanto da visualização global (*beeswarm summary plot*) quanto da análise de predição individual (*waterfall plot*).

---

## Como Executar o Projeto

1. Clone o repositório:
   ```bash
   git clone https://github.com/<seu-usuario>/dio-challenge-python-anomaly-detection-fraud.git
   cd dio-challenge-python-anomaly-detection-fraud
   ```

2. Instale as dependências:
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn xgboost shap matplotlib
   ```

3. Abra e execute o notebook:
   ```bash
   jupyter notebook fraud-detection.ipynb
   ```
   *(O dataset de 284k linhas é baixado automaticamente no primeiro carregamento a partir do link oficial do TensorFlow).*
