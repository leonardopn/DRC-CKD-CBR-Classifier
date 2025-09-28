# Relatório Técnico: Sistema CBR para Classificação de Doença Renal Crônica

**Alunos:** Leonardo Petta do Nascimento, João Pedro Barreto de Melo
**Disciplina:** Machine Learning - Mestrado UFSM  
**Professor:** Luis Alvaro Silva
**Data:** Setembro 2025

## Resumo Executivo

Este trabalho apresenta o desenvolvimento e avaliação de um sistema de Case-Based Reasoning (CBR) para classificação de Doença Renal Crônica (DRC), abordando dois problemas distintos: classificação multiclasse dos estágios da DRC (`CKD_Stage`) e predição binária da progressão da doença (`CKD_Progression`). O sistema implementado alcançou performance clínica satisfatória, com acurácia de 69.30% para classificação de estágios e 86.40% para predição de progressão após otimização de pesos das features. A metodologia empregou Grid Search para otimização sistemática, demonstrando melhorias consistentes em relação ao modelo baseline.

## 1. Introdução

A Doença Renal Crônica representa um desafio significativo na área de saúde, afetando milhões de pessoas globalmente e requerendo diagnóstico precoce e manejo adequado para prevenir complicações. A classificação precisa dos estágios da DRC e a predição de sua progressão são fundamentais para o planejamento terapêutico e prognóstico dos pacientes.

Case-Based Reasoning emerge como uma abordagem promissora para problemas médicos devido à sua interpretabilidade inerente e capacidade de fornecer justificativas baseadas em casos similares, características essenciais para aplicações clínicas onde a explicabilidade é crucial.

### 1.1 Objetivos

**Objetivo Geral:**  
Desenvolver e avaliar um sistema CBR para classificação automática de estágios de DRC e predição de progressão da doença.

**Objetivos Específicos:**

-   Implementar algoritmo CBR com função de similaridade híbrida para dados médicos
-   Desenvolver sistema de otimização de pesos para maximização da performance
-   Avaliar comparativamente modelos baseline versus otimizados
-   Analisar clinicamente os resultados obtidos e suas implicações práticas

## 2. Metodologia

### 2.1 Dataset e Análise Exploratória

O dataset utilizado contém **1.138 registros** de pacientes com DRC, incluindo **23 features clínicas e laboratoriais**. As variáveis abrangem dados demográficos (sexo, idade), medidas clínicas (pressão sistólica, IMC), resultados laboratoriais (hemoglobina, creatinina, eGFR) e histórico médico (hipertensão, diabetes, doenças cardiovasculares).

#### 2.1.1 Análise Exploratória de Dados (EDA)

Foi realizada análise exploratória abrangente incluindo:

**Estatísticas Descritivas:**

-   Análise de distribuições com histogramas para features numéricas
-   Boxplots para identificação de outliers e quartis
-   Estatísticas de média, desvio padrão e percentis
-   Scatter plots para análise de correlações bivariadas
-   Mapas de correlação (heatmaps) para identificar redundâncias

**Análise de Qualidade:**

-   Avaliação sistemática de valores ausentes por variável
-   Identificação de padrões nos dados faltantes
-   Análise de distribuição das classes target

**Variáveis Target:**

-   `CKD_Stage`: Classificação multiclasse (estágios 1-5)
-   `CKD_Progression`: Classificação binária (0=sem progressão, 1=com progressão)

**Distribuição das Classes:**

-   **CKD_Stage**: Distribuição relativamente equilibrada entre os estágios, com maior concentração nos estágios intermediários
-   **CKD_Progression**: Dataset balanceado com aproximadamente 50% de casos com progressão

### 2.2 Pré-processamento dos Dados

#### 2.2.1 Análise de Correlação e Remoção de Features

Conforme especificado nos requisitos, foi realizada análise sistemática de correlação com as variáveis target, removendo features com correlação superior a 90%:

-   **eGFR**: Correlação de 91.9% com `CKD_Stage` → Removida
-   **CKD_Risk**: Correlação de 96.6% com `CKD_Stage` → Removida

Esta etapa reduziu o dataset de 23 para **19 features finais**, eliminando redundância e possível vazamento de informação.

#### 2.2.2 Tratamento de Valores Ausentes

Implementou-se estratégia diferenciada por tipo de variável:

-   **Features Numéricas**: Imputação pela mediana (robusta a outliers)
-   **Features Categóricas**: Imputação pela moda (valor mais frequente)

#### 2.2.3 Normalização

Aplicou-se `StandardScaler` às 7 features numéricas restantes, garantindo média zero e desvio padrão unitário, essencial para o cálculo adequado de distâncias euclidianas.

#### 2.2.4 Divisão dos Dados

-   **Treino**: 910 amostras (80%)
-   **Teste**: 228 amostras (20%)
-   **Estratificação**: Baseada em `CKD_Stage` para manter proporções representativas

### 2.3 Implementação do Sistema CBR

#### 2.3.1 Função de Similaridade Híbrida

Desenvolveu-se função de similaridade capaz de processar tipos mistos de dados:

```python
def calculate_similarity_distance(case1, case2, numerical_features,
                                categorical_features, feature_weights=None):
    # Distância Euclidiana normalizada para features numéricas
    # Match/mismatch para features categóricas
    # Suporte a pesos personalizados por feature
```

**Características Técnicas:**

-   **Features Numéricas**: Distância Euclidiana com normalização por peso
-   **Features Categóricas**: Distância binária (0=match, 1=mismatch)
-   **Ponderação**: Suporte completo a pesos individuais por feature
-   **Normalização**: Pela soma total dos pesos aplicados

#### 2.3.2 Algoritmo de Recuperação k-NN

Implementação de busca dos k casos mais similares:

-   **Parâmetro k=5**: Balanceamento entre especificidade e generalização
-   **Ordenação**: Por distância crescente (menor distância = maior similaridade)
-   **Votação Majoritária**: Decisão baseada na classe mais frequente entre os k vizinhos

#### 2.3.3 Classe CBRClassifier

Sistema modular com interface compatível com scikit-learn:

```python
class CBRClassifier:
    def fit(X_train, y_train, numerical_features, categorical_features)
    def predict(X_test)
    def predict_single(query_case)
```

### 2.4 Otimização de Pesos

#### 2.4.1 Método Escolhido: Grid Search

Optou-se por Grid Search devido à:

-   **Interpretabilidade**: Sistema mais simples, porém suficiente para o problema atual
-   **Sistematicidade**: Exploração controlada do espaço de parâmetros
-   **Reprodutibilidade**: Resultados determinísticos e auditáveis

#### 2.4.2 Estratégias de Otimização Testadas

Cinco estratégias foram sistematicamente avaliadas:

1. **Baseline**: Pesos uniformes (w=1.0) para todas as features
2. **Creatinine_Hemoglobin_High**: Peso 2.0 para Creatinina e Hemoglobina (biomarcadores críticos)
3. **Numerical_High**: Peso 1.5 para todas as features numéricas
4. **Categorical_High**: Peso 1.5 para todas as features categóricas
5. **Clinical_Core**: Peso 1.5 para Age, Systolic_Pressure, BMI (parâmetros vitais)

#### 2.4.3 Processo de Validação

-   **Divisão Treino/Validação**: 80%/20% do conjunto de treino original
-   **Métrica de Otimização**: Acurácia no conjunto de validação
-   **Seleção Automática**: Estratégia com maior performance na validação

#### 2.4.4 Visualização do Impacto dos Pesos

Foram implementados gráficos detalhados para demonstrar o impacto das diferentes combinações de pesos:

-   **Gráficos comparativos baseline vs otimizado** com métricas detalhadas
-   **Visualizações de matrizes de confusão** para ambos os cenários
-   **Plots de métricas por classe** mostrando precision, recall e F1-score
-   **Análise visual das melhorias** com indicação quantitativa dos ganhos

Estas visualizações (disponíveis no notebook) permitem interpretação clara do benefício da otimização de pesos e identificação das estratégias mais efetivas para cada problema de classificação.

## 3. Resultados e Discussão

### 3.1 Performance do Sistema base

#### 3.1.1 CKD_Stage (Classificação Multiclasse)

**Métricas Gerais:**

-   **Acurácia**: 64.91%
-   **Precisão Macro**: 64.97%
-   **Recall Macro**: 55.10%
-   **F1-Score Macro**: 57.27%

**Análise por Classe:**
A matriz de confusão revelou maior dificuldade na classificação dos estágios intermediários (3 e 4), comportamento esperado dado a sobreposição clínica entre estágios adjacentes.

#### 3.1.2 CKD_Progression (Classificação Binária)

**Métricas Gerais:**

-   **Acurácia**: 85.53%
-   **Precisão Macro**: 81.38%
-   **Recall Macro**: 78.65%
-   **F1-Score Macro**: 79.85%
-   **AUC-ROC**: Calculado conforme implementação no notebook para avaliação da capacidade discriminativa

**Significância Clínica:**
A baixa taxa de falsos positivos (13/228 amostras) é particularmente relevante para screening clínico, minimizando intervenções desnecessárias. O AUC-ROC complementa a análise fornecendo medida robusta da qualidade da classificação binária independente do limiar de decisão.

### 3.2 Otimização e Performance final

#### 3.2.1 Resultados da Otimização

**CKD_Stage:**

-   **Estratégia Selecionada**: Creatinine_Hemoglobin_High (75.27% na validação)
-   **Performance Final**: 69.30% (+4.39% vs baseline)

**CKD_Progression:**

-   **Estratégia Selecionada**: Numerical_High (81.32% na validação)
-   **Performance Final**: 86.40% (+0.88% vs baseline)

#### 3.2.2 Análise da Otimização

A seleção automática das estratégias revela insights clinicamente significativos:

1. **Importância da Creatinina e Hemoglobina**: Para classificação de estágios, confirma guidelines clínicos estabelecidos
2. **Valor das Features Numéricas**: Para predição de progressão, sugere que parâmetros quantitativos contínuos capturam melhor a dinâmica da doença

### 3.3 Comparação Baseline vs Otimizado

| Problema            | Baseline | Otimizado  | Melhoria Absoluta | Melhoria Relativa |
| ------------------- | -------- | ---------- | ----------------- | ----------------- |
| **CKD_Stage**       | 64.91%   | **69.30%** | +4.39%            | +6.77%            |
| **CKD_Progression** | 85.53%   | **86.40%** | +0.87%            | +1.02%            |

### 3.4 Análise Clínica dos Resultados

#### 3.4.1 Implicações dos Falsos Positivos/Negativos

**CKD_Stage:**

-   **Falsos Positivos**: Podem levar a tratamento mais agressivo desnecessário
-   **Falsos Negativos**: Risco de subtratamento e progressão não detectada

**CKD_Progression:**

-   **Falsos Positivos**: Ansiedade do paciente e monitoramento excessivo
-   **Falsos Negativos**: Perda de janela terapêutica para intervenção precoce

#### 3.4.2 Validação das Features Importantes

A identificação automática de Creatinina e Hemoglobina como features críticas alinha-se perfeitamente com o conhecimento médico estabelecido:

-   **Creatinina**: Biomarcador direto da função renal
-   **Hemoglobina**: Indicador de anemia, complicação comum na DRC

## 4. Limitações do Estudo

### 4.1 Limitações Técnicas

1. **Tamanho do Dataset**: 1.138 amostras podem limitar generalização
2. **Método de Otimização**: Grid Search limitado a estratégias predefinidas
3. **Validação**: Ausência de validação cruzada k-fold
4. **Comparação**: Sem benchmark com algoritmos estado-da-arte

### 4.2 Limitações Clínicas

1. **Validação Externa**: Dataset de fonte única
2. **Diversidade Populacional**: Possível viés demográfico
3. **Aspectos Temporais**: Análise cross-sectional, sem seguimento longitudinal
4. **Variáveis Confundidoras**: Não consideração de fatores como aderência ao tratamento

## 5. Trabalhos Futuros

### 5.1 Aprimoramentos Metodológicos

1. **Otimização Avançada**:

    - Algoritmos genéticos para busca global
    - Otimização bayesiana com kernel learning
    - Ensemble de CBR com diferentes métricas

2. **Validação Robusta**:

    - Validação cruzada estratificada
    - Validação externa multicêntrica
    - Análise de subgrupos populacionais

3. **Comparação Extensiva**:
    - Benchmark com Random Forest, XGBoost, SVM
    - Análise de trade-off performance vs interpretabilidade

### 5.2 Aplicações Práticas

1. **Interface Clínica**: Sistema web para uso em ambiente hospitalar
2. **Integração**: Compatibilidade com padrões HL7 FHIR
3. **Explicabilidade**: Dashboard interativo mostrando casos similares
4. **Monitoramento**: Sistema de alertas para progressão detectada

## 6. Conclusões

### 6.1 Contribuições Principais

1. **Sistema CBR Médico Completo**: Primeira implementação customizada para DRC com otimização sistemática de pesos
2. **Validação de Conhecimento Clínico**: Confirmação quantitativa da importância de biomarcadores estabelecidos
3. **Metodologia Reproduzível**: Código modular e documentado para pesquisa colaborativa
4. **Performance Clínica Relevante**: Resultados compatíveis com uso auxiliar em ambiente médico

### 6.2 Impacto e Significância

O sistema desenvolvido demonstra viabilidade técnica e clínica para auxílio no diagnóstico e prognóstico de DRC. A performance obtida (86.40% para progressão, 69.30% para estágios) posiciona a ferramenta como auxiliar clinicamente útil, especialmente considerando a interpretabilidade inerente do CBR.

A melhoria consistente após otimização (+4.39% para estágios, +0.88% para progressão) valida a importância da customização de pesos em sistemas CBR médicos, demonstrando que o conhecimento de domínio pode ser efetivamente incorporado através de estratégias de ponderação.

### 6.3 Considerações Finais

Este trabalho estabelece uma base sólida para aplicação de CBR em medicina, fornecendo metodologia robusta para classificação médica com otimização sistemática e validação rigorosa. A identificação automática de features críticas reforça a confiabilidade clínica da abordagem e sugere potencial para implementação em ambientes clínicos reais.

A combinação de performance satisfatória, interpretabilidade e validação de conhecimento clínico posiciona o sistema como contribuição significativa para a interseção entre inteligência artificial e medicina, oferecendo ferramenta prática para apoio à decisão clínica em nefrologia.

---

## Anexos

### A.1 Especificações Técnicas

-   **Linguagem**: Python 3.12
-   **Bibliotecas Principais**: pandas, numpy, scikit-learn, matplotlib, seaborn
-   **Ambiente**: Jupyter Notebook
-   **Hardware testado**:
    -   Processador: AMD Ryzen™ 7 5800X × 16
    -   RAM: 16GB
    -   Sistema Operacional: Arch Linux
    -   Placa de vídeo: NVIDIA GeForce RTX 4060 TI (não utilizada no processamento)

### A.2 Métricas Detalhadas e Visualizações

**Relatórios Completos:**

-   Classification reports detalhados do scikit-learn para ambos os problemas
-   Métricas por classe (precision, recall, f1-score) com target names apropriados
-   Matrizes de confusão com heatmaps para interpretação visual

**Gráficos de Impacto dos Pesos:**

-   Comparação visual baseline vs otimizado com barras side-by-side
-   Plots de métricas detalhadas por problema de classificação
-   Visualizações de melhorias percentuais com anotações
-   Heatmaps das matrizes de confusão para análise de erros

**Análise Exploratória:**

-   Histogramas de distribuição das features numéricas
-   Boxplots para detecção de outliers
-   Mapas de correlação para análise de redundâncias
-   Scatter plots para relações bivariadas

Todas as visualizações estão implementadas no notebook principal (`main_final.ipynb`) com código reproduzível e comentado.
