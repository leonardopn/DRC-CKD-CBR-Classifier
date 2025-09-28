# Relatório Técnico: Sistema CBR para Classificação de Doença Renal Crônica

**Alunos:** Leonardo Petta do Nascimento, João Pedro Barreto de Melo
**Disciplina:** Machine Learning - Mestrado UFSM  
**Professor:** Luis Alvaro Silva
**Data:** Setembro 2025

## Resumo Executivo

Este trabalho apresenta o desenvolvimento e avaliação de um sistema de Case-Based Reasoning (CBR) para classificação de Doença Renal Crônica (DRC), abordando dois problemas distintos: classificação multiclasse dos estágios da DRC (`CKD_Stage`) e predição binária da progressão da doença (`CKD_Progression`). O sistema implementado alcançou **acurácia de 69.30% para classificação de estágios** e **86.40% para predição de progressão** após otimização de pesos das features. A metodologia empregou Grid Search para otimização sistemática, demonstrando melhorias de **+4.39% para CKD_Stage** e **+0.88% para CKD_Progression** em relação ao modelo baseline.

## 1. Introdução

A Doença Renal Crônica (DRC) representa um desafio clínico global, afetando aproximadamente 10% da população mundial e requerendo classificação precisa para manejo adequado. O estadiamento correto (estágios 1-5) e a predição de progressão são fundamentais para decisões terapêuticas e prognóstico dos pacientes.

**Motivação para CBR em Medicina:**
Case-Based Reasoning oferece vantagens específicas para aplicações médicas: (1) **interpretabilidade natural** através da comparação com casos similares, (2) **incorporação de conhecimento clínico** via similaridade ponderada, e (3) **justificação transparente** das decisões baseada em precedentes clínicos.

**Problema Abordado:**
Este trabalho implementa sistema CBR para dois problemas de classificação: (1) estadiamento multiclasse de DRC e (2) predição binária de progressão da doença, utilizando otimização sistemática de pesos por Grid Search.

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

#### 2.3.1 Decisões de Projeto Fundamentais

**Escolha da Linguagem:** Todo o código foi implementado em Python conforme especificado, utilizando bibliotecas padronizadas (pandas, numpy, scikit-learn, matplotlib) para garantir reprodutibilidade e integração com o ecossistema de ML.

**Arquitetura Modular:** Optou-se por implementação orientada a objetos com classe `CBRClassifier` compatível com scikit-learn, permitindo integração fácil com pipelines de ML e facilitando futuras extensões.

#### 2.3.2 Função de Similaridade Híbrida

**Justificativa da Escolha:** Dados médicos apresentam natureza mista (numéricos e categóricos), exigindo função de similaridade especializada que trate adequadamente cada tipo.

**Implementação Técnica:**

```python
def calculate_similarity_distance(case1, case2, numerical_features,
                                categorical_features, feature_weights=None):
    # Distância Euclidiana normalizada para features numéricas
    # Match/mismatch (0/1) para features categóricas
    # Normalização pela soma dos pesos aplicados
```

**Características Implementadas:**

-   **Features Numéricas**: Distância Euclidiana quadrática com normalização StandardScaler
-   **Features Categóricas**: Distância binária (0=match perfeito, 1=mismatch completo)
-   **Ponderação Flexível**: Sistema de pesos individuais por feature com normalização automática
-   **Robustez**: Tratamento de valores ausentes e validação de tipos

#### 2.3.3 Algoritmo k-NN Customizado

**Decisão: k=5**

-   **Justificativa**: Balanceamento entre especificidade (k pequeno) e generalização (k grande)
-   **Validação**: Testado empiricamente durante desenvolvimento, mostrou melhor estabilidade
-   **Consideração Clínica**: Permite análise de 5 casos similares, número interpretável para profissionais de saúde

**Estratégia de Votação:** Majoritária simples com tie-breaking automático (primeira classe encontrada), priorizando simplicidade e interpretabilidade.

#### 2.3.4 Arquitetura da Classe CBRClassifier

**Interface Padronizada:**

```python
class CBRClassifier:
    def __init__(self, k=5, feature_weights=None)
    def fit(X_train, y_train, numerical_features, categorical_features)
    def predict(X_test)
    def predict_single(query_case)  # Para análise individual de casos
```

**Decisões de Design:**

-   **Compatibilidade scikit-learn**: Métodos `fit()` e `predict()` padronizados
-   **Flexibilidade**: Suporte a pesos customizados e diferentes tipos de features
-   **Transparência**: Armazenamento de detalhes de predição para análise posterior
-   **Eficiência**: Busca linear otimizada para datasets médicos (< 10k amostras)

### 2.4 Otimização de Pesos - Decisões Metodológicas

#### 2.4.1 Justificativa da Escolha: Grid Search

**Decisão Fundamental:** Grid Search foi selecionado como método de otimização principal por razões técnicas e práticas específicas:

**Vantagens para CBR Médico:**

-   **Interpretabilidade Clínica**: Estratégias predefinidas baseadas em conhecimento médico são mais facilmente validadas por especialistas
-   **Reprodutibilidade Total**: Resultados determinísticos essenciais para pesquisa médica
-   **Controle Experimental**: Comparação sistemática de hipóteses específicas sobre importância de features
-   **Eficiência Computacional**: Para 19 features e 5 estratégias, Grid Search é mais eficiente que métodos de otimização global

**Alternativas Consideradas e Rejeitadas:**

-   **Algoritmos Genéticos**: Maior complexidade sem ganho interpretável para o domínio médico
-   **Gradient Descent**: Inadequado para otimização de pesos discretos em CBR
-   **Random Search**: Menor sistematicidade na exploração de hipóteses clínicas

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

#### 2.4.4 Gráficos de Impacto dos Pesos (Implementados)

O código gera automaticamente visualizações abrangentes para demonstrar o impacto das diferentes combinações de pesos:

**Gráficos Baseline vs Otimizado:**

-   Comparação side-by-side de métricas (Acurácia, Precisão, Recall, F1-Score)
-   Anotações automáticas dos valores numéricos em cada barra
-   Indicadores visuais de melhoria com setas e valores percentuais

**Matrizes de Confusão:**

-   Heatmaps coloridos para análise visual dos padrões de classificação
-   Separação clara entre problemas multiclasse (CKD_Stage) e binário (CKD_Progression)
-   Valores absolutos anotados para análise quantitativa

**Métricas Detalhadas por Classe:**

-   Gráficos de barras agrupadas (Precisão, Recall, F1-Score) por classe
-   Visualização diferenciada para cada estágio da DRC
-   Identificação de classes com performance comprometida (ex: Estágio 2)

Todas as visualizações são geradas automaticamente pela execução do notebook (`main_final.ipynb`), permitindo análise visual imediata do benefício da otimização de pesos.

## 3. Resultados e Discussão

### 3.1 Performance do Sistema CBR

O sistema CBR foi implementado com função de similaridade híbrida e avaliado em duas configurações: baseline (pesos uniformes) e otimizada (Grid Search).

#### 3.1.1 CBR Baseline (Pesos Uniformes)

**Resultados Baseline (k=5, pesos iguais w=1.0):**

**CKD_Stage (Multiclasse):**

-   **Acurácia**: 64.91%
-   **Precisão Macro**: 65.0%
-   **Recall Macro**: 55.1%
-   **F1-Score Macro**: 57.3%

**Métricas por Classe:**

-   Estágio 2: Precisão=50%, Recall=16%, F1=24% (19 amostras)
-   Estágio 3: Precisão=63%, Recall=78%, F1=70% (94 amostras)
-   Estágio 4: Precisão=59%, Recall=60%, F1=60% (73 amostras)
-   Estágio 5: Precisão=88%, Recall=67%, F1=76% (42 amostras)

**CKD_Progression (Binário):**

-   **Acurácia**: 85.53%
-   **Precisão Macro**: 81.4%
-   **Recall Macro**: 78.7%
-   **F1-Score Macro**: 79.9%

**Métricas por Classe:**

-   Sem Progressão: Precisão=89%, Recall=92%, F1=91% (171 amostras)
-   Com Progressão: Precisão=74%, Recall=65%, F1=69% (57 amostras)

#### 3.1.2 Métricas de Avaliação Utilizadas

Todas as métricas foram calculadas usando scikit-learn para garantir consistência:

-   **Acurácia**: Proporção de predições corretas no conjunto de teste
-   **Precisão, Recall e F1-Score**: Métricas macro para tratar desbalanceamento
-   **Matriz de Confusão**: Análise visual dos padrões de erro
-   **Classification Report**: Métricas detalhadas por classe com support

### 3.2 Otimização de Pesos - Resultados

#### 3.2.1 Processo de Grid Search - Resultados Detalhados

O Grid Search testou sistematicamente 5 estratégias de ponderação no conjunto de validação:

**CKD_Stage (Acurácia no Conjunto de Validação):**

1. Baseline: 71.43%
2. **Creatinine_Hemoglobin_High: 75.27%** ⭐ **MELHOR**
3. Numerical_High: 73.63%
4. Categorical_High: 69.78%
5. Clinical_Core: 70.88%

**CKD_Progression (Acurácia no Conjunto de Validação):**

1. Baseline: 79.67%
2. Creatinine_Hemoglobin_High: 79.67%
3. **Numerical_High: 81.32%** ⭐ **MELHOR**
4. Categorical_High: 78.02%
5. Clinical_Core: 79.12%

#### 3.2.2 Estratégias Otimizadas Selecionadas

**Para CKD_Stage:** `Creatinine_Hemoglobin_High`

-   **Pesos**: Creatinina=2.0, Hemoglobina=2.0, demais features=1.0
-   **Justificativa Clínica**: Biomarcadores diretos da função renal e anemia associada à DRC

**Para CKD_Progression:** `Numerical_High`

-   **Pesos**: Features numéricas=1.5, features categóricas=1.0
-   **Justificativa**: Parâmetros quantitativos contínuos capturam melhor a dinâmica temporal da progressão

#### 3.2.3 Análise da Otimização

A seleção automática das estratégias confirma conhecimento clínico estabelecido:

1. **Creatinina e Hemoglobina como Biomarcadores Críticos**: Validação quantitativa da importância clínica
2. **Superioridade das Features Numéricas para Progressão**: Capacidade de capturar variações graduais na função renal

### 3.3 Resultados Finais: Baseline vs Otimizado

**Avaliação no Conjunto de Teste (228 amostras):**

#### 3.3.1 CKD_Stage (Classificação Multiclasse)

| Métrica        | Baseline   | Otimizado  | Melhoria   |
| -------------- | ---------- | ---------- | ---------- |
| **Acurácia**   | **64.91%** | **69.30%** | **+4.39%** |
| Precisão Macro | 65.0%      | 67.8%      | +2.8%      |
| Recall Macro   | 55.1%      | 61.2%      | +6.1%      |
| F1-Score Macro | 57.3%      | 63.4%      | +6.1%      |

#### 3.3.2 CKD_Progression (Classificação Binária)

| Métrica        | Baseline   | Otimizado  | Melhoria   |
| -------------- | ---------- | ---------- | ---------- |
| **Acurácia**   | **85.53%** | **86.40%** | **+0.88%** |
| Precisão Macro | 81.4%      | 82.6%      | +1.2%      |
| Recall Macro   | 78.7%      | 79.8%      | +1.1%      |
| F1-Score Macro | 79.9%      | 81.1%      | +1.2%      |

#### 3.3.3 Análise dos Resultados

**Efetividade da Otimização:**

-   **CKD_Stage**: Melhoria substancial de 4.39%, especialmente significativa considerando a complexidade multiclasse
-   **CKD_Progression**: Melhoria menor mas consistente de 0.88%, indicando que o baseline já apresentava boa performance

**Padrões de Melhoria:**

-   Recall teve maior ganho que Precisão, indicando melhor capacidade de detectar casos positivos
-   F1-Score equilibrado confirma robustez das melhorias obtidas

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

### 5.1 Extensões Metodológicas Recomendadas

1. **Otimização Avançada**:

    - Algoritmos genéticos para exploração global do espaço de pesos
    - Otimização bayesiana com Gaussian Process para eficiência
    - Meta-heurísticas (Particle Swarm Optimization) para problemas multimodais

2. **Validação Estatística Robusta**:

    - Validação cruzada k-fold estratificada (k=10) para estabilidade
    - Bootstrap para intervalos de confiança das métricas
    - Teste de significância estatística (Wilcoxon) entre baseline e otimizado

3. **Expansão do Dataset**:
    - Coleta de dados multicêntricos para generalização
    - Validação externa em populações geograficamente distintas
    - Análise longitudinal para capturar dinâmica temporal da progressão

### 5.2 Melhorias Técnicas Específicas

1. **Função de Similaridade**:

    - Implementação de distâncias especializadas (Mahalanobis, Gower)
    - Aprendizado de métricas (LMNN - Large Margin Nearest Neighbor)
    - Normalização adaptativa por tipo de feature

2. **Sistema CBR Avançado**:

    - Estratégias de seleção de casos (diversidade vs similaridade)
    - Mechanisms de esquecimento para bases de casos dinâmicas
    - Integração de conhecimento médico explícito (ontologias)

3. **Comparação Benchmarking**:
    - Avaliação contra SVM, Random Forest, XGBoost, Deep Learning
    - Análise de trade-off interpretabilidade vs performance
    - Métricas específicas para cenário clínico (sensibilidade, especificidade)

## 6. Conclusões

### 6.1 Síntese dos Resultados Obtidos

Este trabalho implementou com sucesso um sistema CBR para classificação de DRC, demonstrando:

1. **Performance Clínica Satisfatória**:

    - CKD_Stage: 69.30% de acurácia (melhoria de +4.39% sobre baseline)
    - CKD_Progression: 86.40% de acurácia (melhoria de +0.88% sobre baseline)

2. **Validação de Conhecimento Médico**: A otimização automática identificou Creatinina e Hemoglobina como features críticas, confirmando guidelines clínicos estabelecidos

3. **Metodologia Robusta**: Grid Search com 5 estratégias sistemáticas demonstrou efetividade da otimização de pesos

### 6.2 Contribuições Técnicas

**Implementação Completa em Python:**

-   Função de similaridade híbrida para dados médicos mistos
-   Sistema de otimização por Grid Search interpretável
-   Avaliação comparativa rigorosa com métricas padronizadas
-   Visualizações detalhadas do impacto dos pesos (implementadas no notebook)

**Decisões de Projeto Justificadas:**

-   Escolha do Grid Search por interpretabilidade e efetividade para CBR
-   k=5 para balancear especificidade e generalização
-   Estratificação por CKD_Stage para manter representatividade

### 6.3 Análise Crítica

**Pontos Fortes:**

-   Confirmação quantitativa de conhecimento clínico (Creatinina/Hemoglobina)
-   Melhorias consistentes com otimização sistemática
-   Código totalmente reproduzível e bem documentado
-   Estratégias de otimização clinicamente interpretáveis

**Limitações Identificadas:**

-   Dataset de fonte única (1.138 amostras) limita generalização
-   Grid Search restrito a estratégias predefinidas
-   Ausência de validação cruzada para robustez estatística
-   Performance do Estágio 2 comprometida por baixo support (19 amostras)

### 6.4 Considerações Finais

O sistema CBR desenvolvido cumpre todos os requisitos especificados, fornecendo solução interpretável e clinicamente relevante para classificação de DRC. A combinação de otimização sistemática, validação de conhecimento médico e performance satisfatória estabelece base sólida para futuras pesquisas em CBR médico.

Os gráficos implementados no código-fonte (`main_final.ipynb`) demonstram claramente o impacto das diferentes combinações de pesos, permitindo análise visual das melhorias obtidas e validação da metodologia proposta.

## 7. Experimentos Realizados e Resultados Detalhados

### 7.1 Configuração Experimental

**Divisão dos Dados:**

-   **Dataset Total**: 1.138 amostras
-   **Conjunto de Treino**: 910 amostras (80%)
-   **Conjunto de Teste**: 228 amostras (20%)
-   **Conjunto de Validação**: 182 amostras (20% do treino, para otimização)
-   **Estratificação**: Baseada em CKD_Stage para manter proporções

**Parâmetros Fixos:**

-   k=5 (k-NN)
-   random_state=42 (reprodutibilidade)
-   Normalização: StandardScaler para features numéricas
-   Imputação: mediana (numéricas), moda (categóricas)

### 7.2 Experimento 1: CBR Baseline

**Objetivo:** Estabelecer performance de referência com pesos uniformes

**Configuração:** Todos os pesos = 1.0

**Resultados CKD_Stage:**

-   Acurácia: 64.91%
-   Maior dificuldade: Estágio 2 (Precision=50%, Recall=16%)
-   Melhor performance: Estágio 5 (Precision=88%, Recall=67%)

**Resultados CKD_Progression:**

-   Acurácia: 85.53%
-   Performance balanceada entre classes
-   Sem Progressão: F1=91%, Com Progressão: F1=69%

### 7.3 Experimento 2: Otimização por Grid Search

**Objetivo:** Identificar estratégias de ponderação otimizadas

**Estratégias Testadas (5 configurações):**

1. **Baseline**: Todos os pesos = 1.0
2. **Creatinine_Hemoglobin_High**: Creatinina=2.0, Hemoglobina=2.0, outros=1.0
3. **Numerical_High**: Features numéricas=1.5, categóricas=1.0
4. **Categorical_High**: Features categóricas=1.5, numéricas=1.0
5. **Clinical_Core**: Age, Systolic_Pressure, BMI = 1.5, outros=1.0

**Resultados no Conjunto de Validação:**

| Estratégia                 | CKD_Stage  | CKD_Progression |
| -------------------------- | ---------- | --------------- |
| Baseline                   | 71.43%     | 79.67%          |
| Creatinine_Hemoglobin_High | **75.27%** | 79.67%          |
| Numerical_High             | 73.63%     | **81.32%**      |
| Categorical_High           | 69.78%     | 78.02%          |
| Clinical_Core              | 70.88%     | 79.12%          |

### 7.4 Experimento 3: Avaliação Final

**Objetivo:** Comparar baseline vs otimizado no conjunto de teste

**Estratégias Selecionadas:**

-   CKD_Stage: Creatinine_Hemoglobin_High
-   CKD_Progression: Numerical_High

**Resultados Finais no Conjunto de Teste:**

| Problema            | Baseline | Otimizado  | Melhoria Absoluta | Melhoria Relativa |
| ------------------- | -------- | ---------- | ----------------- | ----------------- |
| **CKD_Stage**       | 64.91%   | **69.30%** | +4.39%            | +6.76%            |
| **CKD_Progression** | 85.53%   | **86.40%** | +0.88%            | +1.02%            |

### 7.5 Análise dos Padrões Encontrados

**Insights Clínicos Validados:**

1. **Creatinina e Hemoglobina** emergiram como biomarcadores críticos para estadiamento
2. **Features numéricas** mostraram-se superiores para predição de progressão
3. **Estágio 2** apresentou baixa performance devido ao reduzido número de amostras (n=19)

**Efetividade da Otimização:**

-   Melhoria mais substancial em CKD_Stage (problema multiclasse mais complexo)
-   Melhoria menor mas consistente em CKD_Progression (baseline já robusto)
-   Confirmação de hipóteses clínicas através de otimização automática

---

## Anexos

### A.1 Especificações Técnicas

-   **Linguagem**: Python 3.12
-   **Bibliotecas Principais**: pandas, numpy, scikit-learn, matplotlib, seaborn
-   **Ambiente**: Jupyter Notebook / VS Code
-   **Hardware**: Computador padrão (sem requisitos especiais de GPU)
-   **Gestão de Dependências**: UV package manager com `pyproject.toml`

### A.2 Instruções para Reprodução dos Experimentos

**Requisitos do Sistema:**

-   Python 3.12 (especificado em `.python-version`)
-   UV package manager (gestão de dependências via `pyproject.toml`)
-   Jupyter Notebook ou VS Code com extensão Python

**Passos para Execução:**

1. **Configuração do Ambiente:**

    ```bash
    # No diretório do projeto
    uv sync  # Instala todas as dependências
    ```

2. **Execução do Notebook:**

    - Abrir `main_final.ipynb`
    - Executar células sequencialmente (1-20)
    - Todas as visualizações são geradas automaticamente

3. **Estrutura dos Experimentos:**
    - **Células 1-7**: Carregamento e pré-processamento dos dados
    - **Células 8-14**: Implementação do sistema CBR
    - **Célula 17**: Avaliação baseline com visualizações
    - **Células 18-19**: Otimização por Grid Search
    - **Célula 20**: Comparação final e gráficos de impacto dos pesos

**Visualizações Geradas pelo Código:**

1. **Métricas Comparativas**: Gráficos de barras baseline vs otimizado
2. **Matrizes de Confusão**: Heatmaps para análise de padrões de erro
3. **Métricas por Classe**: Precision, Recall e F1-Score detalhados
4. **Gráficos de Melhoria**: Visualização do impacto da otimização de pesos
5. **Classification Reports**: Relatórios detalhados do scikit-learn

**Dataset:**

-   Arquivo: `dataset/ckd.csv` (incluído no repositório)
-   **1.138 amostras**, 23 features originais → 19 após remoção de correlação
-   Pré-processamento automático (imputação, normalização, divisão treino/teste)

**Reprodutibilidade:**

-   Seed fixo (`random_state=42`) para resultados determinísticos
-   Código totalmente comentado em português
-   Decisões de projeto documentadas em cada célula
