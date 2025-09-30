# Sistema CBR para Classificação de Doença Renal Crônica - Relatório Simplificado

**Alunos:** Leonardo Petta do Nascimento, João Pedro Barreto de Melo  
**Disciplina:** Machine Learning - Mestrado UFSM  
**Professor:** Luis Alvaro Silva  
**Data:** Setembro 2025

## Como rodar o código

Eu recomendo a utilização do ambiente virtual UV (https://uv.readthedocs.io/en/latest/) para garantir a reprodutibilidade do ambiente. O projeto já contém o arquivo `pyproject.toml` com todas as dependências necessárias.

1. Instale o UV seguindo os passos da [documentação oficial](https://docs.astral.sh/uv/getting-started/installation).

2. Instale as dependências do projeto:

    ```bash
    uv sync
    ```

3. Inicie o Jupyter Notebook:
    ```bash
    uv run jupyter notebook
    ```
4. No navegador que vai se abrir, abra o arquivo `main.ipynb`.

5. Selecione o kernel `Python 3.12` para garantir que está usando o ambiente correto e possa executar o código sem problemas.

6. Execute as células do notebook na ordem, de cima para baixo ou execute todas de uma vez.

**Anexo, vou enviar um vídeo curto mostrando como fazer isso.**

## Objetivo do Trabalho

Desenvolver um sistema de **Case-Based Reasoning (CBR)** para classificar dois problemas relacionados à Doença Renal Crônica:

1. **CKD_Stage**: Classificação multiclasse dos estágios da DRC (estágios 2-5)
2. **CKD_Progression**: Classificação binária da progressão da doença (sim/não)

O dataset contém **1.138 pacientes** com **23 features clínicas** incluindo dados demográficos, exames laboratoriais e histórico médico.

## Bloco 1: Carregamento e Análise Inicial dos Dados

### O que foi feito:

-   Carregamento do dataset `ckd.csv`
-   Identificação dos tipos de features (numéricas vs categóricas)
-   Análise de valores ausentes
-   Análise da distribuição das classes target

### Por que essa abordagem:

-   **Separação automática** de features numéricas/categóricas baseada em critérios objetivos e tipagem do Python e pandas usando dtype e número de valores únicos
-   **Análise exploratória** foi feita por ser essencial para entender a qualidade e distribuição dos dados antes da modelagem

### Principais descobertas:

-   **7 features numéricas**, **16 categóricas**
-   Valores ausentes detectados principalmente em **BMI**
-   **CKD_Stage**: Distribuição relativamente equilibrada entre estágios
-   **CKD_Progression**: 75% sem progressão, 25% com progressão

## Bloco 2: Pré-processamento dos Dados

### O que foi feito:

-   **Remoção de features com correlação > 90%** com os targets
-   **Tratamento de valores ausentes** fizemos imputação por mediana para os dados numéricos, moda para os categóricos
-   **Normalização** das features numéricas usando StandardScaler
-   **Divisão treino/teste** (80%/20%) com estratificação

### Por que essas escolhas:

#### Remoção de correlação > 90%:

-   **eGFR** (91.9% correlação) e **CKD_Risk** (96.6% correlação) foram removidas pra cumprir requisito do trabalho e evitar vazamento de informação (features muito correlacionadas com o target podem "entregar" a resposta)

#### Imputação de valores ausentes:

-   **Mediana** para numéricas: Vimos que é uma boa prática para dados numéricos com distribuição assimétrica
-   **Moda** para categóricas: Mantém a distribuição original e pegamos a categoria que mais aparece.

#### Normalização usando StandardScaler:

-   **Essencial para CBR**: Garante que todas as features numéricas contribuam igualmente para o cálculo de distância
-   **Média = 0, Desvio = 1**: Padronização necessária para distância euclidiana

#### Estratificação:

-   **Baseada em CKD_Stage**: Mantém proporções das classes nos conjuntos de treino e teste e assim evitamos desbalanceamento.

### Resultado:

-   Dataset final: **19 features** (de 23 originais)
-   **910 amostras de treino**, **228 de teste**

## Bloco 3: Implementação do Sistema CBR

### O que foi feito:

-   **Função de similaridade híbrida** para dados mistos (numéricos + categóricos)
-   **Algoritmo k-NN customizado** para recuperação de k casos similares
-   **Classe CBRClassifier** criamos uma classe para realizar os passos de um sistema CBR e ser compatível com scikit-learn

### Por que essas escolhas:

#### Função de Similaridade Híbrida:

Fizemos uma função de similaridade que dependendo do tipo de dado, realiza o cálculo adequado:

-   Features numéricas: Distância euclidiana
-   Features categóricas: Match/mismatch (0 ou 1)
-   Normalização pelos pesos aplicados

**Motivo**: Os dados médicos que temos no dataset são mistos, precisamos tratar cada tipo adequadamente

-   **Numéricos**: Distância euclidiana captura diferenças graduais como por exemplo idade, pressão e etc.
-   **Categóricos**: Match/mismatch simples é mas efetivo para sexo, presença de diabetes e etc.

#### k=5 no k-NN:

Escolhemos o k=5 baseado em:

-   **Balanceamento**: Nem muito específico (k pequeno) nem muito genérico (k grande)
-   **Interpretabilidade clínica**: 5 casos similares é um número compreensível.

#### Votação Majoritária:

-   **Simplicidade**: A classe mais frequente entre os k vizinhos vence
-   **Robustez**: Reduz impacto de casos outliers individuais

## Bloco 4: Avaliação CBR Baseline (primeira versão)

### O que foi feito:

-   Teste do CBR com **pesos iguais** (w=1.0 para todas as features)
-   Cálculo de métricas detalhadas para ambos os problemas (logo abaixo)
-   **Visualizações** das métricas e matrizes de confusão

### Por que começamos com baseline:

-   **Referência obrigatória**: Estabelece performance mínima antes da otimização
-   **Comparação justa**: Permite medir o real impacto da otimização de pesos

### Resultados Baseline:

**CKD_Stage** (MULTICLASS):

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Estágio 2        | 0.50      | 0.16   | 0.24     | 19      |
| Estágio 3        | 0.63      | 0.78   | 0.70     | 94      |
| Estágio 4        | 0.59      | 0.60   | 0.60     | 73      |
| Estágio 5        | 0.88      | 0.67   | 0.76     | 42      |
| **accuracy**     | –         | –      | **0.65** | 228     |
| **macro avg**    | 0.65      | 0.55   | 0.57     | 228     |
| **weighted avg** | 0.65      | 0.65   | 0.64     | 228     |

**CKD_Progression** (BINÁRIO):

| Classe           | Precision | Recall | F1-Score | Support |
| ---------------- | --------- | ------ | -------- | ------- |
| Sem Progressão   | 0.89      | 0.92   | 0.91     | 171     |
| Com Progressão   | 0.74      | 0.65   | 0.69     | 57      |
| **Accuracy**     | –         | –      | **0.86** | 228     |
| **Macro avg**    | 0.81      | 0.79   | 0.80     | 228     |
| **Weighted avg** | 0.85      | 0.86   | 0.85     | 228     |

### Resumo da Performance Baseline:

| Problema            | Acurácia   | Precisão | Recall | F1-Score |
| ------------------- | ---------- | -------- | ------ | -------- |
| **CKD_Stage**       | **64.91%** | 65.0%    | 55.1%  | 57.3%    |
| **CKD_Progression** | **85.53%** | 81.4%    | 78.7%  | 79.9%    |

### Observações importantes:

-   **CKD_Progression** tem performance muito superior (problema binário mais simples)
-   **Estágio 2** tem performance baixa (apenas 19 amostras no teste, o que indica um desbalanceamento)
-   **Estágio 5** tem melhor performance (casos mais característicos)

## Bloco 5: Otimização de Pesos

### O que foi feito:

-   Aplicação da técnica de **Grid Search** com 5 estratégias predefinidas de ponderação
-   Divisão adicional treino/validação (80%/20%) para otimização
-   Seleção automática da melhor estratégia para cada problema

### Por que escolhemos o Grid Search:

-   **Interpretabilidade**: Estratégia mais simplista e caixa-branca, então é mais fácil de explicar e ser aceito na área médica
-   **Reprodutibilidade**: Resultados determinísticos, ou seja, podem ser repetidos exatamente da mesma forma e dar os mesmos resultados.
-   **Eficiência**: Para 19 features, pode ser mais prático que métodos evolucionários
-   **Controle experimental**: Testa hipóteses específicas sobre importância de features

### Estratégias Testadas:

Testamos 5 estratégias diferentes de ponderação baseada em conhecimento médico que encontramos na internet:

1. **Baseline**: Todos os pesos = 1.0
2. **Creatinine_Hemoglobin_High**: Creatinina=2.0, Hemoglobina=2.0 (biomarcadores críticos)
3. **Numerical_High**: Features numéricas=1.5 (captura variações graduais)
4. **Categorical_High**: Features categóricas=1.5 (informação binária importante)
5. **Clinical_Core**: Age, Pressão, BMI = 1.5 (parâmetros vitais básicos)

### Resultados da Otimização:

| Problema            | Melhor Estratégia              | Acurácia Validação |
| ------------------- | ------------------------------ | ------------------ |
| **CKD_Stage**       | **Creatinine_Hemoglobin_High** | **75.27%**         |
| **CKD_Progression** | **Numerical_High**             | **81.32%**         |

### Por que esses resultados fazem sentido clinicamente:

-   **Creatinina + Hemoglobina**: São biomarcadores diretos da função renal e anemia (complicação da DRC)
-   **Features Numéricas**: Para progressão, valores contínuos capturam melhor a dinâmica temporal

## Bloco 6: Avaliação Final - Baseline vs Otimizado

### O que foi feito:

-   Comparação final no conjunto de teste reservado
-   **Avaliação rigorosa** com métricas padronizadas do scikit-learn
-   **Visualizações** detalhadas do impacto da otimização

### Resultados Finais:

| Problema            | Baseline | Otimizado  | Melhoria   |
| ------------------- | -------- | ---------- | ---------- |
| **CKD_Stage**       | 64.91%   | **69.30%** | **+4.39%** |
| **CKD_Progression** | 85.53%   | **86.40%** | **+0.88%** |

**CKD_STAGE OTIMIZADO**:

| Classe    | Precision | Recall | F1-Score | Support |
| --------- | --------- | ------ | -------- | ------- |
| Estágio 2 | 0.57      | 0.21   | 0.31     | 19      |
| Estágio 3 | 0.67      | 0.81   | 0.73     | 94      |
| Estágio 4 | 0.65      | 0.66   | 0.65     | 73      |
| Estágio 5 | 0.88      | 0.71   | 0.79     | 42      |
| accuracy  | -         | -      | **0.69** | 228     |
| macro avg | 0.69      | 0.60   | 0.62     | 228     |

**CKD_PROGRESSION OTIMIZADO**:

| Classe         | Precision | Recall | F1-Score | Support |
| -------------- | --------- | ------ | -------- | ------- |
| Sem Progressão | 0.89      | 0.94   | 0.91     | 171     |
| Com Progressão | 0.77      | 0.65   | 0.70     | 57      |
| accuracy       | –         | –      | **0.86** | 228     |
| macro avg      | 0.83      | 0.79   | 0.81     | 228     |
| weighted avg   | 0.86      | 0.86   | 0.86     | 228     |

## Discussão dos Resultados

### 1. Efetividade da Otimização

**CKD_Stage (Multiclasse):**

-   **Melhoria substancial** de 4.39% demonstra que a otimização foi efetiva
-   **Creatinina + Hemoglobina** emergiram como features críticas, validando conhecimento médico
-   Problema mais complexo se beneficiou mais da otimização

**CKD_Progression (Binário):**

-   **Melhoria menor** (0.88%) mas consistente
-   Baseline já tinha boa performance (85.53%)
-   **Features numéricas** se mostraram mais importantes para capturar progressão

### 2. Validação de Conhecimento Clínico

A otimização automática **confirmou** biomarcadores clinicamente estabelecidos:

-   **Creatinina**: Indicador direto da função renal
-   **Hemoglobina**: Anemia é complicação comum da DRC
-   **Features numéricas**: Capturam melhor a dinâmica temporal da progressão

### 3. Limitações Identificadas

1. **Dataset único**: Ter somente 1.138 amostras pode limitar na generalização
2. **Estágio 2**: Essa categoria tem muito pouca representatividade, comprometendo a performance (apenas 19 amostras)
3. **Grid Search**: Limitado a estratégias predefinidas, ou seja, não explora todo o espaço de hiperparâmetros nem é dinâmico
4. **Validação**: Ausência de validação cruzada

### 4. Pontos Fortes

1. **Metodologia robusta**: Pré-processamento rigoroso e avaliação padronizada
2. **Interpretabilidade**: CBR permite explicar decisões via casos similares
3. **Validação médica**: Confirmação quantitativa de conhecimento clínico
4. **Reprodutibilidade**: Código completo e bem documentado

## Conclusões

### Principais Contribuições:

1. **Sistema CBR completo** para classificação de DRC com otimização sistemática
2. **Validação quantitativa** de biomarcadores clínicos estabelecidos
3. **Metodologia interpretável** adequada para aplicações médicas
4. **Performance satisfatória** para ambos os problemas de classificação

### Desempenho Final:

-   **CKD_Stage**: 69.30% de acurácia (melhoria de 6.76% sobre baseline)
-   **CKD_Progression**: 86.40% de acurácia (melhoria de 1.02% sobre baseline)

### Trabalhos Futuros:

1. **Validação externa** com datasets de outros centros médicos
2. **Algoritmos de otimização** mais sofisticados (genéticos, bayesianos)
3. **Validação cruzada** para robustez estatística
4. **Comparação** com algoritmos pré-existentes (Random Forest, XGBoost)

### Considerações Finais:

O sistema CBR desenvolvido demonstra **viabilidade técnica** e **relevância clínica** para classificação de DRC. A combinação de **interpretabilidade**, **performance satisfatória** e **validação de conhecimento médico** estabelece uma base sólida para futuras pesquisas em CBR médico.

## Especificações Técnicas

**Ambiente:** Python 3.12, UV package manager  
**Bibliotecas:** pandas, numpy, scikit-learn, matplotlib, seaborn  
**Reprodutibilidade:** `random_state=42`, código totalmente comentado  
**Hardware:** Computador padrão (sem GPU necessária)
