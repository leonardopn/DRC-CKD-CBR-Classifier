# Guia Passo a Passo - Prova CBR para Classificação de DRC

## 📋 Visão Geral da Prova

**Objetivo**: Implementar algoritmos CBR (Case-Based Reasoning) para classificar Doença Renal Crônica em dois problemas:

-   **Problema 1**: Classificação multiclasse (`CKD_Stage` - estágios 1-5)
-   **Problema 2**: Classificação binária (`CKD_Progression` - sim/não)

**Dataset**: 1140 amostras com 23 features clínicas + 2 targets

---

## 🎯 Checklist de Entregáveis

-   [ ] Código Python bem comentado (main.ipynb)
-   [ ] Relatório PDF com análises e conclusões
-   [ ] Implementação CBR baseline (pesos iguais)
-   [ ] Implementação de otimização de pesos (Grid Search/Gradiente/Genético)
-   [ ] Avaliação completa com métricas clínicas
-   [ ] Comparação entre abordagens e problemas

---

## 📝 PASSO 1: Configuração do Ambiente

### 1.1 Verificar Dependências

```bash
# Já instalado: cbrkit[all]>=0.28.5
uv add pandas numpy scikit-learn matplotlib seaborn
uv add jupyter plotly scipy
```

### 1.2 Estrutura do Projeto

```
✓ main.ipynb          # Notebook principal
✓ dataset/ckd.csv     # Dataset com 1140 amostras
✓ README.md           # Especificações
✓ Prova.pdf          # Documento original
```

---

## 📊 PASSO 2: Análise Exploratória (EDA)

### 2.1 Carregamento e Inspeção Inicial

-   [ ] Carregar `dataset/ckd.csv`
-   [ ] Verificar shape: (1140, 25) - 23 features + 2 targets
-   [ ] Identificar tipos de dados (numerical, categorical)
-   [ ] Mapear features por categoria clínica

### 2.2 Análise de Qualidade dos Dados

-   [ ] **Valores faltantes**: Focar em BMI (tem nulls mencionados)
-   [ ] **Distribuições**: Histogramas para numéricas, contagens para categóricas
-   [ ] **Outliers**: Boxplots para detectar valores extremos
-   [ ] **Balance das classes**: Para ambos os targets

### 2.3 Análise de Correlação

-   [ ] **Matriz de correlação** entre features
-   [ ] **Correlação features-target** (CKD_Stage e CKD_Progression)
-   [ ] **Identificar features com >90% correlação** com targets (CRÍTICO!)

**⚠️ ATENÇÃO**: Features com correlação >90% devem ser removidas obrigatoriamente!

---

## 🔧 PASSO 3: Pré-processamento (OBRIGATÓRIO)

### 3.1 Limpeza de Dados

-   [ ] **Tratar valores faltantes** (imputação ou remoção)
-   [ ] **Remover features com correlação >90%** com targets
-   [ ] **Verificar consistência** dos dados clínicos

### 3.2 Transformação de Features

-   [ ] **Encoding categóricas**: Label encoding ou One-hot
-   [ ] **Normalização/Padronização**: MinMaxScaler ou StandardScaler
-   [ ] **Feature engineering** se necessário

### 3.3 División dos Dados

-   [ ] **Train/Test split** (ex: 80/20)
-   [ ] **Separar por problema**:
    -   Dataset para CKD_Stage (multiclass)
    -   Dataset para CKD_Progression (binary)
-   [ ] **Diferentes features** podem ser usadas para cada problema

**📍 CHECKPOINT**: Dois datasets limpos e preparados

---

## 🧠 PASSO 4: Implementação CBR Baseline

### 4.1 Função de Similaridade

-   [ ] **Design modular** para diferentes tipos:
    -   Numérica: Distância Euclidiana normalizada
    -   Categórica: Match/mismatch ou Jaccard
    -   Textual: Similaridade de strings
-   [ ] **Função combinada** com pesos ajustáveis
-   [ ] **Validação** com casos de teste

### 4.2 Algoritmo CBR Base

-   [ ] **Recuperação de casos**: k-NN baseado em similaridade
-   [ ] **Classificação**: Votação majoritária
-   [ ] **Pesos iniciais**: Todos iguais (w = 1)
-   [ ] **Implementação modular** para reutilização

### 4.3 Avaliação Baseline

-   [ ] **Teste para CKD_Stage** (multiclass)
-   [ ] **Teste para CKD_Progression** (binary)
-   [ ] **Métricas iniciais**: Accuracy, Precision, Recall, F1
-   [ ] **Documentar performance** baseline

**📍 CHECKPOINT**: CBR funcionando com pesos iguais

---

## ⚡ PASSO 5: Otimização de Pesos

### 5.1 Escolha do Método (implementar 1 dos 3)

#### Opção A: Grid Search

-   [ ] **Definir espaço de busca** para pesos
-   [ ] **Cross-validation** no conjunto de treino
-   [ ] **Busca exaustiva** em grid de parâmetros

#### Opção B: Gradiente Descendente

-   [ ] **Função objetivo**: Minimizar erro de classificação
-   [ ] **Gradiente numérico** ou analítico
-   [ ] **Learning rate** e critérios de parada

#### Opção C: Algoritmo Genético

-   [ ] **População de soluções** (conjuntos de pesos)
-   [ ] **Função fitness**: Accuracy ou F1-Score
-   [ ] **Operadores**: Seleção, crossover, mutação

### 5.2 Implementação e Tuning

-   [ ] **Justificar escolha** do método
-   [ ] **Documentar parâmetros**: iterações, população, thresholds
-   [ ] **Experimentos controlados**: Testar variações
-   [ ] **Convergência**: Monitorar otimização

### 5.3 Validação da Otimização

-   [ ] **Aplicar pesos otimizados** ao CBR
-   [ ] **Teste nos dois problemas**
-   [ ] **Comparar com baseline**

**📍 CHECKPOINT**: Pesos otimizados funcionando

---

## 📈 PASSO 6: Avaliação Completa

### 6.1 Métricas Quantitativas

-   [ ] **Para ambos problemas** (CKD_Stage e CKD_Progression):
    -   Accuracy, Precision, Recall, F1-Score
    -   Matriz de Confusão
    -   AUC-ROC (quando aplicável)
-   [ ] **Comparação**: Baseline vs Otimizado

### 6.2 Análise Clínica

-   [ ] **Interpretação médica**:
    -   Falsos positivos: Pacientes sem DRC classificados como tendo
    -   Falsos negativos: Pacientes com DRC não detectados
-   [ ] **Implicações clínicas** de cada tipo de erro
-   [ ] **Relevância das features** mais importantes

### 6.3 Análise Comparativa

-   [ ] **CBR Baseline vs CBR Otimizado**
-   [ ] **CKD_Stage vs CKD_Progression**: Qual problema é mais difícil?
-   [ ] **Impacto da otimização** por tipo de problema

**📍 CHECKPOINT**: Avaliação completa documentada

---

## 📋 PASSO 7: Experimentação e Refinamento

### 7.1 Experimentos Controlados

-   [ ] **Variação de parâmetros**:
    -   Número de casos recuperados (k)
    -   Thresholds de similaridade
    -   Parâmetros do otimizador
-   [ ] **Análise de sensibilidade**
-   [ ] **Documentar comportamento** observado

### 7.2 Visualizações

-   [ ] **Gráficos de otimização**: Convergência dos pesos
-   [ ] **Heatmaps**: Importância das features
-   [ ] **Curvas ROC**: Para análise de performance
-   [ ] **Comparações visuais**: Baseline vs Otimizado

---

## 📄 PASSO 8: Documentação e Relatório

### 8.1 Código (main.ipynb)

-   [ ] **Código bem comentado** em português
-   [ ] **Instruções de reprodução** claras
-   [ ] **Células organizadas** por seção
-   [ ] **Outputs preservados** para demonstração

### 8.2 Relatório PDF

-   [ ] **Introdução**: Problema e objetivos
-   [ ] **Metodologia**: CBR e otimização escolhida
-   [ ] **Experimentos**: Configurações e resultados
-   [ ] **Resultados**: Tabelas e gráficos de performance
-   [ ] **Análise crítica**: Interpretação dos resultados
-   [ ] **Conclusões**: Limitações e trabalhos futuros
-   [ ] **Referências**: Bibliografia utilizada

### 8.3 Estrutura Sugerida do Relatório

1. **Resumo Executivo** (1 página)
2. **Análise Exploratória** (2-3 páginas)
3. **Metodologia CBR** (2-3 páginas)
4. **Experimentos e Resultados** (3-4 páginas)
5. **Discussão Clínica** (2 páginas)
6. **Conclusões** (1 página)

---

## ⚠️ Pontos Críticos para Sucesso

### ❗ Obrigatórios (Pode reprovar se não fizer)

-   Remoção de features com correlação >90% com target
-   Implementação CBR com função de similaridade apropriada
-   Otimização de pesos com pelo menos um método
-   Avaliação completa nos dois problemas
-   Código Python bem documentado

### 🎯 Diferenciais (Nota máxima)

-   Justificativa sólida da escolha de otimização
-   Análise clínica aprofundada dos erros
-   Experimentos bem controlados e documentados
-   Visualizações claras e informativas
-   Discussão crítica das limitações

### 🚨 Armadilhas Comuns

-   Não remover features altamente correlacionadas
-   Usar dados de teste para otimização de pesos
-   Não justificar escolhas metodológicas
-   Ignorar aspectos clínicos dos resultados
-   Código mal documentado ou irreproduível

**🏁 Meta**: Prova completa e bem documentada!
