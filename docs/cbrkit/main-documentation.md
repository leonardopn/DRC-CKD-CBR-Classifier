# CBRkit - Complete Documentation

Source: https://wi2trier.github.io/cbrkit/cbrkit.html

## Overview

CBRkit is a customizable and modular toolkit for Case-Based Reasoning (CBR) in Python. It provides a set of tools for loading cases and queries, defining similarity measures, and retrieving cases based on a query.

## Installation

The library is available on PyPI:

```bash
pip install cbrkit
# For all optional dependencies:
pip install cbrkit[all]
```

Optional dependencies:

-   `all`: All optional dependencies
-   `api`: REST API Server
-   `cli`: Command Line Interface (CLI)
-   `eval`: Evaluation tools for common metrics like precision and recall
-   `graphs`: Graph libraries like networkx and rustworkx
-   `llm`: Large Language Models (LLM) APIs like Ollama and OpenAI
-   `nlp`: Standalone NLP tools levenshtein, nltk, openai, and spacy
-   `timeseries`: Time series similarity measures like dtw and smith_waterman
-   `transformers`: Advanced NLP tools based on pytorch and transformers

## Loading Cases

### File Formats Supported

-   CSV, JSON, TOML, XML, YAML, Python objects

### Basic Loading

```python
import cbrkit
# Load from file
casebase = cbrkit.loaders.file("path/to/cases.[json,toml,yaml,xml,csv]")

# Integration with pandas/polars
import polars as pl
df = pl.read_csv("path/to/cases.csv")
casebase = cbrkit.loaders.polars(df)
```

## Defining Queries

```python
# Single query as dict
query = {"name": "John", "age": 25}

# Load multiple queries
queries = cbrkit.loaders.polars(pl.read_csv("path/to/queries.csv"))
queries = cbrkit.loaders.file("path/to/queries.[json,toml,yaml,xml,csv]")

# Extract single query from collection
query = cbrkit.helpers.singleton(queries)
```

## Similarity Measures and Aggregation

### Custom Similarity Measures

```python
def color_similarity(x: str, y: str) -> float:
    if x == y:
        return 1.0
    elif x in y or y in x:
        return 0.5
    return 0.0
```

**Important:** Parameters must be named `x` and `y`

### Built-in Similarity Measures

#### String Similarity

```python
# Edit distance measures
levenshtein_sim = cbrkit.sim.strings.levenshtein()
jaro_sim = cbrkit.sim.strings.jaro()

# Exact matching
equality_sim = cbrkit.sim.generic.equality()
```

#### Number Similarity

```python
# Linear similarity with optional thresholds
linear_sim = cbrkit.sim.numbers.linear(max_distance=100)

# Exponential decay similarity
exp_sim = cbrkit.sim.numbers.exponential(alpha=0.1)

# Step functions
threshold_sim = cbrkit.sim.numbers.threshold(threshold=50)
```

#### Embedding-Based Similarity

```python
# Build a similarity function with embedding and scorer
embed_sim = cbrkit.sim.embed.build(
    conversion_func=cbrkit.sim.embed.sentence_transformers(
        model="all-MiniLM-L6-v2"
    ),
    sim_func=cbrkit.sim.embed.cosine()  # or dot(), angular(), euclidean(), manhattan()
)

# Using OpenAI embeddings
openai_sim = cbrkit.sim.embed.build(
    conversion_func=cbrkit.sim.embed.openai(
        model="text-embedding-3-small"
    ),
    sim_func=cbrkit.sim.embed.cosine()
)

# Caching embeddings for performance
cached_embed_func = cbrkit.sim.embed.cache(
    func=cbrkit.sim.embed.sentence_transformers(
        model="all-MiniLM-L6-v2"
    ),
    path="embeddings_cache.npz",
    autodump=True,
    autoload=True
)

cached_sim = cbrkit.sim.embed.build(
    conversion_func=cached_embed_func,
    sim_func=cbrkit.sim.embed.cosine()
)
```

#### Utility Functions

```python
# Combining multiple similarity functions
combined_sim = cbrkit.sim.combine(
    sim_funcs=[sim1, sim2, sim3],
    aggregator=cbrkit.sim.aggregator(pooling="mean")
)

# Caching similarity results
cached_sim = cbrkit.sim.cache(base_sim_func)

# Transposing similarity functions
transposed_sim = cbrkit.sim.transpose(
    sim_func=number_sim,
    to_x=lambda s: float(s),
    to_y=lambda s: float(s)
)
```

### Global Similarity and Aggregation

#### Basic Aggregation

```python
similarities = [0.8, 0.6, 0.9]
aggregator = cbrkit.sim.aggregator(pooling="mean")
global_similarity = aggregator(similarities)
```

#### Attribute-Value Based Data

```python
global_sim = cbrkit.sim.attribute_value(
    attributes={
        "price": cbrkit.sim.numbers.linear(),
        "color": color_similarity,  # custom measure (no parentheses!)
        # ...
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)
```

#### Nested Similarity Functions

```python
nested_sim = cbrkit.sim.attribute_value(
    attributes={
        "manufacturer": cbrkit.sim.attribute_value(
            attributes={
                "name": cbrkit.sim.strings.spacy(model="en_core_web_lg"),
                "country": cbrkit.sim.strings.levenshtein(),
            },
            aggregator=cbrkit.sim.aggregator(pooling="mean"),
        ),
        "color": color_similarity,
        # ...
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean"),
)
```

## Retrieval

### Basic Retrieval

```python
# Build retriever
retriever = cbrkit.retrieval.build(
    cbrkit.sim.attribute_value(...)
)

# Apply query
result = cbrkit.retrieval.apply_query(casebase, query, retriever)
```

### Result Attributes

-   `similarities`: Dictionary containing similarity scores for each case
-   `ranking`: List of case indices sorted by similarity score
-   `casebase`: The casebase containing only retrieved cases

### Multi-Stage Retrieval (MAC/FAC Pattern)

```python
# Create retrievers
retriever1 = cbrkit.retrieval.dropout(..., min_similarity=0.5, limit=20)
retriever2 = cbrkit.retrieval.dropout(..., limit=10)

# Apply sequentially
result = cbrkit.retrieval.apply_query(casebase, query, (retriever1, retriever2))

# Result attributes:
# - final_step: Result of last retriever
# - steps: List of results for each retriever
```

## Advanced Retrieval

### BM25 Retrieval

```python
retriever = cbrkit.retrieval.bm25(
    key="text_field",  # Field to search in
    limit=10
)
result = cbrkit.retrieval.apply_query(casebase, query, retriever)
```

### Combining Multiple Retrievers

```python
retriever1 = cbrkit.retrieval.build(...)
retriever2 = cbrkit.retrieval.bm25(...)

combined = cbrkit.retrieval.combine(
    retrievers=[retriever1, retriever2],
    aggregator=cbrkit.sim.aggregator(pooling="mean")
)
result = cbrkit.retrieval.apply_query(casebase, query, combined)
```

### Distributed Processing

```python
retriever = cbrkit.retrieval.distribute(
    cbrkit.retrieval.build(...),
    batch_size=1000
)
```

## Adaptation Functions

### Custom Adaptation Functions

```python
def replace_adapter(case: str, query: str) -> str:
    return query if case != query else case
```

**Important:** Parameters must be named `case` and `query` for pair functions, or `casebase` and `query` for map/reduce functions.

### Built-in Adaptation Functions

```python
# Number aggregator
number_adapter = cbrkit.adapt.numbers.aggregate(pooling="mean")

# Attribute-value based adapter
adapter = cbrkit.adapt.attribute_value(
    attributes={
        "price": cbrkit.adapt.numbers.aggregate(),
        "color": cbrkit.adapt.strings.regex("CASE_PATTERN", "QUERY_PATTERN", "REPLACEMENT"),
        # ...
    }
)
```

## Reuse

```python
# Build reuser
reuser = cbrkit.reuse.build(
    cbrkit.adapt.attribute_value(...),
)

# Apply to retrieval result
result = cbrkit.reuse.apply_query(retrieval_result, query, reuser)
```

### Result Attributes

-   `adaptations`: Dictionary containing adapted values for each case
-   `ranking`: List of case indices matching retrieval result
-   `casebase`: The casebase containing only adapted cases

### Multiple Reuse Pipelines

```python
reuser1 = cbrkit.reuse.build(...)
reuser2 = cbrkit.reuse.build(...)
result = cbrkit.reuse.apply_query(retrieval_result, query, (reuser1, reuser2))
```

## Evaluation

### Basic Evaluation

```python
results = cbrkit.eval.compute(
    qrels,  # Ground truth relevance scores
    run,    # Retrieval similarity scores
    metrics=["precision@5", "recall@5", "f1@5"]
)
```

### Custom Metrics

```python
def custom_metric(
    qrels: Mapping[str, Mapping[str, int]],
    run: Mapping[str, Mapping[str, float]],
    k: int,
    relevance_level: int,
) -> float:
    # Custom metric logic here
    return 0.0

results = cbrkit.eval.compute(
    qrels,
    run,
    metrics=["custom_metric@5"],
    metric_funcs={"custom_metric": custom_metric},
)
```

### Built-in Metrics

Standard IR metrics: `precision`, `recall`, `f1`
CBRkit custom metrics:

-   `correctness`: Measures ranking relevance ordering (-1 to 1)
-   `completeness`: Measures fraction of relevance pairs preserved (0 to 1)

```python
# Generate metrics at different k values
metrics = cbrkit.eval.metrics_at_k(["precision", "recall", "f1"], [1, 5, 10])
```

## Logging

```python
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)

logging.getLogger("cbrkit.sim.XXX").setLevel(logging.DEBUG)
logging.getLogger("cbrkit.retrieval").setLevel(logging.DEBUG)
```

## Modules Overview

-   `cbrkit.loaders` and `cbrkit.dumpers`: Load/export cases and queries
-   `cbrkit.sim`: Similarity functions for common data types
    -   `cbrkit.sim.strings`: String similarity measures
    -   `cbrkit.sim.numbers`: Numeric similarity measures
    -   `cbrkit.sim.collections`: Collection/sequence similarity
    -   `cbrkit.sim.embed`: Embedding-based similarity
    -   `cbrkit.sim.graphs`: Graph similarity algorithms
    -   `cbrkit.sim.taxonomy`: Taxonomy-based similarity
    -   `cbrkit.sim.generic`: Generic similarity functions
    -   `cbrkit.sim.attribute_value`: Attribute-value similarity
    -   `cbrkit.sim.aggregator`: Combine multiple measures
-   `cbrkit.retrieval`: Retrieval pipelines
-   `cbrkit.adapt`: Adaptation functions
-   `cbrkit.reuse`: Reuse pipelines
-   `cbrkit.eval`: Evaluation metrics
-   `cbrkit.model`: Data models for graphs and results
-   `cbrkit.cycle`: CBR cycle implementation
-   `cbrkit.typing`: Generic type definitions
-   `cbrkit.synthesis`: LLM integration for insights generation

## Key Patterns for Your Project

### 1. Loading Medical Data

```python
import pandas as pd
import cbrkit

# Load your CKD dataset
df = pd.read_csv('dataset/ckd.csv')
casebase = cbrkit.loaders.polars(df)
```

### 2. Medical Feature Similarity

```python
# For clinical data
medical_sim = cbrkit.sim.attribute_value(
    attributes={
        "Age": cbrkit.sim.numbers.linear(max_distance=50),
        "BMI": cbrkit.sim.numbers.exponential(alpha=0.1),
        "Sex": cbrkit.sim.generic.equality(),
        "Creatinine": cbrkit.sim.numbers.linear(max_distance=5.0),
        # ... other clinical features
    },
    aggregator=cbrkit.sim.aggregator(pooling="weighted_mean")  # Can use weights!
)
```

### 3. Weighted Aggregation (For Optimization)

```python
# Baseline: equal weights
baseline_aggregator = cbrkit.sim.aggregator(pooling="mean")

# Optimized: custom weights from Grid Search
optimized_weights = {"Age": 0.8, "Creatinine": 2.0, "BMI": 1.2}
optimized_aggregator = cbrkit.sim.aggregator(
    pooling="weighted_mean",
    weights=optimized_weights
)
```

### 4. Complete CBR Pipeline

```python
# 1. Define similarity
similarity = cbrkit.sim.attribute_value(attributes=..., aggregator=...)

# 2. Build retriever
retriever = cbrkit.retrieval.build(similarity)

# 3. Apply query (for each test case)
result = cbrkit.retrieval.apply_query(casebase, query, retriever)

# 4. Get predictions from top-k cases
top_cases = result.casebase[:k]
predictions = majority_vote(top_cases)
```
