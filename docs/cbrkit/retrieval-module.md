# CBRkit.retrieval - Retrieval Module API Reference

Source: https://wi2trier.github.io/cbrkit/cbrkit/retrieval.html

## Overview

The retrieval module provides utility functions for building and applying CBR retrieval pipelines. It's the core module for implementing the "retrieve" phase of CBR.

## Core Classes and Functions

### build

**Most important function for basic CBR implementation.**

Creates a retriever function based on a similarity function.

**Arguments:**

-   `similarity_func`: Similarity function to compute similarity between cases
-   `multiprocessing`: Boolean/int/Pool for multiprocessing support
-   `chunksize`: Number of batches to process at a time

**Returns:** A retriever function that computes similarity between cases

**Example:**

```python
import cbrkit
retriever = cbrkit.retrieval.build(
    cbrkit.sim.attribute_value(
        attributes={
            "price": cbrkit.sim.numbers.linear(max=100000),
            "year": cbrkit.sim.numbers.linear(max=50),
            "model": cbrkit.sim.attribute_value(
                attributes={
                    "make": cbrkit.sim.generic.equality(),
                }
            ),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )
)
```

### apply_query

**Core function for CBR case retrieval.**

Applies a single query to a casebase using retriever functions.

**Arguments:**

-   `casebase`: The casebase to retrieve similar cases from
-   `query`: The query to retrieve similar cases for
-   `retrievers`: Retriever functions (can be single or list for multi-stage)

**Returns:** Retrieval result object

**Example:**

```python
import cbrkit
import polars as pl

# Load data
df = pl.read_csv("./data/cars-1k.csv")
casebase = cbrkit.loaders.polars(df)

# Build retriever
retriever = cbrkit.retrieval.build(
    cbrkit.sim.attribute_value(
        attributes={
            "price": cbrkit.sim.numbers.linear(max=100000),
            "year": cbrkit.sim.numbers.linear(max=50),
            "miles": cbrkit.sim.numbers.linear(max=1000000),
        },
        aggregator=cbrkit.sim.aggregator(pooling="mean"),
    )
)

# Apply query
result = cbrkit.retrieval.apply_query(casebase, query, retriever)
```

### apply_queries

Applies multiple queries to a casebase using retriever functions.

**Arguments:**

-   `casebase`: The casebase to retrieve similar cases from
-   `queries`: Mapping of queries to retrieve similar cases for
-   `retrievers`: Retriever functions

**Returns:** Retrieval result object

## Advanced Retrieval Functions

### dropout

Filters retrieved cases based on similarity values and limits.

**Arguments:**

-   `retriever_func`: The base retriever function (typically from `build`)
-   `limit`: Maximum number of cases to return
-   `min_similarity`: Minimum similarity threshold
-   `max_similarity`: Maximum similarity threshold

**Returns:** Filtered retriever function

**Use case:** Implementing k-NN with similarity thresholds

```python
# Get top 10 cases with similarity > 0.5
filtered_retriever = cbrkit.retrieval.dropout(
    base_retriever,
    limit=10,
    min_similarity=0.5
)
```

### combine

Combines multiple retriever functions into one.

**Arguments:**

-   `retriever_funcs`: List of retriever functions to combine
-   `aggregator`: Function to aggregate results from retrievers
-   `strategy`: "intersection" or "union" for combining results
-   `default_sim`: Default similarity for union strategy

**Returns:** Combined retriever function

### distribute

Distributes retrieval process across multiple processes.

**Arguments:**

-   `retriever_func`: The retriever function to distribute
-   `multiprocessing`: Boolean/int/Pool for multiprocessing

**Returns:** Distributed retriever function

### transpose

Transforms a retriever function from one type to another.

**Arguments:**

-   `retriever_func`: The retriever function to transform
-   `conversion_func`: Function to convert input values

**Returns:** Transformed retriever function

## Result Classes

### Result

Main result object returned by retrieval operations.

**Key Attributes:**

-   `steps`: List of ResultStep objects (for multi-stage retrieval)
-   `duration`: Time taken for retrieval
-   `first_step`: First retrieval step
-   `final_step`: Final retrieval step
-   `metadata`: Additional metadata
-   `queries`: Mapping of query results
-   `default_query`: Default query result
-   `similarities`: Dictionary of case similarities **← Important for CBR**
-   `ranking`: List of case IDs sorted by similarity **← Important for CBR**
-   `casebase`: Filtered casebase with retrieved cases **← Important for CBR**

### QueryResultStep

Result for a single query retrieval.

**Key Attributes:**

-   `similarities`: Dictionary mapping case IDs to similarity scores
-   `ranking`: List of case IDs sorted by similarity (most similar first)
-   `casebase`: Casebase containing only retrieved cases
-   `query`: The original query
-   `duration`: Time taken
-   `casebase_similarities`: Mapping of case ID to (case, similarity) tuples

## Specialized Retrievers

### bm25

BM25 retriever for text-based similarity.

**Arguments:**

-   `language`: Language for text processing
-   `stopwords`: List of stopwords to ignore
-   `auto_index`: Whether to automatically build index

### sentence_transformers

Semantic similarity using sentence transformers.

**Arguments:**

-   `model`: Name of sentence transformer model
-   `query_chunk_size`: Chunk size for query processing
-   `corpus_chunk_size`: Chunk size for corpus processing
-   `device`: Device to run on (CPU/GPU)

### cohere

Semantic similarity using Cohere's rerank models.

**Arguments:**

-   `model`: Name of Cohere rerank model
-   `max_tokens_per_doc`: Maximum tokens per document

### voyageai

Semantic similarity using Voyage AI's rerank models.

**Arguments:**

-   `model`: Name of Voyage AI rerank model
-   `truncation`: Whether to truncate long texts

## Key Patterns for Medical CBR

### 1. Basic CBR Pipeline

```python
# Step 1: Build similarity function
similarity = cbrkit.sim.attribute_value(
    attributes={
        "Age": cbrkit.sim.numbers.linear(max_distance=50),
        "BMI": cbrkit.sim.numbers.exponential(alpha=0.1),
        "Sex": cbrkit.sim.generic.equality(),
        # ... other medical features
    },
    aggregator=cbrkit.sim.aggregator(pooling="mean")  # Baseline
)

# Step 2: Build retriever
retriever = cbrkit.retrieval.build(similarity)

# Step 3: Apply to query
result = cbrkit.retrieval.apply_query(casebase, patient_query, retriever)

# Step 4: Get top-k cases for classification
top_k_cases = result.casebase[:k]
similarities = result.similarities
ranking = result.ranking
```

### 2. Multi-Stage Retrieval (MAC/FAC Pattern)

```python
# Stage 1: Fast pre-filter
fast_retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.build(simple_similarity),
    limit=100,
    min_similarity=0.3
)

# Stage 2: Detailed similarity
detailed_retriever = cbrkit.retrieval.dropout(
    cbrkit.retrieval.build(detailed_similarity),
    limit=10
)

# Apply both stages
result = cbrkit.retrieval.apply_query(
    casebase,
    query,
    (fast_retriever, detailed_retriever)
)

# Access results
final_result = result.final_step  # Detailed similarity results
pre_filter_result = result.steps[0]  # Fast pre-filter results
```

### 3. Optimized Weights Integration

```python
# Baseline retriever (equal weights)
baseline_similarity = cbrkit.sim.attribute_value(
    attributes=feature_similarities,
    aggregator=cbrkit.sim.aggregator(pooling="mean")
)
baseline_retriever = cbrkit.retrieval.build(baseline_similarity)

# Optimized retriever (Grid Search weights)
optimized_weights = {"Age": 0.8, "Creatinine": 2.0, "BMI": 1.2}
optimized_similarity = cbrkit.sim.attribute_value(
    attributes=feature_similarities,
    aggregator=cbrkit.sim.aggregator(
        pooling="weighted_mean",
        pooling_weights=optimized_weights
    )
)
optimized_retriever = cbrkit.retrieval.build(optimized_similarity)

# Compare results
baseline_result = cbrkit.retrieval.apply_query(casebase, query, baseline_retriever)
optimized_result = cbrkit.retrieval.apply_query(casebase, query, optimized_retriever)
```

## Critical Notes for Your Project

1. **Use `build()` + `apply_query()`** - This is the standard CBR pattern
2. **`result.ranking`** - Case IDs sorted by similarity (use for k-NN voting)
3. **`result.similarities`** - Exact similarity scores for analysis
4. **`result.casebase`** - Pre-filtered cases for efficiency
5. **Multi-stage retrieval** - Pass tuple of retrievers for MAC/FAC pattern
6. **`dropout()` with `limit`** - Essential for k-NN implementation
7. **Result objects are Pydantic models** - Well-structured and typed
