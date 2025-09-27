# CBRkit.sim - Similarity Functions API Reference

Source: https://wi2trier.github.io/cbrkit/cbrkit/sim.html

## Overview

CBRkit contains a selection of similarity measures for different data types:

-   Numbers (`cbrkit.sim.numbers`)
-   Strings (`cbrkit.sim.strings`)
-   Collections/Lists (`cbrkit.sim.collections`)
-   Generic data (`cbrkit.sim.generic`)
-   Attribute-value data (`cbrkit.sim.attribute_value`)
-   Aggregator to combine multiple local measures into global scores

## Core Classes and Functions

### transpose

Transforms a similarity function from one type to another.

**Arguments:**

-   `similarity_func`: The similarity function to be used on the converted values
-   `conversion_func`: A function that converts the input values from one type to another

**Example:**

```python
from cbrkit.sim.generic import equality
sim = transpose(
    similarity_func=equality(),
    conversion_func=lambda x: x.lower(),
)
sim([("A", "a"), ("b", "B")])
# Returns: [1.0, 1.0]
```

### cache

Caches similarity computation results for performance optimization.

**Arguments:**

-   `similarity_func`: The similarity function to cache
-   `conversion_func`: Optional conversion function for cache key generation

### combine

Combines multiple similarity functions into one.

**Arguments:**

-   `sim_funcs`: A list of similarity functions to be combined
-   `aggregator`: A function to aggregate the results from the similarity functions

**Returns:** A similarity function that combines results from multiple similarity functions

### dynamic_table

Allows importing similarity values from a table/lookup structure.

**Arguments:**

-   `entries`: Sequence[tuple[a, b, sim(a, b)]]
-   `symmetric`: If True, assumes sim(a, b) = sim(b, a)
-   `default`: Default similarity value for pairs not in table
-   `key_getter`: Function that extracts the key for lookup from input values

**Example:**

```python
from cbrkit.helpers import identity
from cbrkit.sim.generic import static
sim = dynamic_table(
    {
        ("a", "b"): static(0.5),
        ("b", "c"): static(0.7)
    },
    symmetric=True,
    default=static(0.0),
    key_getter=identity,
)
sim([("b", "a"), ("a", "c")])
# Returns: [0.5, 0.0]
```

### type_table

Creates similarity function based on type mappings.

### attribute_table

Creates similarity function based on attribute mappings.

## attribute_value - Core Function for Structured Data

**The most important class for your medical CBR project.**

Computes attribute-value similarity between two cases (perfect for medical records).

**Arguments:**

-   `attributes`: Mapping of attribute names to similarity functions for those attributes
-   `aggregator`: Function that aggregates local similarity scores into single global similarity
-   `value_getter`: Function that retrieves attribute value from a case
-   `default`: Default similarity score when computation error occurs

**Example:**

```python
equality = lambda x, y: 1.0 if x == y else 0.0
sim = attribute_value({
    "name": equality,
    "age": equality,
})
scores = sim([
    ({"name": "John", "age": 25}, {"name": "John", "age": 30}),
    ({"name": "Jane", "age": 30}, {"name": "John", "age": 30}),
])
# Returns AttributeValueSim objects with .value and .attributes
```

## aggregator - Critical for Weight Optimization

Aggregates local similarities to global similarity using specified pooling function.

**Arguments:**

-   `pooling`: The pooling function ("mean", "weighted_mean", "max", "min", etc.)
-   `pooling_weights`: Weights to apply during pooling (sequence or mapping) - **KEY FOR OPTIMIZATION**
-   `default_pooling_weight`: Default weight if similarity key not found

**Examples:**

```python
# Basic mean aggregation (baseline CBR)
agg = aggregator("mean")
agg([0.5, 0.75, 1.0])  # Returns: 0.75

# Weighted mean (optimized CBR) - THIS IS WHERE GRID SEARCH WEIGHTS GO
agg = aggregator("mean", {1: 1, 2: 1, 3: 0})
agg({1: 1, 2: 1, 3: 1})  # Returns: 1.0

# Custom weights from optimization
agg = aggregator("mean", {1: 1, 2: 1, 3: 2})  # Feature 3 gets double weight
agg({1: 1, 2: 1, 3: 1})  # Returns: 1.0
```

## AttributeValueSim - Result Structure

The result of `attribute_value` similarity computation.

**Attributes:**

-   `value`: The aggregated global similarity score (float)
-   `attributes`: Mapping of individual attribute similarities

## Key Pattern for Medical CBR

```python
# Step 1: Define similarity functions for each clinical feature
clinical_similarities = {
    "Age": cbrkit.sim.numbers.linear(max_distance=50),
    "BMI": cbrkit.sim.numbers.exponential(alpha=0.1),
    "Sex": cbrkit.sim.generic.equality(),
    "Creatinine": cbrkit.sim.numbers.linear(max_distance=5.0),
    "CKD_Cause": cbrkit.sim.generic.equality(),
    # ... other features
}

# Step 2: Baseline aggregator (equal weights)
baseline_aggregator = cbrkit.sim.aggregator(pooling="mean")

# Step 3: Baseline similarity function
baseline_sim = cbrkit.sim.attribute_value(
    attributes=clinical_similarities,
    aggregator=baseline_aggregator
)

# Step 4: Optimized aggregator (Grid Search results)
optimized_weights = {
    "Age": 0.8,
    "Creatinine": 2.0,  # More important clinically
    "BMI": 1.2,
    "Sex": 0.5
}
optimized_aggregator = cbrkit.sim.aggregator(
    pooling="weighted_mean",
    pooling_weights=optimized_weights
)

# Step 5: Optimized similarity function
optimized_sim = cbrkit.sim.attribute_value(
    attributes=clinical_similarities,
    aggregator=optimized_aggregator
)
```

## Available Pooling Functions

-   `"mean"`: Average of similarities
-   `"weighted_mean"`: Weighted average (use this for optimization)
-   `"max"`: Maximum similarity
-   `"min"`: Minimum similarity
-   `"sum"`: Sum of similarities
-   Custom pooling functions can be defined

## Sub-modules

-   `cbrkit.sim.numbers`: Numeric similarity measures
-   `cbrkit.sim.strings`: String similarity measures
-   `cbrkit.sim.collections`: Collection/sequence similarity
-   `cbrkit.sim.generic`: Generic similarity functions
-   `cbrkit.sim.graphs`: Graph similarity algorithms
-   `cbrkit.sim.embed`: Embedding-based similarity
-   `cbrkit.sim.taxonomy`: Taxonomy-based similarity
-   `cbrkit.sim.pooling`: Pooling function utilities
-   `cbrkit.sim.aggregator`: Aggregation utilities

## Critical Notes for Your Project

1. **Use `attribute_value`** - Perfect for medical records with mixed data types
2. **Use `aggregator`** - Essential for implementing weight optimization
3. **`pooling_weights`** - This is where your Grid Search results go
4. **Baseline vs Optimized** - Same similarity functions, different aggregators
5. **AttributeValueSim.value** - This is your final similarity score for CBR ranking
