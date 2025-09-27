# Copilot Instructions - DRC-CKD-CBR-Classifier

## Project Overview

This is a graduate-level Machine Learning assignment (UFSM) implementing Case-Based Reasoning (CBR) for Chronic Kidney Disease (CKD) classification. The project tackles two classification problems: multi-class CKD staging (`CKD_Stage`) and binary progression prediction (`CKD_Progression`).

## Core Architecture & Data Flow

### Dataset Structure (`dataset/ckd.csv`)

-   **23 clinical features** including demographics (Sex, Age), vitals (Systolic_Pressure, BMI), lab results (Hemoglobin, Creatinine, eGFR), and medical history
-   **Two target variables**: `CKD_Stage` (1-5, multi-class) and `CKD_Progression` (0/1, binary)
-   **1140 samples** with mixed data types (numerical, categorical, some missing values)
-   **Critical preprocessing requirement**: Remove features with >90% correlation to target labels

### CBR Implementation Requirements

1. **Similarity Function**: Handle mixed data types (numerical: Euclidean/normalized distance, categorical: match/mismatch or Jaccard, textual: string similarity)
2. **Weight Optimization**: Implement one of Grid Search, Gradient Descent, or Genetic Algorithm
3. **Case Retrieval**: Rank by similarity with configurable k-nearest neighbors
4. **Classification**: Majority voting from retrieved cases

## Development Environment

### Dependencies Management

-   **UV package manager** with `pyproject.toml` configuration
-   Python 3.12 (`.python-version` file)
-   Virtual environment in `.venv/`
-   Run `uv sync` to install dependencies
-   Add new packages with `uv add package_name`

### Project Structure

```
main.ipynb          # Primary implementation notebook
dataset/ckd.csv     # Clinical dataset
Prova.pdf          # Assignment specifications (Portuguese)
README.md          # Detailed assignment description
```

## Key Implementation Patterns

### Mandatory Preprocessing Pipeline

```python
# Always implement these steps per assignment requirements:
1. Remove features with >90% correlation to target (may differ between binary/multi-class)
2. Handle missing values (dataset has some BMI nulls)
3. Normalize/standardize numerical features
4. Split train/test for weight optimization vs final evaluation
```

### CBR Algorithm Structure

```python
# Expected workflow:
1. baseline_cbr(weights=equal_weights)  # All weights = 1
2. optimize_weights(method='grid_search|gradient_descent|genetic_algorithm')
3. optimized_cbr(weights=optimized_weights)
4. compare_performance(baseline_vs_optimized)
```

### Evaluation Requirements

-   **Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, AUC-ROC
-   **Clinical Analysis**: Discuss false positives/negatives implications
-   **Comparative Analysis**: Baseline CBR vs optimized CBR, CKD_Stage vs CKD_Progression

## Assignment-Specific Conventions

### Feature Categories (from README.md table)

-   **Demographics**: Sex, Age
-   **Clinical Measurements**: Systolic_Pressure, BMI, Hemoglobin, Albumin, Creatinine, eGFR
-   **Risk Factors**: CKD_Cause, CKD_Risk, Hypertension, Previous_CVD, Diabetes
-   **Lab Results**: Dipstick_Proteinuria, Proteinuria, Occult_Blood_in_Urine, Protein_Creatinine_Ratio, UPCR_Severity
-   **Medications**: RAAS_Inhibitor, Calcium_Channel_Blocker, Diuretics

### Critical Implementation Notes

-   **Language**: All code must be in Python
-   **Documentation**: Code must be well-commented with experiment reproduction instructions
-   **Validation**: Use train set for weight optimization, test set for final evaluation only
-   **Experimentation**: Test different parameter combinations and document behavior

### Expected Deliverables Structure

1. **EDA Section**: Descriptive statistics, missing values analysis, correlation heatmaps
2. **Preprocessing Module**: Feature correlation removal, normalization, train/test split
3. **CBR Implementation**: Similarity function, case retrieval, majority voting
4. **Optimization Module**: Chosen method with parameter justification
5. **Evaluation Module**: All required metrics with clinical interpretation

## File Workflow

-   Use `main.ipynb` as the primary development notebook
-   Implement modular functions that can be extracted if needed
-   Document all parameter choices and experimental decisions
-   Include visualizations showing weight optimization impact
