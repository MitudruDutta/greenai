# Data Card: HACK4EARTH Green AI Dataset

## Dataset Summary

**Name:** HACK4EARTH Green AI Scaffold Dataset  
**Version:** 1.0  
**Source:** Kaggle Competition  
**License:** Competition use  
**Purpose:** Hackathon benchmark for Green AI principles

## Dataset Composition

### Training Data (`train.csv`)
- **Samples:** 5
- **Features:** 2 numerical (`feature_1`, `feature_2`)
- **Target:** Binary (0.0 or 1.0)
- **IDs:** TR001 - TR005

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| example_id | string | TR001-TR005 | Unique identifier |
| feature_1 | float | 0.12-0.91 | Numerical feature |
| feature_2 | float | 9-13 | Numerical feature |
| target | float | 0.0, 1.0 | Binary target |

### Test Data (`test.csv`)
- **Samples:** 3
- **Features:** None (ID only)
- **IDs:** TS001 - TS003

## Pattern Analysis

### Discovered Pattern

The target equals the parity of the numeric part of the ID (the ID number modulo 2):

`target = ID_number % 2`

- If `ID_number` is odd → `target = 1.0`
- If `ID_number` is even → `target = 0.0`

Training examples:
- `TR001` (ID=1, odd) → `1.0`
- `TR002` (ID=2, even) → `0.0`
- `TR003` (ID=3, odd) → `1.0`
- `TR004` (ID=4, even) → `0.0`
- `TR005` (ID=5, odd) → `1.0`

Correlation: 1.00 (perfect)

## Data Quality

### Completeness
✅ No missing values  
✅ No outliers  
✅ Consistent formatting  

### Appropriateness
✅ Relevant for demonstrating pattern recognition  
✅ Suitable for Green AI principles  
⚠️ Very small dataset (5 samples)  
⚠️ Deterministic pattern (not realistic)

## Bias Analysis

### Class Distribution
- **Class 0:** 2 samples (40%)
- **Class 1:** 3 samples (60%)
- **Imbalance:** Slight but not problematic

### Feature Correlations
- **ID ↔ target:** 0.00 (linear)
- **ID parity ↔ target:** 1.00 (perfect)
- **feature_1 ↔ target:** 0.00
- **feature_2 ↔ target:** 0.00

## Ethical Considerations

### Privacy
✅ No personal data  
✅ No sensitive attributes  
✅ Synthetic/scaffold data

### Fairness
✅ Deterministic rule - no learned bias  
✅ Equal treatment across all IDs

## Intended Use

### ✅ Appropriate For
- Green AI demonstrations
- Algorithmic efficiency comparisons
- Educational pattern recognition
- Hackathon benchmarking

### ❌ Not Appropriate For
- Production ML systems
- Real-world decision making
- Complex pattern learning
- Generalization studies

## Limitations

1. **Size:** Only 5 training samples
2. **Complexity:** Single deterministic pattern
3. **Realism:** Scaffold dataset, not real-world
4. **Generalization:** No test set ground truth provided

## Access & License

**Access:** [Kaggle Competition Page](https://www.kaggle.com/competitions/kaggle-community-olympiad-hack-4-earth-green-ai)  
**License:** Competition use only  
**Citation:** HACK4EARTH Green AI Hackathon 2025

---

**Version:** 1.0  
**Last Updated:** October 2025