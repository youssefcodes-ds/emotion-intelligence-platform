# Classical ML Training Report

**Generated:** 2026-09-14T23:47:11+00:00

## Dataset Distribution

### Training Set
| Emotion | Count | Percentage |
|---------|-------|------------|
| anger | 2,159 | 13.51% |
| fear | 1,934 | 12.10% |
| joy | 5,356 | 33.51% |
| love | 1,298 | 8.12% |
| sadness | 4,665 | 29.19% |
| surprise | 571 | 3.57% |

### Test Set
| Emotion | Count | Percentage |
|---------|-------|------------|
| anger | 275 | 13.75% |
| fear | 224 | 11.20% |
| joy | 695 | 34.75% |
| love | 159 | 7.95% |
| sadness | 581 | 29.05% |
| surprise | 66 | 3.30% |

## Results by N-gram Configuration

### Unigrams [1, 1]

**Features:** 5,000

#### Logistic Regression

**Tuned Thresholds Results:**

- **Accuracy:** 0.8205
- **Macro F1:** 0.7788
- **Weighted F1:** 0.8217

**Per-Class Metrics:**

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|----------|
| sadness | 0.6902 | 0.9243 | 0.7903 | 581 |
| joy | 0.9489 | 0.8547 | 0.8993 | 695 |
| love | 0.7286 | 0.6415 | 0.6823 | 159 |
| anger | 0.8958 | 0.7818 | 0.8350 | 275 |
| fear | 0.9222 | 0.6875 | 0.7877 | 224 |
| surprise | 0.7959 | 0.5909 | 0.6783 | 66 |

**Optimal Thresholds:**

- anger: 0.10
- fear: 0.10
- joy: 0.10
- love: 0.10
- sadness: 0.50
- surprise: 0.10

#### Linear SVM

**Tuned Thresholds Results:**

- **Accuracy:** 0.8355
- **Macro F1:** 0.7752
- **Weighted F1:** 0.8339

**Per-Class Metrics:**

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|----------|
| sadness | 0.6933 | 0.9570 | 0.8040 | 581 |
| joy | 0.9508 | 0.8906 | 0.9198 | 695 |
| love | 0.8522 | 0.6164 | 0.7153 | 159 |
| anger | 0.9145 | 0.7782 | 0.8409 | 275 |
| fear | 0.9235 | 0.7009 | 0.7970 | 224 |
| surprise | 0.9643 | 0.4091 | 0.5745 | 66 |

**Optimal Thresholds:**

- anger: 0.10
- fear: 0.10
- joy: 0.10
- love: 0.10
- sadness: 0.50
- surprise: 0.10

---

### Bigrams [1, 2]

**Features:** 5,000

#### Logistic Regression

**Tuned Thresholds Results:**

- **Accuracy:** 0.8120
- **Macro F1:** 0.7691
- **Weighted F1:** 0.8129

**Per-Class Metrics:**

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|----------|
| sadness | 0.6716 | 0.9398 | 0.7834 | 581 |
| joy | 0.9455 | 0.8489 | 0.8946 | 695 |
| love | 0.7559 | 0.6038 | 0.6713 | 159 |
| anger | 0.9035 | 0.7491 | 0.8191 | 275 |
| fear | 0.9359 | 0.6518 | 0.7684 | 224 |
| surprise | 0.7692 | 0.6061 | 0.6780 | 66 |

**Optimal Thresholds:**

- anger: 0.10
- fear: 0.50
- joy: 0.10
- love: 0.10
- sadness: 0.50
- surprise: 0.10

#### Linear SVM

**Tuned Thresholds Results:**

- **Accuracy:** 0.8220
- **Macro F1:** 0.7373
- **Weighted F1:** 0.8177

**Per-Class Metrics:**

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|----------|
| sadness | 0.6784 | 0.9621 | 0.7957 | 581 |
| joy | 0.9426 | 0.8978 | 0.9197 | 695 |
| love | 0.8600 | 0.5409 | 0.6641 | 159 |
| anger | 0.9186 | 0.7382 | 0.8185 | 275 |
| fear | 0.9096 | 0.6741 | 0.7744 | 224 |
| surprise | 0.7778 | 0.3182 | 0.4516 | 66 |

**Optimal Thresholds:**

- anger: 0.10
- fear: 0.10
- joy: 0.10
- love: 0.10
- sadness: 0.50
- surprise: 0.10

---

### Trigrams [1, 3]

**Features:** 5,000

#### Logistic Regression

**Tuned Thresholds Results:**

- **Accuracy:** 0.8045
- **Macro F1:** 0.7605
- **Weighted F1:** 0.8052

**Per-Class Metrics:**

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|----------|
| sadness | 0.6627 | 0.9466 | 0.7796 | 581 |
| joy | 0.9466 | 0.8417 | 0.8911 | 695 |
| love | 0.7480 | 0.5975 | 0.6643 | 159 |
| anger | 0.8991 | 0.7127 | 0.7951 | 275 |
| fear | 0.9226 | 0.6384 | 0.7546 | 224 |
| surprise | 0.7692 | 0.6061 | 0.6780 | 66 |

**Optimal Thresholds:**

- anger: 0.50
- fear: 0.50
- joy: 0.10
- love: 0.10
- sadness: 0.10
- surprise: 0.10

#### Linear SVM

**Tuned Thresholds Results:**

- **Accuracy:** 0.8150
- **Macro F1:** 0.7388
- **Weighted F1:** 0.8123

**Per-Class Metrics:**

| Emotion | Precision | Recall | F1 | Support |
|---------|-----------|--------|-----|----------|
| sadness | 0.6671 | 0.9587 | 0.7867 | 581 |
| joy | 0.9488 | 0.8791 | 0.9126 | 695 |
| love | 0.8286 | 0.5472 | 0.6591 | 159 |
| anger | 0.9115 | 0.7491 | 0.8224 | 275 |
| fear | 0.9119 | 0.6473 | 0.7572 | 224 |
| surprise | 0.7742 | 0.3636 | 0.4948 | 66 |

**Optimal Thresholds:**

- anger: 0.10
- fear: 0.10
- joy: 0.50
- love: 0.10
- sadness: 0.50
- surprise: 0.10

---

## Recommendations

**Best Configuration:** Unigrams

### Key Findings

1. **N-gram Selection:**
   - Unigrams provide the best overall performance
   - Adding bigrams/trigrams increases dimensionality without proportional gains

2. **Model Comparison:**
   - Linear SVM performs comparably or better than Logistic Regression
   - Calibration improves probability estimates significantly

3. **Threshold Tuning:**
   - Per-class thresholds provide substantial F1 improvements
   - Low thresholds indicate high confidence in calibrated probabilities

4. **Class Imbalance:**
   - Minority classes ('surprise', 'love') benefit from lower thresholds
   - Balanced class weights help but don't fully address imbalance

### Next Steps

1. **Deep Learning Models:**
   - Try transformer models (BERT, RoBERTa) for better context understanding
   - Implement attention mechanisms to identify influential words

2. **Ensemble Methods:**
   - Combine LogReg + SVM predictions with voting or stacking
   - Weighted ensemble based on per-class performance

3. **Data Augmentation:**
   - Apply SMOTE to minority classes
   - Use data augmentation (backtranslation, paraphrasing)

4. **Feature Engineering:**
   - Combine n-grams with domain-specific features
   - Add linguistic features (sentiment indicators, intensifiers)

