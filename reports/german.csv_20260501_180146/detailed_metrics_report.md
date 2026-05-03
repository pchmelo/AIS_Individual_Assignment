# Detailed Group Metrics Report

## Stage 4: Base Fairness ML Model

### Detailed Group Metrics: personal + status + sex

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| male : single | 140 | 0.7786 | 0.6751 | 0.7429 | 0.8214 | 0.0962 | 0.5833 |
| female : divorced/separated/married | 77 | 0.7662 | 0.7054 | 0.6753 | 0.7792 | 0.0962 | 0.5200 |
| male : married/widowed | 23 | 0.6522 | 0.4889 | 0.6522 | 0.9130 | 0.0667 | 0.8750 |
| male : divorced/separated | 10 | 0.7000 | 0.6970 | 0.4000 | 0.5000 | 0.2500 | 0.3333 |

### Detailed Group Metrics: age

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| Young | 93 | 0.7312 | 0.6951 | 0.6344 | 0.7097 | 0.1525 | 0.4706 |
| Early-Middle | 83 | 0.7831 | 0.5896 | 0.7831 | 0.9036 | 0.0615 | 0.7778 |
| Late-Career | 17 | 0.8824 | 0.7976 | 0.7647 | 0.8824 | 0.0000 | 0.5000 |
| Mid-Career | 57 | 0.7368 | 0.6677 | 0.6667 | 0.7895 | 0.1053 | 0.5789 |

### Detailed Group Metrics: personal + status + sex + age

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| male : single + Young | 38 | 0.7368 | 0.6955 | 0.6842 | 0.6842 | 0.1923 | 0.4167 |
| male : single + Early-Middle | 59 | 0.8475 | 0.6110 | 0.8475 | 0.9322 | 0.0400 | 0.7778 |
| female : divorced/separated/married + Young | 40 | 0.7000 | 0.6703 | 0.6000 | 0.7000 | 0.1667 | 0.5000 |
| male : married/widowed + Young | 14 | 0.7857 | 0.7143 | 0.6429 | 0.8571 | 0.0000 | 0.6000 |
| female : divorced/separated/married + Early-Middle | 15 | 0.9333 | 0.8800 | 0.8000 | 0.8667 | 0.0000 | 0.3333 |
| male : divorced/separated + Young | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| female : divorced/separated/married + Late-Career | 11 | 0.9091 | 0.8070 | 0.8182 | 0.9091 | 0.0000 | 0.5000 |
| male : single + Mid-Career | 37 | 0.7838 | 0.7259 | 0.6486 | 0.8108 | 0.0417 | 0.5385 |
| female : divorced/separated/married + Mid-Career | 11 | 0.6364 | 0.5417 | 0.6364 | 0.8182 | 0.1429 | 0.7500 |
| male : married/widowed + Early-Middle | 3 | 0.3333 | 0.2500 | 0.3333 | 1.0000 | 0.0000 | 1.0000 |
| male : married/widowed + Mid-Career | 6 | 0.6667 | 0.4000 | 0.8333 | 0.8333 | 0.2000 | 1.0000 |
| male : divorced/separated + Early-Middle | 6 | 0.6667 | 0.6667 | 0.3333 | 0.6667 | 0.0000 | 0.5000 |
| male : divorced/separated + Mid-Career | 3 | 0.3333 | 0.2500 | 0.6667 | 0.6667 | 0.5000 | 1.0000 |
| male : single + Late-Career | 6 | 0.8333 | 0.7778 | 0.6667 | 0.8333 | 0.0000 | 0.5000 |

## Stage 4.5: Per-Attribute Fairness ML Model

### Detailed Group Metrics: personal + status + sex

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| male : single | 140 | 0.7786 | 0.6751 | 0.7429 | 0.8214 | 0.0962 | 0.5833 |
| female : divorced/separated/married | 77 | 0.7662 | 0.7054 | 0.6753 | 0.7792 | 0.0962 | 0.5200 |
| male : married/widowed | 23 | 0.6522 | 0.4889 | 0.6522 | 0.9130 | 0.0667 | 0.8750 |
| male : divorced/separated | 10 | 0.7000 | 0.6970 | 0.4000 | 0.5000 | 0.2500 | 0.3333 |

### Detailed Group Metrics: age

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| Young | 93 | 0.7312 | 0.6951 | 0.6344 | 0.7097 | 0.1525 | 0.4706 |
| Early-Middle | 83 | 0.7831 | 0.5896 | 0.7831 | 0.9036 | 0.0615 | 0.7778 |
| Late-Career | 17 | 0.8824 | 0.7976 | 0.7647 | 0.8824 | 0.0000 | 0.5000 |
| Mid-Career | 57 | 0.7368 | 0.6677 | 0.6667 | 0.7895 | 0.1053 | 0.5789 |

## Stage 4.5: Intersectional Fairness ML Model

### Detailed Group Metrics: personal + status + sex + age

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| male : single + Young | 38 | 0.7368 | 0.6955 | 0.6842 | 0.6842 | 0.1923 | 0.4167 |
| male : single + Early-Middle | 59 | 0.8475 | 0.6110 | 0.8475 | 0.9322 | 0.0400 | 0.7778 |
| female : divorced/separated/married + Young | 40 | 0.7000 | 0.6703 | 0.6000 | 0.7000 | 0.1667 | 0.5000 |
| male : married/widowed + Young | 14 | 0.7857 | 0.7143 | 0.6429 | 0.8571 | 0.0000 | 0.6000 |
| female : divorced/separated/married + Early-Middle | 15 | 0.9333 | 0.8800 | 0.8000 | 0.8667 | 0.0000 | 0.3333 |
| male : divorced/separated + Young | 1 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| female : divorced/separated/married + Late-Career | 11 | 0.9091 | 0.8070 | 0.8182 | 0.9091 | 0.0000 | 0.5000 |
| male : single + Mid-Career | 37 | 0.7838 | 0.7259 | 0.6486 | 0.8108 | 0.0417 | 0.5385 |
| female : divorced/separated/married + Mid-Career | 11 | 0.6364 | 0.5417 | 0.6364 | 0.8182 | 0.1429 | 0.7500 |
| male : married/widowed + Early-Middle | 3 | 0.3333 | 0.2500 | 0.3333 | 1.0000 | 0.0000 | 1.0000 |
| male : married/widowed + Mid-Career | 6 | 0.6667 | 0.4000 | 0.8333 | 0.8333 | 0.2000 | 1.0000 |
| male : divorced/separated + Early-Middle | 6 | 0.6667 | 0.6667 | 0.3333 | 0.6667 | 0.0000 | 0.5000 |
| male : divorced/separated + Mid-Career | 3 | 0.3333 | 0.2500 | 0.6667 | 0.6667 | 0.5000 | 1.0000 |
| male : single + Late-Career | 6 | 0.8333 | 0.7778 | 0.6667 | 0.8333 | 0.0000 | 0.5000 |

## Stage 6: Post-Mitigation ML Model (Reweighting)

### Detailed Group Metrics: personal + status + sex

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| male : single | 140 | 0.7571 | 0.6303 | 0.7429 | 0.8429 | 0.0962 | 0.6667 |
| female : divorced/separated/married | 77 | 0.7403 | 0.6375 | 0.6753 | 0.8571 | 0.0577 | 0.6800 |
| male : married/widowed | 23 | 0.6522 | 0.4889 | 0.6522 | 0.9130 | 0.0667 | 0.8750 |
| male : divorced/separated | 10 | 0.5000 | 0.4949 | 0.4000 | 0.7000 | 0.2500 | 0.6667 |

### Detailed Group Metrics: age

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| Young | 93 | 0.6667 | 0.5999 | 0.6344 | 0.7742 | 0.1525 | 0.6471 |
| Early-Middle | 83 | 0.7952 | 0.5997 | 0.7831 | 0.9157 | 0.0462 | 0.7778 |
| Late-Career | 17 | 0.8824 | 0.7976 | 0.7647 | 0.8824 | 0.0000 | 0.5000 |
| Mid-Career | 57 | 0.7018 | 0.5875 | 0.6667 | 0.8596 | 0.0789 | 0.7368 |

### Detailed Group Metrics: personal + status + sex + age

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| male : single + Young | 38 | 0.6316 | 0.5522 | 0.6842 | 0.7368 | 0.2308 | 0.6667 |
| male : single + Early-Middle | 59 | 0.8305 | 0.5948 | 0.8475 | 0.9153 | 0.0600 | 0.7778 |
| female : divorced/separated/married + Young | 40 | 0.7000 | 0.6581 | 0.6000 | 0.7500 | 0.1250 | 0.5625 |
| male : married/widowed + Young | 14 | 0.7143 | 0.5758 | 0.6429 | 0.9286 | 0.0000 | 0.8000 |
| female : divorced/separated/married + Early-Middle | 15 | 0.8000 | 0.4444 | 0.8000 | 1.0000 | 0.0000 | 1.0000 |
| male : divorced/separated + Young | 1 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 1.0000 |
| female : divorced/separated/married + Late-Career | 11 | 0.9091 | 0.8070 | 0.8182 | 0.9091 | 0.0000 | 0.5000 |
| male : single + Mid-Career | 37 | 0.7568 | 0.6813 | 0.6486 | 0.8378 | 0.0417 | 0.6154 |
| female : divorced/separated/married + Mid-Career | 11 | 0.6364 | 0.3889 | 0.6364 | 1.0000 | 0.0000 | 1.0000 |
| male : married/widowed + Early-Middle | 3 | 0.3333 | 0.2500 | 0.3333 | 1.0000 | 0.0000 | 1.0000 |
| male : married/widowed + Mid-Career | 6 | 0.6667 | 0.4000 | 0.8333 | 0.8333 | 0.2000 | 1.0000 |
| male : divorced/separated + Early-Middle | 6 | 0.6667 | 0.6667 | 0.3333 | 0.6667 | 0.0000 | 0.5000 |
| male : divorced/separated + Mid-Career | 3 | 0.3333 | 0.2500 | 0.6667 | 0.6667 | 0.5000 | 1.0000 |
| male : single + Late-Career | 6 | 0.8333 | 0.7778 | 0.6667 | 0.8333 | 0.0000 | 0.5000 |

### Comparative Group Metrics: Before vs After (Reweighting)

#### personal + status + sex

| Group | F1 (Before) | F1 (After) | FNR (Before) | FNR (After) | FPR (Before) | FPR (After) | Sel. Rate (Before) | Sel. Rate (After) |
|-------|-------------|------------|--------------|-------------|--------------|-------------|--------------------|-------------------|
| male : single | 0.6751 | 0.6303 | 0.0962 | 0.0962 | 0.5833 | 0.6667 | 0.8214 | 0.8429 |
| female : divorced/separated/married | 0.7054 | 0.6375 | 0.0962 | 0.0577 | 0.5200 | 0.6800 | 0.7792 | 0.8571 |
| male : married/widowed | 0.4889 | 0.4889 | 0.0667 | 0.0667 | 0.8750 | 0.8750 | 0.9130 | 0.9130 |
| male : divorced/separated | 0.6970 | 0.4949 | 0.2500 | 0.2500 | 0.3333 | 0.6667 | 0.5000 | 0.7000 |

#### age

| Group | F1 (Before) | F1 (After) | FNR (Before) | FNR (After) | FPR (Before) | FPR (After) | Sel. Rate (Before) | Sel. Rate (After) |
|-------|-------------|------------|--------------|-------------|--------------|-------------|--------------------|-------------------|
| Young | 0.6951 | 0.5999 | 0.1525 | 0.1525 | 0.4706 | 0.6471 | 0.7097 | 0.7742 |
| Early-Middle | 0.5896 | 0.5997 | 0.0615 | 0.0462 | 0.7778 | 0.7778 | 0.9036 | 0.9157 |
| Late-Career | 0.7976 | 0.7976 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.8824 | 0.8824 |
| Mid-Career | 0.6677 | 0.5875 | 0.1053 | 0.0789 | 0.5789 | 0.7368 | 0.7895 | 0.8596 |

#### personal + status + sex + age

| Group | F1 (Before) | F1 (After) | FNR (Before) | FNR (After) | FPR (Before) | FPR (After) | Sel. Rate (Before) | Sel. Rate (After) |
|-------|-------------|------------|--------------|-------------|--------------|-------------|--------------------|-------------------|
| male : single + Young | 0.6955 | 0.5522 | 0.1923 | 0.2308 | 0.4167 | 0.6667 | 0.6842 | 0.7368 |
| male : single + Early-Middle | 0.6110 | 0.5948 | 0.0400 | 0.0600 | 0.7778 | 0.7778 | 0.9322 | 0.9153 |
| female : divorced/separated/married + Young | 0.6703 | 0.6581 | 0.1667 | 0.1250 | 0.5000 | 0.5625 | 0.7000 | 0.7500 |
| male : married/widowed + Young | 0.7143 | 0.5758 | 0.0000 | 0.0000 | 0.6000 | 0.8000 | 0.8571 | 0.9286 |
| female : divorced/separated/married + Early-Middle | 0.8800 | 0.4444 | 0.0000 | 0.0000 | 0.3333 | 1.0000 | 0.8667 | 1.0000 |
| male : divorced/separated + Young | 1.0000 | 0.0000 | 0 | 0 | 0.0000 | 1.0000 | 0.0000 | 1.0000 |
| female : divorced/separated/married + Late-Career | 0.8070 | 0.8070 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.9091 | 0.9091 |
| male : single + Mid-Career | 0.7259 | 0.6813 | 0.0417 | 0.0417 | 0.5385 | 0.6154 | 0.8108 | 0.8378 |
| female : divorced/separated/married + Mid-Career | 0.5417 | 0.3889 | 0.1429 | 0.0000 | 0.7500 | 1.0000 | 0.8182 | 1.0000 |
| male : married/widowed + Early-Middle | 0.2500 | 0.2500 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| male : married/widowed + Mid-Career | 0.4000 | 0.4000 | 0.2000 | 0.2000 | 1.0000 | 1.0000 | 0.8333 | 0.8333 |
| male : divorced/separated + Early-Middle | 0.6667 | 0.6667 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.6667 | 0.6667 |
| male : divorced/separated + Mid-Career | 0.2500 | 0.2500 | 0.5000 | 0.5000 | 1.0000 | 1.0000 | 0.6667 | 0.6667 |
| male : single + Late-Career | 0.7778 | 0.7778 | 0.0000 | 0.0000 | 0.5000 | 0.5000 | 0.8333 | 0.8333 |

## Stage 6: Post-Mitigation ML Model (SMOTE)

### Detailed Group Metrics: personal + status + sex

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| male : single | 163 | 0.8528 | 0.8479 | 0.6074 | 0.5706 | 0.1515 | 0.1406 |
| female : divorced/separated/married | 114 | 0.8684 | 0.8667 | 0.4737 | 0.4123 | 0.2037 | 0.0667 |
| male : divorced/separated | 26 | 0.8846 | 0.7984 | 0.2308 | 0.1154 | 0.5000 | 0.0000 |
| male : married/widowed | 47 | 0.8085 | 0.7548 | 0.3404 | 0.1915 | 0.5000 | 0.0323 |

### Detailed Group Metrics: age

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| Mid-Career | 97 | 0.8557 | 0.8511 | 0.4536 | 0.3711 | 0.2500 | 0.0566 |
| Early-Middle | 98 | 0.8980 | 0.8976 | 0.5714 | 0.4898 | 0.1607 | 0.0238 |
| Young | 117 | 0.8120 | 0.8113 | 0.5043 | 0.4359 | 0.2542 | 0.1207 |
| Late-Career | 38 | 0.8684 | 0.8661 | 0.4211 | 0.4474 | 0.1250 | 0.1364 |

### Detailed Group Metrics: personal + status + sex + age

| Group | Count | Accuracy | F1 Score | Base Rate | Selection Rate | FNR | FPR |
|-------|-------|----------|----------|-----------|----------------|-----|-----|
| male : single + Mid-Career | 53 | 0.8491 | 0.8486 | 0.5094 | 0.4340 | 0.2222 | 0.0769 |
| female : divorced/separated/married + Mid-Career | 28 | 0.8571 | 0.8542 | 0.4643 | 0.3929 | 0.2308 | 0.0667 |
| male : single + Early-Middle | 48 | 0.9375 | 0.9259 | 0.7083 | 0.6875 | 0.0588 | 0.0714 |
| male : single + Young | 44 | 0.7955 | 0.7686 | 0.7045 | 0.6364 | 0.1935 | 0.2308 |
| male : divorced/separated + Early-Middle | 6 | 0.8333 | 0.4545 | 0.1667 | 0.0000 | 1.0000 | 0.0000 |
| male : divorced/separated + Mid-Career | 7 | 0.7143 | 0.6500 | 0.4286 | 0.1429 | 0.6667 | 0.0000 |
| female : divorced/separated/married + Late-Career | 9 | 0.8889 | 0.8615 | 0.7778 | 0.6667 | 0.1429 | 0.0000 |
| male : single + Late-Career | 18 | 0.7778 | 0.7750 | 0.3889 | 0.5000 | 0.1429 | 0.2727 |
| female : divorced/separated/married + Young | 51 | 0.8431 | 0.8322 | 0.3922 | 0.3529 | 0.2500 | 0.0968 |
| male : married/widowed + Young | 18 | 0.7222 | 0.6990 | 0.4444 | 0.2778 | 0.5000 | 0.1000 |
| male : divorced/separated + Late-Career | 9 | 1.0000 | 1.0000 | 0.2222 | 0.2222 | 0.0000 | 0.0000 |
| male : married/widowed + Early-Middle | 18 | 0.7778 | 0.7231 | 0.3889 | 0.1667 | 0.5714 | 0.0000 |
| female : divorced/separated/married + Early-Middle | 26 | 0.9231 | 0.9231 | 0.5385 | 0.4615 | 0.1429 | 0.0000 |
| male : married/widowed + Mid-Career | 9 | 1.0000 | 1.0000 | 0.1111 | 0.1111 | 0.0000 | 0.0000 |
| male : divorced/separated + Young | 4 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |
| male : married/widowed + Late-Career | 2 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 0 | 0.0000 |

### Comparative Group Metrics: Before vs After (SMOTE)

#### personal + status + sex

| Group | F1 (Before) | F1 (After) | FNR (Before) | FNR (After) | FPR (Before) | FPR (After) | Sel. Rate (Before) | Sel. Rate (After) |
|-------|-------------|------------|--------------|-------------|--------------|-------------|--------------------|-------------------|
| male : single | 0.6751 | 0.8479 | 0.0962 | 0.1515 | 0.5833 | 0.1406 | 0.8214 | 0.5706 |
| female : divorced/separated/married | 0.7054 | 0.8667 | 0.0962 | 0.2037 | 0.5200 | 0.0667 | 0.7792 | 0.4123 |
| male : married/widowed | 0.4889 | 0.7548 | 0.0667 | 0.5000 | 0.8750 | 0.0323 | 0.9130 | 0.1915 |
| male : divorced/separated | 0.6970 | 0.7984 | 0.2500 | 0.5000 | 0.3333 | 0.0000 | 0.5000 | 0.1154 |

#### age

| Group | F1 (Before) | F1 (After) | FNR (Before) | FNR (After) | FPR (Before) | FPR (After) | Sel. Rate (Before) | Sel. Rate (After) |
|-------|-------------|------------|--------------|-------------|--------------|-------------|--------------------|-------------------|
| Young | 0.6951 | 0.8113 | 0.1525 | 0.2542 | 0.4706 | 0.1207 | 0.7097 | 0.4359 |
| Early-Middle | 0.5896 | 0.8976 | 0.0615 | 0.1607 | 0.7778 | 0.0238 | 0.9036 | 0.4898 |
| Late-Career | 0.7976 | 0.8661 | 0.0000 | 0.1250 | 0.5000 | 0.1364 | 0.8824 | 0.4474 |
| Mid-Career | 0.6677 | 0.8511 | 0.1053 | 0.2500 | 0.5789 | 0.0566 | 0.7895 | 0.3711 |

#### personal + status + sex + age

| Group | F1 (Before) | F1 (After) | FNR (Before) | FNR (After) | FPR (Before) | FPR (After) | Sel. Rate (Before) | Sel. Rate (After) |
|-------|-------------|------------|--------------|-------------|--------------|-------------|--------------------|-------------------|
| male : single + Young | 0.6955 | 0.7686 | 0.1923 | 0.1935 | 0.4167 | 0.2308 | 0.6842 | 0.6364 |
| male : single + Early-Middle | 0.6110 | 0.9259 | 0.0400 | 0.0588 | 0.7778 | 0.0714 | 0.9322 | 0.6875 |
| female : divorced/separated/married + Young | 0.6703 | 0.8322 | 0.1667 | 0.2500 | 0.5000 | 0.0968 | 0.7000 | 0.3529 |
| male : married/widowed + Young | 0.7143 | 0.6990 | 0.0000 | 0.5000 | 0.6000 | 0.1000 | 0.8571 | 0.2778 |
| female : divorced/separated/married + Early-Middle | 0.8800 | 0.9231 | 0.0000 | 0.1429 | 0.3333 | 0.0000 | 0.8667 | 0.4615 |
| male : divorced/separated + Young | 1.0000 | 1.0000 | 0 | 0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| female : divorced/separated/married + Late-Career | 0.8070 | 0.8615 | 0.0000 | 0.1429 | 0.5000 | 0.0000 | 0.9091 | 0.6667 |
| male : single + Mid-Career | 0.7259 | 0.8486 | 0.0417 | 0.2222 | 0.5385 | 0.0769 | 0.8108 | 0.4340 |
| female : divorced/separated/married + Mid-Career | 0.5417 | 0.8542 | 0.1429 | 0.2308 | 0.7500 | 0.0667 | 0.8182 | 0.3929 |
| male : married/widowed + Early-Middle | 0.2500 | 0.7231 | 0.0000 | 0.5714 | 1.0000 | 0.0000 | 1.0000 | 0.1667 |
| male : married/widowed + Mid-Career | 0.4000 | 1.0000 | 0.2000 | 0.0000 | 1.0000 | 0.0000 | 0.8333 | 0.1111 |
| male : divorced/separated + Early-Middle | 0.6667 | 0.4545 | 0.0000 | 1.0000 | 0.5000 | 0.0000 | 0.6667 | 0.0000 |
| male : divorced/separated + Mid-Career | 0.2500 | 0.6500 | 0.5000 | 0.6667 | 1.0000 | 0.0000 | 0.6667 | 0.1429 |
| male : single + Late-Career | 0.7778 | 0.7750 | 0.0000 | 0.1429 | 0.5000 | 0.2727 | 0.8333 | 0.5000 |
