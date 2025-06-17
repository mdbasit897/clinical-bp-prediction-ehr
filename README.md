# Robust Blood Pressure Prediction from Electronic Health Records: A Comprehensive Framework with External Validation and Clinical Utility Assessment

**1<sup>st</sup> Md Basit Azam**<sup></sup>  
*Department of Computer Science & Engineering*  
Tezpur University  
Napaam - 784 028, Tezpur, Assam, INDIA  
📧 [mdbasit@tezu.ernet.in](mailto:mdbasit@tezu.ernet.in)

**2<sup>nd</sup> Sarangthem Ibotombi Singh**  
*Department of Computer Science & Engineering*  
Tezpur University  
Napaam - 784 028, Tezpur, Assam, INDIA  
📧 [sis@tezu.ernet.in](mailto:sis@tezu.ernet.in)


[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.xxxx%2Fxxxx-blue)](https://doi.org/10.xxxx/xxxx)

## 🔬 Abstract

This repository contains a comprehensive machine learning pipeline (`mimic_bp_pipelineV3.0.py`) for predicting systolic and diastolic blood pressure using readily available clinical variables from Electronic Health Records (EHR). Our methodology addresses critical challenges in clinical ML including data leakage prevention, uncertainty quantification, and external validation across multiple healthcare databases.

**Key Contributions:**
- 🎯 **Rigorous Data Leakage Prevention**: Systematic removal of verification features that could lead to unrealistic performance
- 🔀 **Advanced Feature Engineering**: Clinical interaction features and temporal dynamics analysis  
- 📊 **Bayesian Optimization**: Hyperparameter tuning with uncertainty quantification
- 🏥 **Clinical Utility Focus**: BHS/AAMI standards compliance and workflow impact simulation
- ✅ **External Validation**: Cross-database validation between MIMIC-III and eICU databases
- 📈 **State-of-the-Art Comparison**: Comprehensive benchmarking against existing methods

> **File Overview**: The main implementation is contained in `mimic_bp_pipelineV3.0.py` - a complete, self-contained pipeline that handles data extraction, feature engineering, model training, evaluation, and clinical validation.

## 🏆 Performance Highlights

| Metric | MIMIC-III (Internal) | eICU (External) | Clinical Standard |
|--------|---------------------|-----------------|-------------------|
| **SBP RMSE** | 6.03 mmHg | 7.84 mmHg | <8 mmHg (AAMI) |
| **DBP RMSE** | 7.13 mmHg | 9.31 mmHg | <8 mmHg (AAMI) |
| **SBP R²** | 0.86 | 0.84 | >0.60 (Good) |
| **DBP R²** | 0.49 | 0.32 | >0.50 (Acceptable) |
| **SBP Within 5mmHg** | 57.0% | 49.7% | >60% (BHS Grade A) |
| **DBP Within 5mmHg** | 55.7% | 43.9% | >60% (BHS Grade A) |
| **SBP Within 10mmHg** | 91.1% | N/A | >85% (BHS Grade A) |
| **DBP Within 10mmHg** | 86.7% | N/A | >85% (BHS Grade A) |
| **BHS Grade** | B (Both SBP/DBP) | N/A | A (Excellent) |
| **AAMI Standard** | ✅ PASS | N/A | PASS Required |

**Key Achievement**: 24.7% improvement in SBP RMSE compared to best literature method on MIMIC-III dataset.

## 🏗️ Methodology Overview

### Data Sources
- **Primary**: MIMIC-III Critical Care Database (n=889 patients)
- **External Validation**: eICU Collaborative Research Database (n=378 patients)
- **Inclusion**: ICU patients ≥18 years, >24h stay, valid BP measurements

### Feature Categories
- **Demographics**: Age, gender, ethnicity, BMI (74 final features after alignment)
- **Vital Signs**: Heart rate, respiratory rate, SpO2, temperature
- **Laboratory**: Electrolytes, kidney function, cardiac markers, blood gas
- **Medications**: Antihypertensives, vasopressors, diuretics
- **Comorbidities**: Hypertension, diabetes, heart failure, CKD
- **Temporal**: Heart rate variability, BP lability, measurement patterns
- **Engineered**: Clinical interactions, non-linear transformations

### Model Architecture