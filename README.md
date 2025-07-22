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
- **Primary**: MIMIC-III [[1]](#1) Critical Care Database (n=889 patients)
- **External Validation**: eICU Collaborative Research Database [[2]](#2) (n=378 patients)
- **Inclusion**: ICU patients ≥18 years, >24h stay, valid BP measurements

### Feature Categories
- **Demographics**: Age, gender, ethnicity, BMI (74 final features after alignment)
- **Vital Signs**: Heart rate, respiratory rate, SpO2, temperature
- **Laboratory**: Electrolytes, kidney function, cardiac markers, blood gas
- **Medications**: Antihypertensives, vasopressors, diuretics
- **Comorbidities**: Hypertension, diabetes, heart failure, CKD
- **Temporal**: Heart rate variability, BP lability, measurement patterns
- **Engineered**: Clinical interactions, non-linear transformations

### 🏗️ Model Architecture Overview

**1. Input Layer**  
74 clinical features extracted from EHR data  

**2. Preprocessing Pipeline**  
`Variance Threshold` → `Robust Scaling` → `Power Transform`  
*(Handles missing values, normalizes distributions, and reduces noise)*  

**3. Core Algorithm**  
🧠 **Gradient Boosting Regressor**  
*(Hyperparameters optimized via Bayesian Search)*  

**4. Output Configuration**  
→ **Multi-Output Prediction**: Simultaneous SBP & DBP estimation  
→ **Uncertainty Quantification**: 10th-90th percentile confidence intervals  

**5. Ensemble Strategy**  
🧬 **Stratified Ensemble Modeling**  
*Separate models trained for:*  
- Hypertensive range  
- Normal range  
- Hypotensive range  

### Validation Strategy
- **Internal**: 5-fold Group Cross-Validation (patient-wise splitting)
- **External**: Independent eICU database validation
- **Clinical**: BHS/AAMI standards, workflow simulation
- **Statistical**: Bland-Altman analysis, hypothesis testing

## 🚀 Quick Start

### Prerequisites
```python
# Python 3.8+ required
pip install numpy pandas scikit-learn matplotlib seaborn
pip install xgboost scikit-optimize shap google-cloud-bigquery
pip install missingno scipy statsmodels PyYAML

# Google Cloud access for MIMIC-III/eICU (credentials required)
export GOOGLE_CLOUD_PROJECT="your-project-id"

# Download the main pipeline file
wget https://github.com/yourusername/bp-prediction-ehr/raw/main/mimic_bp_pipelineV3.0.py

# Or clone the repository
git clone https://github.com/yourusername/bp-prediction-ehr.git

# Run the complete pipeline with default settings
python mimic_bp_pipelineV3.0.py

# With custom parameters
python mimic_bp_pipelineV3.0.py --log_level DEBUG --output_dir results

# Skip training (evaluation only)
python mimic_bp_pipelineV3.0.py --skip_training --load_models 20250607_125351
```

### ⚙️ Configuration Options

The pipeline uses a default configuration defined in the `DEFAULT_CONFIG` dictionary within the Python implementation. Key parameters include:

```python
DEFAULT_CONFIG = {
    'data_extraction': {
        'project_id': 'd1namo-ecg-processing',  # Your GCP project
        'max_patients': 1000,                   # Maximum patients to extract
        'use_eicu_validation': True             # Enable external validation
    },
    'model': {
        'bayesian_optimization': True,          # Use Bayesian optimization
        'stratified_modeling': True,            # Enable BP range-specific models
        'uncertainty_quantification': True,     # Add prediction intervals
        'use_stacked_ensemble': True           # Ensemble modeling
    },
    'evaluation': {
        'clinical_utility_metrics': True,      # Calculate clinical metrics
        'shap_analysis': True,                 # Feature importance analysis
        'sota_comparison': True                # Compare with literature
    }
}
```
### 🛠️ Configuration Flexibility
You can customize the pipeline through:
1. **Direct Python edits**: Modify `DEFAULT_CONFIG` dictionary
2. **CLI arguments**: Override specific parameters via command line
3. **External files**: Load custom configurations from YAML/JSON

---

### 📊 Detailed Results & Validation

#### 📈 Model Performance Metrics
| Category              | Metric                          | Value Range       |
|----------------------|---------------------------------|-------------------|
| **Regression**       | RMSE (SBP/DBP)                  | 6.03 mmHg         |
|                      | MAE                             | 4.52 mmHg         |
|                      | R²                              | 0.86              |
|                      | Pearson Correlation             | 0.93              |
| **Clinical Accuracy**| Within 5 mmHg                   | 56.6-69.6%        |
|                      | Within 10 mmHg                  | 82.4-93.2%        |
| **Uncertainty**      | Interval Coverage               | ~80%              |
| **Classification**   | Hypertension Sensitivity        | 92.3%             |
|                      | Hypotension Specificity         | 94.1%             |

#### 🏥 Clinical Validation
✅ **BHS Grade B** certification for both SBP & DBP (MIMIC-III)  
✅ **AAMI Compliance**:  
- Mean difference: <5 mmHg  
- Standard deviation: <8 mmHg  

⚡ **Clinical Impact**:  
- Potential saving: **424.8 nurse-hours/day**  
- Risk stratification:  
  - Low-risk: 383 patients  
  - Medium-risk: 206 patients  
  - High-risk: 300 patients

---

### 🧪 Stratified Performance (MIMIC-III)
**SBP Accuracy by Category**
- **Normal BP**: 56.6% within 5mmHg (n=461)
- **Prehypertension**: 56.3% within 5mmHg (n=302)
- **Stage 1 HTN**: 63.2% within 5mmHg (n=95)
- **Stage 2 HTN**: 69.6% within 5mmHg (n=23)

**DBP Accuracy by Category**
- **Hypotension**: 60.4% within 5mmHg (n=495)
- **Normal**: 52.4% within 5mmHg (n=368)

---

### 📁 Repository Structure

### 📄 File Descriptions
- `README.md`: This document containing methodology overview, configuration, and results
- `mimic_bp_pipelineV3.0.py`: Core implementation of the blood pressure prediction pipeline
- `LICENSE`: MIT License file granting usage rights

> **Note**: As your project grows, you can expand this structure with directories for data, notebooks, and results.

### 🛠️ Pipeline Components (within `mimic_bp_pipelineV3.0.py`)

The single-file implementation contains all necessary components:

**1. Data Extractors**  
- `MIMICDataExtractor`, `eICUDataExtractor`  
  Classes for BigQuery data extraction from clinical databases

**2. Feature Engineering**  
- `EnhancedFeatureProcessor`  
  Creates clinical feature interactions and derived variables

**3. Missing Data Handling**  
- `AdvancedDataImputer`  
  Supports multiple strategies:  
  - MICE (Multivariate Imputation by Chained Equations)  
  - KNN imputation  
  - Domain-specific fallback values

**4. Model Training**  
- `OptimizedBPPredictor`  
  Implements:  
  - Bayesian hyperparameter optimization  
  - Gradient boosting with stratified ensembling  
  - Multi-output SBP/DBP prediction

**5. Evaluation Suite**  
- `ModelEvaluator`  
  Metrics include:  
  - Clinical accuracy thresholds (±5/10/15 mmHg)  
  - Uncertainty calibration  
  - Risk stratification performance

**6. Utility Functions**  
- Data leakage prevention  
- Feature alignment across datasets  
- Clinical validation tools (BHS/AAMI standards)

---

### 🔧 Advanced Usage  
**Command Line Options** 

```python
# Full pipeline with custom settings
python mimic_bp_pipelineV3.0.py --log_level INFO --output_dir my_results

# Skip training (use pre-trained models)
python mimic_bp_pipelineV3.0.py --skip_training --load_models 20250607_125351

# Custom timestamp for reproducibility
python mimic_bp_pipelineV3.0.py --timestamp 20250607_125351
```
**Modifying Configuration**

Edit the `DEFAULT_CONFIG` dictionary in the Python file to customize:

```python
# Example: Reduce dataset size for testing
DEFAULT_CONFIG['data_extraction']['max_patients'] = 100

# Example: Disable external validation for faster runs
DEFAULT_CONFIG['data_extraction']['use_eicu_validation'] = False

# Example: Enable enhanced DBP modeling
DEFAULT_CONFIG['model']['enhanced_dbp_modeling'] = True
```
### 🔑 Key Classes and Functions

All functionality is contained within the single Python file:

- **`MIMICDataExtractor`**  
  Extracts patient data from MIMIC-III via BigQuery  

- **`eICUDataExtractor`**  
  Extracts patient data from eICU via BigQuery  

- **`EnhancedFeatureProcessor`**  
  Creates clinical features and interactions (e.g., medication × vitals)  

- **`OptimizedBPPredictor`**  
  Trains ML models with Bayesian optimization and stratified ensembling  

- **`ModelEvaluator`**  
  Evaluates models using clinical standards (BHS grading, AAMI compliance)  

- **`remove_leaky_features()`**  
  Prevents data leakage by removing verification features  

- **`run_eicu_external_validation()`**  
  Performs external validation on eICU dataset  

---

### 🏥 Clinical Significance

#### 📊 Clinical Decision Support Applications
- **Risk Stratification**: Automatic categorization of patients by BP stability  
- **Monitoring Optimization**: Reduced manual measurements for low-risk patients  
- **Early Warning**: Detection of BP changes 30 minutes earlier than manual methods  
- **Resource Allocation**: Optimized nurse workflow with 424.8 hours saved per day potential  

#### 📏 Regulatory Compliance
- **AAMI Standards**: Mean difference <5 mmHg, SD <8 mmHg ✅ *ACHIEVED*  
- **BHS Grading**: Grade B achieved (target Grade A: ≥60% within 5 mmHg)  
- **FDA Guidance**: Uncertainty quantification and external validation implemented  
- **ISO Standards**: Clinical evaluation and risk management protocols followed  

#### 💡 Clinical Utility Metrics
- **Hypotension Detection**: 100% specificity, 99.1% NPV  
- **Hypertension Detection**: 100% sensitivity and specificity  
- **Prediction Intervals**: 80.3% coverage with 22.89 mmHg average width  

---

### 📈 Key Findings

- **Data Leakage Prevention**: Removal of verification features maintains clinical utility while preventing overfitting  
- **Feature Importance**: Heart rate variability, medication interactions, and temporal patterns are key predictors  
- **Generalizability**: External validation shows 30% RMSE increase (SBP) and 31% (DBP) - indicating reasonable but limited generalizability  
- **Clinical Impact**: Potential for 35% reduction in manual BP measurements while maintaining safety  
- **Uncertainty Quantification**: Reliable prediction intervals enable risk-based clinical decisions (80% coverage achieved)  
- **Stratified Performance**: Better performance in hypertensive patients vs hypotensive patients  

---

### 🔬 Reproducibility

#### 📁 Data Access Requirements
- **MIMIC-III**: Requires [PhysioNet](https://physionet.org/content/mimiciii/1.4/)  credentialed access and approved research project  
- **eICU**: Separate [PhysioNet](https://physionet.org/content/eicu-crd/2.0/)  credentialing required  
- **Google Cloud**: BigQuery access configured with appropriate project permissions

**Environment Setup**

```python
# Create isolated environment
conda create -n bp-prediction python=3.8
conda activate bp-prediction

# Install required packages
pip install numpy pandas scikit-learn matplotlib seaborn xgboost scikit-optimize shap google-cloud-bigquery missingno scipy statsmodels PyYAML

# Set up Google Cloud credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
export GOOGLE_CLOUD_PROJECT="your-project-id"
```

**Running Experiments**
```python
# Download the pipeline
wget https://raw.githubusercontent.com/yourusername/bp-prediction-ehr/main/mimic_bp_pipelineV3.0.py

# Run with default settings (requires MIMIC-III access)
python mimic_bp_pipelineV3.0.py

# Run with custom parameters
python mimic_bp_pipelineV3.0.py --log_level DEBUG --output_dir my_experiment

# Reproduce specific experiment
python mimic_bp_pipelineV3.0.py --timestamp 20250607_125351
```
### 📤 Output Files

The pipeline generates timestamped output files to ensure reproducibility and traceability:

- **`bp_prediction_YYYYMMDD_HHMMSS.log`**  
  Detailed execution log including warnings, errors, and progress tracking  

- **`feature_importance_YYYYMMDD_HHMMSS.csv`**  
  Ranked list of clinical features with SHAP-based importance scores  

- **`MIMIC-III_evaluation_YYYYMMDD_HHMMSS.json`**  
  Internal validation metrics (RMSE, R², clinical thresholds)  

- **`eICU_evaluation_YYYYMMDD_HHMMSS.json`**  
  External validation results with stratified performance metrics  

- **`figures_YYYYMMDD_HHMMSS/`**  
  Directory containing:  
  - Calibration plots  
  - Prediction interval visualizations  
  - Feature importance charts  
  - Bland-Altman agreement plots  

- **`models/`**  
  Trained model artifacts for reuse:  
  - Gradient boosting ensemble  
  - Imputation strategy pickles  
  - Feature processor configurations  

- **`imputation_stats_*_YYYYMMDD_HHMMSS.csv`**  
  Comparative analysis of MICE vs KNN imputation performance  

- **`*_stratified_results_*_YYYYMMDD_HHMMSS.csv`**  
  Performance breakdown by BP category (hypotension/normal/hypertension)

> **Note**: Timestamps (YYYYMMDD_HHMMSS) ensure unique filenames across runs while maintaining traceability to specific experiments.

💡 **Tip**: Use `--output_dir` parameter to customize the output directory location for organized storage across multiple experiments.

***📚 Citation***

**If you use this code in your research, kindly cite:**

## 🤝 Contributing
We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

**Steps to contribute:**
1. Fork the repository  
2. Create a feature branch  
   ```bash
   git checkout -b feature/amazing-feature
   ##Commit changes
   git commit -m 'Add amazing feature'
   ##Push to branch
   git push origin feature/amazing-feature
3. Open a Pull Request

### 🔄 Updates

- **v3.0**: Enhanced pipeline with external validation and clinical utility metrics  
- **v2.1**: Added uncertainty quantification and stratified modeling  
- **v2.0**: Implemented Bayesian optimization and SHAP analysis  
- **v1.0**: Initial release with basic BP prediction pipeline

## 📄 License  
This project uses an MIT License. See the [LICENSE file](LICENSE) for details.  
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE) 

## 🙏 Acknowledgments  
The authors acknowledge support from the Google Cloud Research Credits program under 
Award GCP19980904 and partial compute resources from Google’s TPU Research Cloud (TRC), 
both of which provided critical infrastructure for this research.

### Funding:
The authors declare no funding was received for this research.

## References
<a id="1">[1]</a> 
Johnson, A., Pollard, T., & Mark, R. (2016). MIMIC-III Clinical Database (version 1.4). 
PhysioNet. RRID:SCR_007345. https://doi.org/10.13026/C2XW26

<a id="2">[2]</a> 
Pollard, T., Johnson, A., Raffa, J., Celi, L. A., Badawi, O., & Mark, R. (2019). 
eICU Collaborative Research Database (version 2.0). PhysioNet. RRID:SCR_007345. 
https://doi.org/10.13026/C2WM1R


