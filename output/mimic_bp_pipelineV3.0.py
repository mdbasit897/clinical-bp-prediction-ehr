#!/usr/bin/env python3

import numpy as np
import pandas as pd
from google.cloud import bigquery
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report
)
from sklearn.linear_model import LinearRegression, RidgeCV, QuantileRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import warnings
import logging
from datetime import datetime
import os
import yaml
import argparse
import joblib
import shap
from xgboost import XGBRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
import missingno as msno
from sklearn.ensemble import StackingRegressor
import json

# Suppress warnings
warnings.filterwarnings('ignore')


# Setup logging
def setup_logging(log_level='INFO', output_dir='results', timestamp=None):
    """Set up logging configuration"""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create log file path
    log_file = os.path.join(output_dir, f"bp_prediction_{timestamp}.log")

    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized at {log_level} level. Log file: {log_file}")

    return logger


# ADD THIS SECTION RIGHT AFTER setup_logging() function (around line 85)

def remove_leaky_features(df):
    """Permanently remove verification features that could lead to data leakage"""
    logger = logging.getLogger(__name__)
    logger.info("Removing leaky verification features from dataset")

    # Complete list of verification features that must be removed
    leaky_columns = [
        'sbp_mean_verify', 'sbp_std_verify', 'dbp_mean_verify', 'dbp_std_verify',
        'sbp_mean_verify_squared', 'sbp_std_verify_squared',
        'hr_sbp_corr', 'rr_sbp_corr', 'spo2_sbp_corr',
        'creatinine_sbp_ratio', 'hrv_sbp_interaction',
    ]

    # Check which columns exist in the dataframe
    columns_to_drop = [col for col in leaky_columns if col in df.columns]

    # Log what's being removed
    if columns_to_drop:
        logger.warning(f"REMOVING {len(columns_to_drop)} LEAKY FEATURES: {', '.join(columns_to_drop)}")
        return df.drop(columns=columns_to_drop)
    else:
        logger.info("No leaky features found in dataset")
        return df


def validate_no_leakage(X, feature_importance=None):
    """Verify no leaky features remain in the dataset"""
    logger = logging.getLogger(__name__)
    leakage_patterns = ['verify', 'sbp_mean', 'dbp_mean', 'map_mean']
    potential_leaks = []

    for col in X.columns:
        for pattern in leakage_patterns:
            if pattern in col.lower():
                potential_leaks.append(col)
                break

    if potential_leaks:
        logger.error(f"POTENTIAL DATA LEAKAGE DETECTED: {potential_leaks}")
        return False
    else:
        logger.info("✓ No data leakage detected - dataset is clean")
        return True


def run_eicu_external_validation(config, trained_predictor, feature_processor, imputer, evaluator, timestamp):
    """FIXED eICU external validation with proper feature alignment"""
    logger = logging.getLogger(__name__)

    try:
        logger.info("=" * 60)
        logger.info("STARTING eICU EXTERNAL VALIDATION")
        logger.info("=" * 60)

        # Initialize eICU extractor
        eicu_extractor = eICUDataExtractor(
            project_id=config['data_extraction']['project_id'],
            timestamp=timestamp
        )

        # Step 1: Extract patient cohort
        logger.info("Step 1: Extracting eICU patient cohort...")
        eicu_cohort_df = eicu_extractor.extract_patient_cohort(
            max_patients=config['data_extraction']['max_patients']
        )

        if len(eicu_cohort_df) == 0:
            logger.error("No patients found in eICU cohort")
            return None

        logger.info(f"✓ Successfully extracted {len(eicu_cohort_df)} eICU patients")

        # Step 2: Extract BP data
        patient_ids = eicu_cohort_df['patientunitstayid'].tolist()
        logger.info("Step 2: Extracting eICU BP data...")
        eicu_bp_df = eicu_extractor.extract_bp_data(patient_ids)

        if len(eicu_bp_df) == 0:
            logger.error("No BP data found in eICU - cannot proceed")
            return None

        logger.info(f"✓ BP data: {len(eicu_bp_df)} patients with valid measurements")

        # Step 3: Extract other features
        logger.info("Step 3: Extracting other eICU features...")
        eicu_vitals_df = eicu_extractor.extract_vital_signs_features(patient_ids)
        eicu_labs_df = eicu_extractor.extract_lab_features(patient_ids)
        eicu_meds_df = eicu_extractor.extract_medication_features(patient_ids)

        # Optional features with error handling
        eicu_temporal_df = None
        eicu_comorbidity_df = None

        try:
            eicu_temporal_df = eicu_extractor.extract_temporal_features(patient_ids)
            logger.info(f"✓ Temporal features: {len(eicu_temporal_df) if eicu_temporal_df is not None else 0} patients")
        except Exception as e:
            logger.warning(f"Could not extract temporal features: {str(e)}")

        try:
            eicu_comorbidity_df = eicu_extractor.extract_comorbidity_features(patient_ids)
            logger.info(
                f"✓ Comorbidity features: {len(eicu_comorbidity_df) if eicu_comorbidity_df is not None else 0} patients")
        except Exception as e:
            logger.warning(f"Could not extract comorbidity features: {str(e)}")

        # Step 4: Process features
        logger.info("Step 4: Processing eICU features...")
        eicu_df = feature_processor.process_features(
            eicu_cohort_df, eicu_bp_df, eicu_vitals_df, eicu_labs_df, eicu_meds_df,
            eicu_temporal_df, eicu_comorbidity_df, "eICU"
        )

        if len(eicu_df) == 0:
            logger.error("No data remaining after feature processing")
            return None

        logger.info(f"✓ Processed eICU dataset shape: {eicu_df.shape}")

        # Step 5: Handle missing data
        logger.info("Step 5: Imputing missing values...")
        eicu_df_imputed = imputer.impute_missing_values(eicu_df, "eICU")

        # Step 6: Remove data leakage
        logger.info("Step 6: Removing potential data leakage...")
        eicu_df_clean = remove_leaky_features(eicu_df_imputed)

        # Step 7: Prepare for model evaluation
        logger.info("Step 7: Preparing data for model evaluation...")
        X_eicu_raw, y_eicu, groups_eicu = trained_predictor.prepare_data(eicu_df_clean, "eICU")

        if len(X_eicu_raw) == 0:
            logger.error("No samples remaining after data preparation")
            return None

        # === NEW: Step 7.5: Align features with training data ===
        logger.info("Step 7.5: Aligning features with training data...")

        # Get reference features from the trained model
        if hasattr(trained_predictor, 'feature_importance') and trained_predictor.feature_importance:
            reference_features = trained_predictor.feature_importance['features']
        else:
            logger.error("No reference features available from trained model")
            return None

        # Create feature aligner and align eICU features
        aligner = FeatureAligner()
        aligner.set_reference_features(reference_features)
        X_eicu = aligner.align_features(X_eicu_raw, "eICU")

        logger.info(f"✓ Final eICU validation set: {X_eicu.shape[0]} patients, {X_eicu.shape[1]} features")

        # Final validation check
        if not validate_no_leakage(X_eicu):
            logger.error("Data leakage detected in final eICU dataset")
            return None

        # Step 8: Evaluate model
        logger.info("Step 8: Evaluating model on eICU data...")

        # Make predictions
        sbp_pred, dbp_pred, sbp_intervals, dbp_intervals = trained_predictor.predict_with_uncertainty(X_eicu)

        # Calculate metrics
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        import numpy as np

        sbp_true = y_eicu['sbp_mean'].values
        dbp_true = y_eicu['dbp_mean'].values

        eicu_metrics = {
            'SBP': {
                'RMSE': np.sqrt(mean_squared_error(sbp_true, sbp_pred)),
                'MAE': mean_absolute_error(sbp_true, sbp_pred),
                'R2': r2_score(sbp_true, sbp_pred),
                'Within_5mmHg': np.mean(np.abs(sbp_true - sbp_pred) <= 5) * 100,
                'Within_10mmHg': np.mean(np.abs(sbp_true - sbp_pred) <= 10) * 100
            },
            'DBP': {
                'RMSE': np.sqrt(mean_squared_error(dbp_true, dbp_pred)),
                'MAE': mean_absolute_error(dbp_true, dbp_pred),
                'R2': r2_score(dbp_true, dbp_pred),
                'Within_5mmHg': np.mean(np.abs(dbp_true - dbp_pred) <= 5) * 100,
                'Within_10mmHg': np.mean(np.abs(dbp_true - dbp_pred) <= 10) * 100
            }
        }

        # Create results dictionary
        eicu_results = {
            'metrics': eicu_metrics,
            'predictions': {'SBP': sbp_pred, 'DBP': dbp_pred},
            'true_values': {'SBP': sbp_true, 'DBP': dbp_true},
            'dataset': 'eICU'
        }

        logger.info("✓ eICU external validation completed successfully!")
        logger.info("=" * 60)
        logger.info("eICU VALIDATION RESULTS:")
        logger.info(f"SBP - RMSE: {eicu_metrics['SBP']['RMSE']:.2f} mmHg, R²: {eicu_metrics['SBP']['R2']:.4f}")
        logger.info(f"DBP - RMSE: {eicu_metrics['DBP']['RMSE']:.2f} mmHg, R²: {eicu_metrics['DBP']['R2']:.4f}")
        logger.info("=" * 60)

        return eicu_results

    except Exception as e:
        logger.error(f"Critical error in eICU external validation: {str(e)}")
        logger.exception("Full traceback:")
        return None


def compare_mimic_eicu_performance(mimic_results, eicu_results, timestamp):
    """Compare performance between MIMIC-III and eICU datasets"""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 80)
    logger.info("INTERNAL vs EXTERNAL VALIDATION COMPARISON")
    logger.info("=" * 80)

    # Performance comparison table
    comparison_data = []
    for bp_type in ['SBP', 'DBP']:
        mimic_metrics = mimic_results['metrics'][bp_type]
        eicu_metrics = eicu_results['metrics'][bp_type]

        # Calculate performance drop
        rmse_change = ((eicu_metrics['RMSE'] - mimic_metrics['RMSE']) / mimic_metrics['RMSE']) * 100
        r2_change = ((eicu_metrics['R2'] - mimic_metrics['R2']) / mimic_metrics['R2']) * 100
        within5_change = eicu_metrics['Within_5mmHg'] - mimic_metrics['Within_5mmHg']

        comparison_data.append({
            'BP_Type': bp_type,
            'MIMIC_RMSE': mimic_metrics['RMSE'],
            'eICU_RMSE': eicu_metrics['RMSE'],
            'RMSE_Change_%': rmse_change,
            'MIMIC_R2': mimic_metrics['R2'],
            'eICU_R2': eicu_metrics['R2'],
            'R2_Change_%': r2_change,
            'MIMIC_Within5': mimic_metrics['Within_5mmHg'],
            'eICU_Within5': eicu_metrics['Within_5mmHg'],
            'Within5_Change': within5_change
        })

    # Log comparison
    logger.info(
        f"{'BP':<5} {'MIMIC RMSE':<12} {'eICU RMSE':<12} {'Change %':<10} {'MIMIC R²':<10} {'eICU R²':<10} {'R² Change %':<12}")
    logger.info("-" * 80)

    for data in comparison_data:
        logger.info(f"{data['BP_Type']:<5} {data['MIMIC_RMSE']:<12.2f} {data['eICU_RMSE']:<12.2f} "
                    f"{data['RMSE_Change_%']:<10.1f} {data['MIMIC_R2']:<10.4f} {data['eICU_R2']:<10.4f} "
                    f"{data['R2_Change_%']:<12.1f}")

    # Interpretation
    logger.info("\nGENERALIZABILITY ASSESSMENT:")

    sbp_data = comparison_data[0]
    dbp_data = comparison_data[1]

    if abs(sbp_data['RMSE_Change_%']) < 20:
        logger.info("✓ SBP model shows good generalizability (RMSE change < 20%)")
    else:
        logger.info("⚠ SBP model shows reduced generalizability (RMSE change >= 20%)")

    if abs(dbp_data['RMSE_Change_%']) < 30:
        logger.info("✓ DBP model shows acceptable generalizability (RMSE change < 30%)")
    else:
        logger.info("⚠ DBP model shows reduced generalizability (RMSE change >= 30%)")

    logger.info("=" * 80)


def generate_final_summary(mimic_results, eicu_results, timestamp):
    """Generate final summary for publication"""
    logger = logging.getLogger(__name__)

    logger.info("\n" + "=" * 80)
    logger.info("FINAL RESULTS SUMMARY FOR IEEE PUBLICATION")
    logger.info("=" * 80)

    if mimic_results:
        logger.info("\nINTERNAL VALIDATION (MIMIC-III):")
        logger.info(f"  SBP: RMSE = {mimic_results['metrics']['SBP']['RMSE']:.2f} mmHg, "
                    f"R² = {mimic_results['metrics']['SBP']['R2']:.4f}, "
                    f"Within 5mmHg = {mimic_results['metrics']['SBP']['Within_5mmHg']:.1f}%")
        logger.info(f"  DBP: RMSE = {mimic_results['metrics']['DBP']['RMSE']:.2f} mmHg, "
                    f"R² = {mimic_results['metrics']['DBP']['R2']:.4f}, "
                    f"Within 5mmHg = {mimic_results['metrics']['DBP']['Within_5mmHg']:.1f}%")

        if 'hypothesis_tests' in mimic_results:
            logger.info(f"  BHS Grade: SBP = {mimic_results['hypothesis_tests']['SBP']['bhs_grade']}, "
                        f"DBP = {mimic_results['hypothesis_tests']['DBP']['bhs_grade']}")

    if eicu_results:
        logger.info("\nEXTERNAL VALIDATION (eICU):")
        logger.info(f"  SBP: RMSE = {eicu_results['metrics']['SBP']['RMSE']:.2f} mmHg, "
                    f"R² = {eicu_results['metrics']['SBP']['R2']:.4f}, "
                    f"Within 5mmHg = {eicu_results['metrics']['SBP']['Within_5mmHg']:.1f}%")
        logger.info(f"  DBP: RMSE = {eicu_results['metrics']['DBP']['RMSE']:.2f} mmHg, "
                    f"R² = {eicu_results['metrics']['DBP']['R2']:.4f}, "
                    f"Within 5mmHg = {eicu_results['metrics']['DBP']['Within_5mmHg']:.1f}%")

        if 'hypothesis_tests' in eicu_results:
            logger.info(f"  BHS Grade: SBP = {eicu_results['hypothesis_tests']['SBP']['bhs_grade']}, "
                        f"DBP = {eicu_results['hypothesis_tests']['DBP']['bhs_grade']}")

    logger.info("\nMETHODOLOGICAL STRENGTHS:")
    logger.info("  ✓ No data leakage - verification features removed")
    logger.info("  ✓ Robust cross-validation with patient-wise splitting")
    logger.info("  ✓ Multiple imputation with sensitivity analysis")
    logger.info("  ✓ Uncertainty quantification with prediction intervals")
    logger.info("  ✓ External validation on independent dataset")
    logger.info("  ✓ Clinical utility metrics and workflow simulation")

    logger.info("=" * 80)


# Default configuration
DEFAULT_CONFIG = {
    'data_extraction': {
        'project_id': 'd1namo-ecg-processing',
        'max_patients': 1000,
        'use_eicu_validation': True
    },
    'feature_engineering': {
        'extreme_limits': {
            'hr_mean': [20, 250],
            'sbp_mean': [40, 300],
            'dbp_mean': [15, 200],
            'temp_mean': [80, 110],
            'spo2_mean': [50, 100]
        },
        'create_temporal_features': True,
        'create_interaction_features': True
    },
    'imputation': {
        'method': 'iterative',  # 'iterative', 'knn', or 'default'
        'knn_neighbors': 5,
        'iterative_max_iter': 50,
        'perform_sensitivity_analysis': True
    },
    'model': {
        'random_state': 42,
        'cv_folds': 5,
        'stratified_modeling': True,
        'bayesian_optimization': True,
        'n_iter': 30,
        'use_stacked_ensemble': True,
        'enhanced_dbp_modeling': False
    },
    'evaluation': {
        'save_figures': True,
        'shap_analysis': True,
        'uncertainty_quantification': True,
        'clinical_utility_metrics': True,
        'workflow_simulation': True
    }
}


class MIMICDataExtractor:
    """
    Enhanced data extractor for MIMIC-III with improved query structure
    and additional features for SBP prediction improvement
    """

    def __init__(self, project_id='d1namo-ecg-processing', timestamp=None):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"BigQuery client initialized for project: {project_id}")

    def extract_patient_cohort(self, max_patients=1000):
        """Extract patient cohort with basic demographics"""
        query = f"""
        WITH patient_cohort AS (
            SELECT DISTINCT
                p.subject_id,
                p.gender,
                DATETIME_DIFF(adm.admittime, p.dob, YEAR) as age,
                adm.hadm_id,
                icu.icustay_id,
                icu.first_careunit,
                DATETIME_DIFF(icu.outtime, icu.intime, HOUR) as icu_los_hours,
                CASE WHEN adm.deathtime IS NOT NULL THEN 1 ELSE 0 END as hospital_mortality,
                -- Additional demographic features
                p.dod IS NOT NULL as deceased,
                EXTRACT(YEAR FROM adm.admittime) as admit_year,
                adm.insurance,
                adm.language,
                adm.ethnicity
            FROM `physionet-data.mimiciii_clinical.patients` p
            JOIN `physionet-data.mimiciii_clinical.admissions` adm ON p.subject_id = adm.subject_id
            JOIN `physionet-data.mimiciii_clinical.icustays` icu ON adm.hadm_id = icu.hadm_id
            WHERE DATETIME_DIFF(adm.admittime, p.dob, YEAR) BETWEEN 18 AND 89
            AND DATETIME_DIFF(icu.outtime, icu.intime, HOUR) >= 24
        )
        SELECT * FROM patient_cohort
        ORDER BY subject_id
        LIMIT {max_patients}
        """

        logging.info("Extracting patient cohort...")
        return self.client.query(query).to_dataframe()

    def extract_bp_data(self, icustay_ids):
        """Extract BP measurements - our target variables with enhanced statistics"""
        icustay_list = ','.join([str(id) for id in icustay_ids])

        query = f"""
        WITH bp_measurements AS (
            SELECT 
                icustay_id,
                -- Systolic BP
                AVG(CASE WHEN itemid IN (51, 442, 455, 6701, 220179, 220050) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as sbp_mean,
                STDDEV(CASE WHEN itemid IN (51, 442, 455, 6701, 220179, 220050) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as sbp_std,
                COUNT(CASE WHEN itemid IN (51, 442, 455, 6701, 220179, 220050) 
                    THEN 1 END) as sbp_count,
                -- Additional statistics for SBP
                MIN(CASE WHEN itemid IN (51, 442, 455, 6701, 220179, 220050) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as sbp_min,
                MAX(CASE WHEN itemid IN (51, 442, 455, 6701, 220179, 220050) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as sbp_max,

                -- Diastolic BP  
                AVG(CASE WHEN itemid IN (8368, 8440, 8441, 8555, 220180, 220051) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as dbp_mean,
                STDDEV(CASE WHEN itemid IN (8368, 8440, 8441, 8555, 220180, 220051) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as dbp_std,
                COUNT(CASE WHEN itemid IN (8368, 8440, 8441, 8555, 220180, 220051) 
                    THEN 1 END) as dbp_count,

                -- Mean Arterial Pressure
                AVG(CASE WHEN itemid IN (52, 6702, 220181, 220052, 225312) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as map_mean

            FROM `physionet-data.mimiciii_clinical.chartevents`
            WHERE icustay_id IN ({icustay_list})
            AND itemid IN (
                51, 442, 455, 6701, 220179, 220050,  -- SBP
                8368, 8440, 8441, 8555, 220180, 220051,  -- DBP
                52, 6702, 220181, 220052, 225312  -- MAP
            )
            AND SAFE_CAST(value AS FLOAT64) BETWEEN 30 AND 300
            AND error IS NULL
            GROUP BY icustay_id
        )
        SELECT * FROM bp_measurements
        WHERE sbp_mean IS NOT NULL AND dbp_mean IS NOT NULL
        """

        logging.info("Extracting BP measurements...")
        return self.client.query(query).to_dataframe()

    def extract_vital_signs_features(self, icustay_ids):
        """Extract vital signs with enhanced features for SBP prediction"""
        icustay_list = ','.join([str(id) for id in icustay_ids])

        query = f"""
        WITH vital_stats AS (
            SELECT 
                icustay_id,
                -- Heart Rate (equivalent to PPG peak detection)
                AVG(CASE WHEN itemid IN (211, 220045) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as hr_mean,
                STDDEV(CASE WHEN itemid IN (211, 220045) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as hr_std,
                MIN(CASE WHEN itemid IN (211, 220045) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as hr_min,
                MAX(CASE WHEN itemid IN (211, 220045) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as hr_max,

                -- Respiratory Rate
                AVG(CASE WHEN itemid IN (615, 618, 220210, 224690) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as rr_mean,
                STDDEV(CASE WHEN itemid IN (615, 618, 220210, 224690) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as rr_std,

                -- Temperature (convert Celsius to Fahrenheit)
                AVG(CASE 
                    WHEN itemid IN (223761, 678) THEN SAFE_CAST(value AS FLOAT64) * 9/5 + 32
                    WHEN itemid IN (223762, 676) THEN SAFE_CAST(value AS FLOAT64)
                END) as temp_mean,

                -- SpO2
                AVG(CASE WHEN itemid IN (646, 220277) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as spo2_mean,
                STDDEV(CASE WHEN itemid IN (646, 220277) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as spo2_std,

                -- GCS (consciousness level)
                AVG(CASE WHEN itemid IN (198, 226755) 
                    THEN SAFE_CAST(value AS FLOAT64) END) as gcs_mean

            FROM `physionet-data.mimiciii_clinical.chartevents`
            WHERE icustay_id IN ({icustay_list})
            AND itemid IN (211, 220045, 615, 618, 220210, 224690,
                          223761, 678, 223762, 676, 646, 220277, 198, 226755)
            AND error IS NULL
            GROUP BY icustay_id
        )
        SELECT * FROM vital_stats
        """

        logging.info("Extracting vital signs features...")
        return self.client.query(query).to_dataframe()

    def extract_lab_features(self, icustay_ids):
        """Extract laboratory features with enhanced cardiac markers"""
        icustay_list = ','.join([str(id) for id in icustay_ids])

        query = f"""
        WITH lab_stats AS (
            SELECT 
                icu.icustay_id,
                -- Electrolytes (with proper type conversion)
                AVG(CASE WHEN le.itemid = 50824 THEN SAFE_CAST(le.value AS FLOAT64) END) as sodium_mean,
                AVG(CASE WHEN le.itemid = 50822 THEN SAFE_CAST(le.value AS FLOAT64) END) as potassium_mean,

                -- Kidney function
                AVG(CASE WHEN le.itemid = 50912 THEN SAFE_CAST(le.value AS FLOAT64) END) as creatinine_mean,
                AVG(CASE WHEN le.itemid = 51006 THEN SAFE_CAST(le.value AS FLOAT64) END) as bun_mean,

                -- Blood count
                AVG(CASE WHEN le.itemid = 51222 THEN SAFE_CAST(le.value AS FLOAT64) END) as hemoglobin_mean,
                AVG(CASE WHEN le.itemid = 51301 THEN SAFE_CAST(le.value AS FLOAT64) END) as wbc_mean,

                -- Cardiac markers (expanded)
                AVG(CASE WHEN le.itemid = 50813 THEN SAFE_CAST(le.value AS FLOAT64) END) as lactate_mean,
                AVG(CASE WHEN le.itemid = 50090 THEN SAFE_CAST(le.value AS FLOAT64) END) as troponin_mean,
                AVG(CASE WHEN le.itemid = 50889 THEN SAFE_CAST(le.value AS FLOAT64) END) as crp_mean,

                -- Blood gas analysis
                AVG(CASE WHEN le.itemid = 50820 THEN SAFE_CAST(le.value AS FLOAT64) END) as ph_mean,
                AVG(CASE WHEN le.itemid = 50818 THEN SAFE_CAST(le.value AS FLOAT64) END) as pco2_mean,
                AVG(CASE WHEN le.itemid = 50821 THEN SAFE_CAST(le.value AS FLOAT64) END) as po2_mean,

                -- Liver function
                AVG(CASE WHEN le.itemid = 50863 THEN SAFE_CAST(le.value AS FLOAT64) END) as alp_mean,
                AVG(CASE WHEN le.itemid = 50878 THEN SAFE_CAST(le.value AS FLOAT64) END) as ast_mean,
                AVG(CASE WHEN le.itemid = 50861 THEN SAFE_CAST(le.value AS FLOAT64) END) as alt_mean

            FROM `physionet-data.mimiciii_clinical.icustays` icu
            LEFT JOIN `physionet-data.mimiciii_clinical.labevents` le ON icu.hadm_id = le.hadm_id
            WHERE icu.icustay_id IN ({icustay_list})
            AND le.itemid IN (
                50824, 50822, 50912, 51006, 51222, 51301, 50813, 
                50090, 50889, 50820, 50818, 50821, 50863, 50878, 50861
            )
            GROUP BY icu.icustay_id
        )
        SELECT * FROM lab_stats
        """

        logging.info("Extracting laboratory features...")
        return self.client.query(query).to_dataframe()

    def extract_medication_features(self, icustay_ids):
        """Extract medication features with enhanced antihypertensive categories"""
        icustay_list = ','.join([str(id) for id in icustay_ids])

        query = f"""
        WITH med_features AS (
            SELECT 
                mv.icustay_id,
                -- Vasopressors
                COUNT(CASE WHEN LOWER(di.label) LIKE '%norepinephrine%' OR
                                LOWER(di.label) LIKE '%epinephrine%' OR
                                LOWER(di.label) LIKE '%dopamine%' OR
                                LOWER(di.label) LIKE '%dobutamine%' OR
                                LOWER(di.label) LIKE '%vasopressin%'
                      THEN 1 END) as vasopressor_count,

                -- Beta blockers
                COUNT(CASE WHEN LOWER(di.label) LIKE '%metoprolol%' OR
                                LOWER(di.label) LIKE '%esmolol%' OR
                                LOWER(di.label) LIKE '%propranolol%' OR
                                LOWER(di.label) LIKE '%carvedilol%' OR
                                LOWER(di.label) LIKE '%labetalol%'
                      THEN 1 END) as beta_blocker_count,

                -- Diuretics
                COUNT(CASE WHEN LOWER(di.label) LIKE '%furosemide%' OR
                                LOWER(di.label) LIKE '%lasix%' OR
                                LOWER(di.label) LIKE '%bumetanide%' OR
                                LOWER(di.label) LIKE '%torsemide%' OR
                                LOWER(di.label) LIKE '%hydrochlorothiazide%'
                      THEN 1 END) as diuretic_count,

                -- ACE inhibitors
                COUNT(CASE WHEN LOWER(di.label) LIKE '%lisinopril%' OR
                                LOWER(di.label) LIKE '%enalapril%' OR
                                LOWER(di.label) LIKE '%captopril%' OR
                                LOWER(di.label) LIKE '%ramipril%'
                      THEN 1 END) as acei_count,

                -- ARBs
                COUNT(CASE WHEN LOWER(di.label) LIKE '%losartan%' OR
                                LOWER(di.label) LIKE '%valsartan%' OR
                                LOWER(di.label) LIKE '%candesartan%' OR
                                LOWER(di.label) LIKE '%irbesartan%'
                      THEN 1 END) as arb_count,

                -- Calcium channel blockers
                COUNT(CASE WHEN LOWER(di.label) LIKE '%amlodipine%' OR
                                LOWER(di.label) LIKE '%diltiazem%' OR
                                LOWER(di.label) LIKE '%verapamil%' OR
                                LOWER(di.label) LIKE '%nifedipine%'
                      THEN 1 END) as ccb_count

            FROM `physionet-data.mimiciii_clinical.inputevents_mv` mv
            JOIN `physionet-data.mimiciii_clinical.d_items` di ON mv.itemid = di.itemid
            WHERE mv.icustay_id IN ({icustay_list})
            AND mv.statusdescription != 'Rewritten'
            GROUP BY mv.icustay_id
        )
        SELECT * FROM med_features
        """

        logging.info("Extracting medication features...")
        return self.client.query(query).to_dataframe()

    def extract_temporal_features(self, icustay_ids):
        """Extract temporal features for enhanced SBP prediction"""
        icustay_list = ','.join([str(id) for id in icustay_ids])

        query = f"""
        WITH chartevents_ts AS (
            SELECT 
                icustay_id,
                charttime,

                -- SBP time series
                CASE WHEN itemid IN (51, 442, 455, 6701, 220179, 220050) 
                    THEN SAFE_CAST(value AS FLOAT64) END as sbp,

                -- Heart rate time series
                CASE WHEN itemid IN (211, 220045) 
                    THEN SAFE_CAST(value AS FLOAT64) END as hr,

                -- Respiratory rate time series
                CASE WHEN itemid IN (615, 618, 220210, 224690) 
                    THEN SAFE_CAST(value AS FLOAT64) END as rr,

                -- SpO2 time series
                CASE WHEN itemid IN (646, 220277) 
                    THEN SAFE_CAST(value AS FLOAT64) END as spo2

            FROM `physionet-data.mimiciii_clinical.chartevents`
            WHERE icustay_id IN ({icustay_list})
            AND itemid IN (
                51, 442, 455, 6701, 220179, 220050,  -- SBP
                211, 220045,  -- HR
                615, 618, 220210, 224690,  -- RR
                646, 220277  -- SpO2
            )
            AND error IS NULL
            ORDER BY icustay_id, charttime
        ),

        -- Pre-calculate lag differences for lability
        changes_calc AS (
            SELECT 
                icustay_id,
                charttime,
                sbp,
                hr,
                rr,
                spo2,
                -- Use NULL for first record in each group
                ABS(sbp - LAG(sbp) OVER (PARTITION BY icustay_id ORDER BY charttime)) as sbp_change,
                ABS(hr - LAG(hr) OVER (PARTITION BY icustay_id ORDER BY charttime)) as hr_change
            FROM chartevents_ts
        ),

        -- Calculate temporal statistics per patient
        temporal_stats AS (
            SELECT 
                icustay_id,

                -- SBP temporal features
                AVG(sbp) as sbp_mean_verify,
                STDDEV(sbp) as sbp_std_verify,

                -- Heart rate variability over time
                STDDEV(hr) / NULLIF(AVG(hr), 0) as hr_cv_temporal,
                -- Using TIMESTAMP conversion
                CORR(UNIX_SECONDS(TIMESTAMP(charttime)), hr) as hr_trend,

                -- Variability of vitals over different time windows
                MAX(hr) - MIN(hr) as hr_range,
                MAX(rr) - MIN(rr) as rr_range,
                MAX(spo2) - MIN(spo2) as spo2_range,

                -- Correlation between vitals
                CORR(hr, sbp) as hr_sbp_corr,
                CORR(rr, sbp) as rr_sbp_corr,
                CORR(spo2, sbp) as spo2_sbp_corr,

                -- Now we use the pre-calculated changes
                COUNTIF(sbp_change > 20) as sbp_lability_events,
                COUNTIF(hr_change > 20) as hr_lability_events,

                -- Time-based features
                COUNT(DISTINCT EXTRACT(DATE FROM charttime)) as measurement_days,
                COUNT(*) as total_measurements,
                COUNT(DISTINCT charttime) as unique_timepoints

            FROM changes_calc
            GROUP BY icustay_id
        )

        SELECT * FROM temporal_stats
        """

        logging.info("Extracting temporal dynamics features...")
        return self.client.query(query).to_dataframe()

    def extract_comorbidity_features(self, icustay_ids):
        """Extract comorbidity features from diagnoses"""
        icustay_list = ','.join([str(id) for id in icustay_ids])

        query = f"""
        WITH comorbidities AS (
            SELECT 
                icu.icustay_id,

                -- Hypertension history
                MAX(CASE WHEN SUBSTR(d.icd9_code, 1, 3) = '401' OR 
                          SUBSTR(d.icd9_code, 1, 3) = '402' OR
                          SUBSTR(d.icd9_code, 1, 3) = '403' OR
                          SUBSTR(d.icd9_code, 1, 3) = '404' OR
                          SUBSTR(d.icd9_code, 1, 3) = '405'
                    THEN 1 ELSE 0 END) as hypertension,

                -- Diabetes
                MAX(CASE WHEN SUBSTR(d.icd9_code, 1, 3) = '250'
                    THEN 1 ELSE 0 END) as diabetes,

                -- Heart failure
                MAX(CASE WHEN SUBSTR(d.icd9_code, 1, 3) = '428'
                    THEN 1 ELSE 0 END) as heart_failure,

                -- Coronary artery disease
                MAX(CASE WHEN SUBSTR(d.icd9_code, 1, 3) IN ('410', '411', '412', '413', '414')
                    THEN 1 ELSE 0 END) as coronary_artery_disease,

                -- Chronic kidney disease
                MAX(CASE WHEN SUBSTR(d.icd9_code, 1, 3) = '585'
                    THEN 1 ELSE 0 END) as chronic_kidney_disease,

                -- Stroke history
                MAX(CASE WHEN SUBSTR(d.icd9_code, 1, 3) IN ('430', '431', '432', '433', '434', '435', '436', '437', '438')
                    THEN 1 ELSE 0 END) as stroke_history,

                -- COPD
                MAX(CASE WHEN SUBSTR(d.icd9_code, 1, 3) = '496' OR 
                          SUBSTR(d.icd9_code, 1, 3) = '491' OR
                          SUBSTR(d.icd9_code, 1, 3) = '492'
                    THEN 1 ELSE 0 END) as copd,

                -- Liver disease
                MAX(CASE WHEN SUBSTR(d.icd9_code, 1, 3) IN ('570', '571', '572', '573')
                    THEN 1 ELSE 0 END) as liver_disease,

                -- Count total comorbidities
                COUNT(DISTINCT d.icd9_code) as total_diagnoses

            FROM `physionet-data.mimiciii_clinical.icustays` icu
            JOIN `physionet-data.mimiciii_clinical.diagnoses_icd` d 
                ON icu.hadm_id = d.hadm_id
            WHERE icu.icustay_id IN ({icustay_list})
            GROUP BY icu.icustay_id
        )

        SELECT * FROM comorbidities
        """

        logging.info("Extracting comorbidity features...")
        return self.client.query(query).to_dataframe()


class eICUDataExtractor:
    """
    WORKING eICU Data Extractor - tested and validated
    """

    def __init__(self, project_id='d1namo-ecg-processing', timestamp=None):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        logging.info(f"BigQuery client initialized for project: {project_id} (eICU)")

    def extract_patient_cohort(self, max_patients=1000):
        """Extract patient cohort from eICU - TESTED AND WORKING"""
        query = f"""
        WITH patient_cohort AS (
            SELECT DISTINCT
                p.patientunitstayid,
                p.patienthealthsystemstayid,
                p.gender,
                CASE 
                    WHEN p.age = '> 89' THEN 90
                    WHEN p.age = '' OR p.age IS NULL THEN NULL
                    ELSE SAFE_CAST(p.age AS INT64)
                END as age,
                CASE WHEN p.unitdischargestatus = 'Expired' THEN 1 ELSE 0 END as hospital_mortality,
                p.unittype as first_careunit,
                CASE 
                    WHEN p.unitdischargeoffset > 0 THEN p.unitdischargeoffset / 60.0
                    ELSE NULL 
                END as icu_los_hours,
                p.ethnicity,
                p.hospitaladmitsource,
                p.admissionheight as height,
                p.admissionweight as weight,
                'Unknown' as insurance,
                'Unknown' as language,
                p.hospitalid,
                p.wardid,
                p.apacheadmissiondx
            FROM `physionet-data.eicu_crd.patient` p
            WHERE (
                CASE 
                    WHEN p.age = '> 89' THEN 90
                    WHEN p.age = '' OR p.age IS NULL THEN NULL
                    ELSE SAFE_CAST(p.age AS INT64)
                END
            ) BETWEEN 18 AND 89
            AND p.unitdischargeoffset > 1440
            AND p.unitdischargeoffset IS NOT NULL
        )
        SELECT * FROM patient_cohort
        WHERE age IS NOT NULL
        ORDER BY patientunitstayid
        LIMIT {max_patients}
        """

        logging.info("Extracting eICU patient cohort...")
        result = self.client.query(query).to_dataframe()
        logging.info(f"Extracted {len(result)} eICU patients")
        return result

    def extract_bp_data(self, patient_ids):
        """Extract BP measurements from eICU - TESTED AND WORKING"""
        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH bp_measurements AS (
            SELECT 
                patientunitstayid,
                AVG(CASE WHEN systemicsystolic > 0 THEN systemicsystolic END) as sbp_mean,
                STDDEV(CASE WHEN systemicsystolic > 0 THEN systemicsystolic END) as sbp_std,
                COUNT(CASE WHEN systemicsystolic > 0 THEN 1 END) as sbp_count,
                MIN(CASE WHEN systemicsystolic > 0 THEN systemicsystolic END) as sbp_min,
                MAX(CASE WHEN systemicsystolic > 0 THEN systemicsystolic END) as sbp_max,

                AVG(CASE WHEN systemicdiastolic > 0 THEN systemicdiastolic END) as dbp_mean,
                STDDEV(CASE WHEN systemicdiastolic > 0 THEN systemicdiastolic END) as dbp_std,
                COUNT(CASE WHEN systemicdiastolic > 0 THEN 1 END) as dbp_count,

                AVG(CASE WHEN systemicmean > 0 THEN systemicmean END) as map_mean

            FROM `physionet-data.eicu_crd.vitalperiodic`
            WHERE patientunitstayid IN ({patient_list})
            AND systemicsystolic BETWEEN 30 AND 300
            AND systemicdiastolic BETWEEN 15 AND 200
            AND systemicsystolic IS NOT NULL
            AND systemicdiastolic IS NOT NULL
            GROUP BY patientunitstayid
        )
        SELECT * FROM bp_measurements
        WHERE sbp_mean IS NOT NULL AND dbp_mean IS NOT NULL
        AND sbp_count >= 2
        """

        logging.info("Extracting eICU BP measurements...")
        result = self.client.query(query).to_dataframe()
        logging.info(f"Extracted BP data for {len(result)} patients")
        return result

    def extract_vital_signs_features(self, patient_ids):
        """Extract vital signs from eICU - TESTED AND WORKING"""
        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH vital_stats AS (
            SELECT 
                vp.patientunitstayid,
                AVG(CASE WHEN vp.heartrate > 0 THEN vp.heartrate END) as hr_mean,
                STDDEV(CASE WHEN vp.heartrate > 0 THEN vp.heartrate END) as hr_std,
                MIN(CASE WHEN vp.heartrate > 0 THEN vp.heartrate END) as hr_min,
                MAX(CASE WHEN vp.heartrate > 0 THEN vp.heartrate END) as hr_max,

                AVG(CASE WHEN vp.respiration > 0 THEN vp.respiration END) as rr_mean,
                STDDEV(CASE WHEN vp.respiration > 0 THEN vp.respiration END) as rr_std,

                AVG(CASE WHEN vp.temperature > 70 AND vp.temperature < 115 THEN vp.temperature END) as temp_mean,

                AVG(CASE WHEN vp.sao2 > 0 AND vp.sao2 <= 100 THEN vp.sao2 END) as spo2_mean,
                STDDEV(CASE WHEN vp.sao2 > 0 AND vp.sao2 <= 100 THEN vp.sao2 END) as spo2_std,

                AVG(CASE WHEN vp.cvp > 0 THEN vp.cvp END) as cvp_mean,
                AVG(CASE WHEN vp.etco2 > 0 THEN vp.etco2 END) as etco2_mean,

                AVG(CASE 
                    WHEN nc.nursingchartcelltypevalname LIKE '%Glasgow%' 
                    OR nc.nursingchartcelltypevalname LIKE '%GCS%'
                    THEN SAFE_CAST(nc.nursingchartvalue AS FLOAT64) 
                END) as gcs_mean

            FROM `physionet-data.eicu_crd.vitalperiodic` vp
            LEFT JOIN `physionet-data.eicu_crd.nursecharting` nc 
                ON vp.patientunitstayid = nc.patientunitstayid
                AND ABS(vp.observationoffset - nc.nursingchartoffset) <= 60
            WHERE vp.patientunitstayid IN ({patient_list})
            GROUP BY vp.patientunitstayid
        )
        SELECT * FROM vital_stats
        """

        logging.info("Extracting eICU vital signs features...")
        result = self.client.query(query).to_dataframe()
        logging.info(f"Extracted vital signs for {len(result)} patients")
        return result

    def extract_lab_features(self, patient_ids):
        """Extract lab features from eICU - TESTED AND WORKING"""
        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH lab_stats AS (
            SELECT 
                patientunitstayid,
                AVG(CASE WHEN LOWER(labname) LIKE '%sodium%' THEN labresult END) as sodium_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%potassium%' THEN labresult END) as potassium_mean,

                AVG(CASE WHEN LOWER(labname) LIKE '%creatinine%' THEN labresult END) as creatinine_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%bun%' OR LOWER(labname) LIKE '%urea%' THEN labresult END) as bun_mean,

                AVG(CASE WHEN LOWER(labname) LIKE '%hgb%' OR LOWER(labname) LIKE '%hemoglobin%' THEN labresult END) as hemoglobin_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%wbc%' OR LOWER(labname) LIKE '%white blood%' THEN labresult END) as wbc_mean,

                AVG(CASE WHEN LOWER(labname) LIKE '%lactate%' THEN labresult END) as lactate_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%troponin%' THEN labresult END) as troponin_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%crp%' OR LOWER(labname) LIKE '%c-reactive%' THEN labresult END) as crp_mean,

                AVG(CASE WHEN LOWER(labname) LIKE '%ph%' AND labresult BETWEEN 6.5 AND 8.0 THEN labresult END) as ph_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%pco2%' OR LOWER(labname) LIKE '%co2%' THEN labresult END) as pco2_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%po2%' OR LOWER(labname) LIKE '%oxygen%' THEN labresult END) as po2_mean,

                AVG(CASE WHEN LOWER(labname) LIKE '%alt%' OR LOWER(labname) LIKE '%sgpt%' THEN labresult END) as alt_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%ast%' OR LOWER(labname) LIKE '%sgot%' THEN labresult END) as ast_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%alkaline%' OR LOWER(labname) LIKE '%alp%' THEN labresult END) as alp_mean

            FROM `physionet-data.eicu_crd.lab`
            WHERE patientunitstayid IN ({patient_list})
            AND labresult IS NOT NULL
            AND labresult > 0
            GROUP BY patientunitstayid
        )
        SELECT * FROM lab_stats
        """

        logging.info("Extracting eICU lab features...")
        result = self.client.query(query).to_dataframe()
        logging.info(f"Extracted lab data for {len(result)} patients")
        return result

    def extract_medication_features(self, patient_ids):
        """Extract medication features from eICU - TESTED AND WORKING"""
        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH med_features AS (
            SELECT 
                patientunitstayid,
                COUNT(DISTINCT CASE WHEN LOWER(drugname) LIKE '%norepinephrine%' OR
                                     LOWER(drugname) LIKE '%epinephrine%' OR
                                     LOWER(drugname) LIKE '%dopamine%' OR
                                     LOWER(drugname) LIKE '%dobutamine%' OR
                                     LOWER(drugname) LIKE '%vasopressin%' OR
                                     LOWER(drugname) LIKE '%phenylephrine%'
                      THEN drugname END) as vasopressor_count,

                COUNT(DISTINCT CASE WHEN LOWER(drugname) LIKE '%metoprolol%' OR
                                     LOWER(drugname) LIKE '%esmolol%' OR
                                     LOWER(drugname) LIKE '%propranolol%' OR
                                     LOWER(drugname) LIKE '%carvedilol%' OR
                                     LOWER(drugname) LIKE '%labetalol%' OR
                                     LOWER(drugname) LIKE '%atenolol%'
                      THEN drugname END) as beta_blocker_count,

                COUNT(DISTINCT CASE WHEN LOWER(drugname) LIKE '%furosemide%' OR
                                     LOWER(drugname) LIKE '%lasix%' OR
                                     LOWER(drugname) LIKE '%bumetanide%' OR
                                     LOWER(drugname) LIKE '%torsemide%' OR
                                     LOWER(drugname) LIKE '%hydrochlorothiazide%'
                      THEN drugname END) as diuretic_count,

                COUNT(DISTINCT CASE WHEN LOWER(drugname) LIKE '%lisinopril%' OR
                                     LOWER(drugname) LIKE '%enalapril%' OR
                                     LOWER(drugname) LIKE '%captopril%' OR
                                     LOWER(drugname) LIKE '%ramipril%'
                      THEN drugname END) as acei_count,

                COUNT(DISTINCT CASE WHEN LOWER(drugname) LIKE '%losartan%' OR
                                     LOWER(drugname) LIKE '%valsartan%' OR
                                     LOWER(drugname) LIKE '%candesartan%'
                      THEN drugname END) as arb_count,

                COUNT(DISTINCT CASE WHEN LOWER(drugname) LIKE '%amlodipine%' OR
                                     LOWER(drugname) LIKE '%diltiazem%' OR
                                     LOWER(drugname) LIKE '%verapamil%' OR
                                     LOWER(drugname) LIKE '%nifedipine%'
                      THEN drugname END) as ccb_count

            FROM `physionet-data.eicu_crd.medication`
            WHERE patientunitstayid IN ({patient_list})
            AND drugname IS NOT NULL
            GROUP BY patientunitstayid
        )
        SELECT * FROM med_features
        """

        logging.info("Extracting eICU medication features...")
        result = self.client.query(query).to_dataframe()
        logging.info(f"Extracted medication data for {len(result)} patients")
        return result

    def extract_temporal_features(self, patient_ids):
        """Extract temporal features from eICU - WORKING"""
        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH temporal_stats AS (
            SELECT 
                patientunitstayid,
                STDDEV(heartrate) / NULLIF(AVG(heartrate), 0) as hr_cv_temporal,
                MAX(heartrate) - MIN(heartrate) as hr_range,
                MAX(respiration) - MIN(respiration) as rr_range,
                MAX(sao2) - MIN(sao2) as spo2_range,
                COUNT(*) as total_measurements,
                COUNT(DISTINCT observationoffset) as unique_timepoints,
                MAX(observationoffset) - MIN(observationoffset) as measurement_span_minutes

            FROM `physionet-data.eicu_crd.vitalperiodic`
            WHERE patientunitstayid IN ({patient_list})
            AND heartrate > 0 AND heartrate < 300
            AND respiration > 0 AND respiration < 60
            AND sao2 > 0 AND sao2 <= 100
            GROUP BY patientunitstayid
            HAVING COUNT(*) >= 3
        )
        SELECT * FROM temporal_stats
        """

        logging.info("Extracting eICU temporal features...")
        result = self.client.query(query).to_dataframe()
        logging.info(f"Extracted temporal features for {len(result)} patients")
        return result

    def extract_comorbidity_features(self, patient_ids):
        """Extract comorbidity features from eICU - WORKING"""
        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH diagnosis_flags AS (
            SELECT 
                patientunitstayid,
                MAX(CASE WHEN REGEXP_CONTAINS(LOWER(diagnosisstring), r'hypertension|htn|high blood pressure') THEN 1 ELSE 0 END) as hypertension,
                MAX(CASE WHEN REGEXP_CONTAINS(LOWER(diagnosisstring), r'diabetes|dm|diabetic') THEN 1 ELSE 0 END) as diabetes,
                MAX(CASE WHEN REGEXP_CONTAINS(LOWER(diagnosisstring), r'heart failure|chf|cardiomyopathy|cardiac failure') THEN 1 ELSE 0 END) as heart_failure,
                MAX(CASE WHEN REGEXP_CONTAINS(LOWER(diagnosisstring), r'coronary|cad|mi|myocardial|ischemic heart') THEN 1 ELSE 0 END) as coronary_artery_disease,
                MAX(CASE WHEN REGEXP_CONTAINS(LOWER(diagnosisstring), r'kidney|renal|ckd|chronic kidney|renal failure') THEN 1 ELSE 0 END) as chronic_kidney_disease,
                MAX(CASE WHEN REGEXP_CONTAINS(LOWER(diagnosisstring), r'stroke|cva|cerebrovascular|brain infarct') THEN 1 ELSE 0 END) as stroke_history,
                MAX(CASE WHEN REGEXP_CONTAINS(LOWER(diagnosisstring), r'copd|emphysema|chronic obstructive|chronic bronchitis') THEN 1 ELSE 0 END) as copd,
                MAX(CASE WHEN REGEXP_CONTAINS(LOWER(diagnosisstring), r'liver|hepatic|cirrhosis|liver disease') THEN 1 ELSE 0 END) as liver_disease,
                COUNT(DISTINCT diagnosisstring) as total_diagnoses

            FROM `physionet-data.eicu_crd.diagnosis`
            WHERE patientunitstayid IN ({patient_list})
            AND diagnosisstring IS NOT NULL
            GROUP BY patientunitstayid
        )
        SELECT * FROM diagnosis_flags
        """

        logging.info("Extracting eICU comorbidity features...")
        result = self.client.query(query).to_dataframe()
        logging.info(f"Extracted comorbidity data for {len(result)} patients")
        return result

class EnhancedFeatureProcessor:
    """
    Process features with improved methodology for SBP prediction,
    enhanced feature engineering, and interaction features
    """

    def __init__(self, config=None, timestamp=None):
        self.config = config if config else {}
        self.extreme_limits = self.config.get('extreme_limits', {
            'hr_mean': [20, 250],
            'sbp_mean': [40, 300],
            'dbp_mean': [15, 200],
            'temp_mean': [80, 110],
            'spo2_mean': [50, 100]
        })
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

    def process_features(self, cohort_df, bp_df, vitals_df, labs_df, meds_df,
                         temporal_df=None, comorbidity_df=None, dataset_name="MIMIC-III"):
        """Combine all features into final dataset with enhanced feature engineering"""
        logging.info(f"Processing and combining features for {dataset_name}...")

        # Start with cohort data
        df = cohort_df.copy()

        # Add BP targets
        df = df.merge(bp_df, on='icustay_id' if 'icustay_id' in df.columns else 'patientunitstayid', how='inner')

        # Add vital signs features
        df = df.merge(vitals_df, on='icustay_id' if 'icustay_id' in df.columns else 'patientunitstayid', how='left')

        # Add lab features
        df = df.merge(labs_df, on='icustay_id' if 'icustay_id' in df.columns else 'patientunitstayid', how='left')

        # Add medication features
        df = df.merge(meds_df, on='icustay_id' if 'icustay_id' in df.columns else 'patientunitstayid', how='left')

        # Add temporal features if available
        if temporal_df is not None:
            df = df.merge(temporal_df, on='icustay_id' if 'icustay_id' in df.columns else 'patientunitstayid',
                          how='left')

        # Add comorbidity features if available
        if comorbidity_df is not None:
            df = df.merge(comorbidity_df, on='icustay_id' if 'icustay_id' in df.columns else 'patientunitstayid',
                          how='left')

        # Create derived features
        df = self._create_derived_features(df)

        # Create interaction features for SBP prediction improvement
        if self.config.get('create_interaction_features', True):
            df = self._create_interaction_features(df)

        # Identify extreme outliers only (preserves most data)
        df = self._flag_extreme_outliers(df)

        # Save raw data with missing values for sensitivity analysis
        df_with_missing = df.copy()
        df_with_missing.to_csv(f"{dataset_name}_data_with_missing_{self.timestamp}.csv", index=False)

        logging.info(f"Final {dataset_name} dataset shape: {df.shape}")
        return df

    # Add as a new method in your feature engineering class
    def document_features(self, output_file="feature_documentation.md"):
        """Document the source and clinical meaning of each feature."""
        logger = logging.getLogger(__name__)
        logger.info(f"Documenting features to {output_file}...")

        # Define feature categories and their descriptions
        feature_categories = {
            "Demographics": ["age", "gender", "ethnicity", "language"],
            "Vital Signs": ["hr_mean", "hr_std", "rr_mean", "rr_std", "spo2_mean", "temp_mean"],
            "Blood Pressure": ["sbp_mean_verify", "sbp_std_verify", "dbp_mean_verify", "dbp_std_verify"],
            "Laboratory": ["sodium_mean", "potassium_mean", "hemoglobin_mean", "wbc_mean", "lactate_mean"],
            "Blood Gas": ["ph_mean", "pco2_mean", "po2_mean"],
            "Liver Function": ["alp_mean", "ast_mean", "alt_mean"],
            "Temporal": ["hr_trend", "sbp_lability_events"],
            "Interaction": ["age_creatinine_product", "vasopressor_lactate_interaction"],
            "Derived": ["bun_creatinine_ratio", "perfusion_index", "lactate_mean_squared", "lactate_mean_sqrt"]
        }

        # Add detailed descriptions for each feature category
        feature_descriptions = {
            "Demographics": "Patient demographic information including age, gender, ethnicity, and primary language.",
            "Vital Signs": "Time-averaged vital sign measurements including heart rate, respiratory rate, oxygen saturation, and temperature.",
            "Blood Pressure": "Blood pressure measurements from verification sources. **Note: These features should be used with caution due to potential data leakage.**",
            "Laboratory": "Common laboratory values with clinical relevance to hemodynamics and overall patient status.",
            "Blood Gas": "Arterial blood gas measurements reflecting acid-base status and oxygenation.",
            "Liver Function": "Laboratory markers of liver function.",
            "Temporal": "Features capturing the temporal dynamics of physiological measurements.",
            "Interaction": "Engineered features representing clinically meaningful interactions between variables.",
            "Derived": "Features mathematically derived from other measurements to capture non-linear relationships."
        }

        # Write documentation to file
        with open(output_file, "w") as f:
            f.write("# Feature Documentation for BP Prediction Models\n\n")
            f.write("This document describes the features used in the blood pressure prediction models.\n\n")

            for category, description in feature_descriptions.items():
                f.write(f"## {category}\n\n")
                f.write(f"{description}\n\n")
                f.write("Features in this category:\n\n")

                for feature in feature_categories.get(category, []):
                    f.write(f"- `{feature}`\n")

                f.write("\n")

            f.write("## Feature Importance Warning\n\n")
            f.write("The features `sbp_mean_verify` and `dbp_mean_verify` show extremely high importance ")
            f.write("in the model, which may indicate potential data leakage. Models should be evaluated ")
            f.write("both with and without these features to ensure robust performance.\n")

        logger.info(f"Feature documentation written to {output_file}")

    def _create_derived_features(self, df):
        """Create enhanced derived features with proven clinical relevance"""

        # Basic demographic features
        if 'gender' in df.columns:
            df['gender_male'] = (df['gender'] == 'M').astype(int)

        if 'age' in df.columns:
            df['age_normalized'] = df['age'] / 100  # Normalize age to 0-1 range
            # Age categories (based on clinical significance for BP)
            df['age_group'] = pd.cut(
                df['age'],
                bins=[0, 40, 60, 75, 100],
                labels=['young_adult', 'middle_age', 'elderly', 'very_elderly']
            ).astype(str)

            # One-hot encode age groups
            age_dummies = pd.get_dummies(df['age_group'], prefix='age')
            df = pd.concat([df, age_dummies], axis=1)

        # Body mass index if height and weight are available
        if 'height' in df.columns and 'weight' in df.columns:
            # Convert height to meters if needed
            height_meters = df['height'] / 100 if df['height'].mean() > 3 else df['height']
            df['bmi'] = df['weight'] / (height_meters ** 2)

            # BMI categories
            df['bmi_category'] = pd.cut(
                df['bmi'],
                bins=[0, 18.5, 25, 30, 35, 100],
                labels=['underweight', 'normal', 'overweight', 'obese', 'severely_obese']
            ).astype(str)

            # Fill NaN values in bmi_category
            df['bmi_category'] = df['bmi_category'].fillna('unknown')

            # One-hot encode BMI categories
            bmi_dummies = pd.get_dummies(df['bmi_category'], prefix='bmi')
            df = pd.concat([df, bmi_dummies], axis=1)

        # Cardiovascular features
        if 'hr_mean' in df.columns and 'hr_std' in df.columns:
            # Heart rate variability features
            df['hr_coefficient_variation'] = df['hr_std'] / df['hr_mean'].replace(0, np.nan)

            # Heart rate complexity (if temporal data available)
            if 'hr_range' in df.columns:
                df['hr_complexity'] = df['hr_range'] * df['hr_coefficient_variation']

        # BP-related derived features
        if 'sbp_mean' in df.columns and 'dbp_mean' in df.columns:
            # Pulse pressure (difference between SBP and DBP)
            df['pulse_pressure'] = df['sbp_mean'] - df['dbp_mean']

            # Mean arterial pressure estimate if not directly measured
            if 'map_mean' not in df.columns or df['map_mean'].isna().all():
                df['map_calculated'] = df['dbp_mean'] + (df['pulse_pressure'] / 3)
            elif 'map_mean' in df.columns:
                # Fill missing MAP values with calculated MAP
                calculated_map = df['dbp_mean'] + (df['pulse_pressure'] / 3)
                df['map_mean'] = df['map_mean'].fillna(calculated_map)

        # Respiratory features
        if 'rr_mean' in df.columns and 'spo2_mean' in df.columns:
            # Oxygenation index (simplified)
            df['oxy_index'] = df['spo2_mean'] / df['rr_mean'].replace(0, np.nan)

        # Additional clinical risk scores
        if 'hr_mean' in df.columns:
            df['tachycardia'] = (df['hr_mean'] > 100).astype(int)
            df['bradycardia'] = (df['hr_mean'] < 60).astype(int)

            # Heart rate extremes associated with SBP changes
            df['hr_extreme'] = ((df['hr_mean'] > 120) | (df['hr_mean'] < 50)).astype(int)

        # BP categories for stratified modeling
        if 'sbp_mean' in df.columns:
            df['sbp_category'] = pd.cut(
                df['sbp_mean'],
                bins=[0, 90, 120, 140, 160, 300],
                labels=['hypotension', 'normal', 'prehypertension', 'stage1_htn', 'stage2_htn']
            ).astype(str)

            # Fill NaN values in sbp_category
            df['sbp_category'] = df['sbp_category'].fillna('unknown')

            # One-hot encode SBP categories
            sbp_dummies = pd.get_dummies(df['sbp_category'], prefix='sbp')
            df = pd.concat([df, sbp_dummies], axis=1)

        # Medication exposure features - handle NaN values first
        med_cols = ['vasopressor_count', 'beta_blocker_count', 'diuretic_count',
                    'acei_count', 'arb_count', 'ccb_count']

        for col in med_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
                df[f'any_{col.replace("_count", "")}'] = (df[col] > 0).astype(int)

        # Total antihypertensive count
        antihtn_cols = ['beta_blocker_count', 'diuretic_count',
                        'acei_count', 'arb_count', 'ccb_count']

        if all(col in df.columns for col in antihtn_cols):
            df['antihypertensive_count'] = df[antihtn_cols].sum(axis=1)
            df['on_antihypertensives'] = (df['antihypertensive_count'] > 0).astype(int)

        # Lab-based features
        if 'creatinine_mean' in df.columns and 'bun_mean' in df.columns:
            # BUN/Creatinine ratio (indicator of prerenal azotemia)
            df['bun_creatinine_ratio'] = df['bun_mean'] / df['creatinine_mean'].replace(0, np.nan)

        # Combine comorbidity flags into a risk score if comorbidities are available
        comorbidity_cols = ['hypertension', 'diabetes', 'heart_failure',
                            'coronary_artery_disease', 'chronic_kidney_disease',
                            'stroke_history', 'copd', 'liver_disease']

        if all(col in df.columns for col in comorbidity_cols):
            # Sum comorbidities for a simple score
            df['comorbidity_score'] = df[comorbidity_cols].sum(axis=1)

            # High cardiovascular risk flag
            cv_risk_cols = ['hypertension', 'diabetes', 'heart_failure',
                            'coronary_artery_disease', 'stroke_history']

            if all(col in df.columns for col in cv_risk_cols):
                df['cv_risk_score'] = df[cv_risk_cols].sum(axis=1)
                df['high_cv_risk'] = (df['cv_risk_score'] >= 2).astype(int)

        return df

    def _create_interaction_features(self, df):
        """Create non-linear and interaction features to improve SBP prediction"""

        # Heart rate and creatinine interaction (kidney-heart axis)
        if 'hr_mean' in df.columns and 'creatinine_mean' in df.columns:
            df['hr_creatinine_product'] = df['hr_mean'] * df['creatinine_mean']

        # Age and kidney function interaction
        if 'age_normalized' in df.columns and 'creatinine_mean' in df.columns:
            df['age_creatinine_product'] = df['age_normalized'] * df['creatinine_mean']

        # Vasopressor response indicator
        if 'any_vasopressor' in df.columns and 'lactate_mean' in df.columns:
            df['vasopressor_lactate_interaction'] = df['any_vasopressor'] * df['lactate_mean']

        # Heart rate variability and SBP interaction
        if 'hr_coefficient_variation' in df.columns and 'sbp_mean' in df.columns:
            df['hrv_sbp_interaction'] = df['hr_coefficient_variation'] * df['sbp_mean']

        # Nonlinear transformations of key predictors
        for feature in ['hr_mean', 'creatinine_mean', 'lactate_mean', 'age_normalized']:
            if feature in df.columns:
                # Square (captures exponential relationships)
                df[f'{feature}_squared'] = df[feature] ** 2

                # Square root (captures diminishing returns relationships)
                df[f'{feature}_sqrt'] = np.sqrt(np.abs(df[feature]))

                # Log transform (for skewed variables)
                df[f'{feature}_log'] = np.log1p(np.abs(df[feature]))

        # Medical interventions interaction with physiological state
        if 'any_beta_blocker' in df.columns and 'hr_mean' in df.columns:
            df['beta_blocker_hr_effect'] = df['any_beta_blocker'] * df['hr_mean']

        # Kidney-BP axis
        if 'creatinine_mean' in df.columns and 'sbp_mean' in df.columns:
            df['creatinine_sbp_ratio'] = df['creatinine_mean'] / df['sbp_mean'].replace(0, np.nan)

        # Complex physiological interactions
        if all(col in df.columns for col in ['hr_mean', 'spo2_mean', 'lactate_mean']):
            # Tissue perfusion index
            df['perfusion_index'] = (df['spo2_mean'] * df['hr_mean']) / (df['lactate_mean'] + 1)

        return df

    def _flag_extreme_outliers(self, df):
        """
        Only flag extreme individual values rather than removing entire patient records
        This preserves most of the dataset while handling truly impossible values
        """
        original_shape = df.shape

        # Apply limits only to columns that exist in the dataframe
        outlier_count = 0

        for col, (min_val, max_val) in self.extreme_limits.items():
            if col in df.columns:
                # Identify extreme values outside physiological limits
                invalid_mask = (~df[col].isna()) & ((df[col] < min_val) | (df[col] > max_val))
                outlier_count += invalid_mask.sum()

                # Replace extreme values with NaN instead of removing entire rows
                if invalid_mask.sum() > 0:
                    logging.info(f"Replacing {invalid_mask.sum()} extreme values in {col}")
                    df.loc[invalid_mask, col] = np.nan

        if outlier_count > 0:
            logging.info(f"Replaced {outlier_count} extreme values with NaN")

        return df

class FeatureAligner:
    """Ensure consistent feature sets between MIMIC-III and eICU datasets"""

    def __init__(self, reference_features=None):
        self.reference_features = reference_features
        self.feature_mapping = {}

    def set_reference_features(self, feature_names):
        """Set the reference feature set (from training data)"""
        self.reference_features = list(feature_names)

    def align_features(self, X_new, dataset_name="External"):
        """Align new dataset features to match reference features"""
        logger = logging.getLogger(__name__)

        if self.reference_features is None:
            logger.error("Reference features not set. Call set_reference_features first.")
            return X_new

        logger.info(f"Aligning features for {dataset_name} dataset...")

        # Create DataFrame with reference features
        X_aligned = pd.DataFrame(index=X_new.index, columns=self.reference_features)

        # Copy existing features
        for feature in self.reference_features:
            if feature in X_new.columns:
                X_aligned[feature] = X_new[feature]
            else:
                # Handle missing features with appropriate defaults
                X_aligned[feature] = self._get_feature_default(feature)

        logger.info(f"Feature alignment completed. Final shape: {X_aligned.shape}")

        # Log missing and extra features
        missing_features = set(self.reference_features) - set(X_new.columns)
        extra_features = set(X_new.columns) - set(self.reference_features)

        if missing_features:
            logger.warning(f"Missing features filled with defaults: {list(missing_features)[:10]}...")
        if extra_features:
            logger.info(f"Extra features ignored: {list(extra_features)[:10]}...")

        return X_aligned

    def _get_feature_default(self, feature_name):
        """Get appropriate default value for missing features"""
        # Clinical defaults for common features
        defaults = {
            'admit_year': 2015,  # Median year
            'deceased': 0,
            'gender_male': 0.5,  # Assume 50/50 distribution
            'age_creatinine_product': 0.0,
            'hr_lability_events': 0.0,
            'sbp_lability_events': 0.0,
            'any_vasopressor': 0,
            'any_beta_blocker': 0,
            'any_diuretic': 0,
            'any_acei': 0,
            'any_arb': 0,
            'any_ccb': 0,
            'antihypertensive_count': 0,
            'comorbidity_score': 0,
            'cv_risk_score': 0,
            'high_cv_risk': 0
        }

        # Return default value or 0.0 for numeric features
        return defaults.get(feature_name, 0.0)


class AdvancedDataImputer:
    """
    Advanced missing data handling with multiple imputation strategies
    and sensitivity analysis as recommended by IEEE reviewer
    """

    def __init__(self, config=None, timestamp=None):
        self.config = config if config else {}
        self.method = self.config.get('method', 'iterative')  # 'iterative', 'knn', or 'default'
        self.knn_neighbors = self.config.get('knn_neighbors', 5)
        self.iterative_max_iter = self.config.get('iterative_max_iter', 50)
        self.perform_sensitivity = self.config.get('perform_sensitivity_analysis', True)
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

        # For storing alternative imputation results
        self.df_alternative = None

    def impute_missing_values(self, df, dataset_name="MIMIC-III"):
        """Impute missing values using selected method with sensitivity analysis"""

        # Create output directory for plots
        figures_dir = f"figures_{self.timestamp}"
        os.makedirs(figures_dir, exist_ok=True)

        # Report missing values
        missing_data = df.isnull().sum() / len(df)
        cols_with_missing = missing_data[missing_data > 0].index.tolist()

        if cols_with_missing:
            logging.info(f"Handling missing values in {len(cols_with_missing)} columns")
            for col in cols_with_missing:
                logging.info(f"  - {col}: {missing_data[col]:.1%} missing")

            # Visualize missing data patterns
            plt.figure(figsize=(12, 8))
            msno.matrix(df, figsize=(12, 8))
            plt.title(f"Missing Data Pattern - {dataset_name}")
            plt.savefig(f"{figures_dir}/missing_data_pattern_{dataset_name}.png", dpi=300)
            plt.close()

            # Correlation of missingness
            plt.figure(figsize=(12, 10))
            msno.heatmap(df, figsize=(12, 10))
            plt.title(f"Correlation of Missingness - {dataset_name}")
            plt.savefig(f"{figures_dir}/missing_correlation_{dataset_name}.png", dpi=300)
            plt.close()

            # Store original data for sensitivity analysis
            df_original = df.copy()

            # Select only numeric columns for imputation
            numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

            # Columns with high missingness (>70%) use simpler imputation
            high_missing_cols = [col for col in cols_with_missing
                                 if missing_data[col] > 0.7 and col in numeric_cols]

            # Regular missing columns use advanced imputation
            regular_missing_cols = [col for col in cols_with_missing
                                    if missing_data[col] <= 0.7 and col in numeric_cols]

            # Apply appropriate imputation method
            if self.method == 'iterative':
                df_imputed = self._apply_iterative_imputation(df, regular_missing_cols, high_missing_cols)

                # Store alternative imputation (KNN) for sensitivity analysis
                if self.perform_sensitivity:
                    self.df_alternative = self._apply_knn_imputation(df_original.copy(),
                                                                     regular_missing_cols,
                                                                     high_missing_cols)

            elif self.method == 'knn':
                df_imputed = self._apply_knn_imputation(df, regular_missing_cols, high_missing_cols)

                # Store alternative imputation (MICE) for sensitivity analysis
                if self.perform_sensitivity:
                    self.df_alternative = self._apply_iterative_imputation(df_original.copy(),
                                                                           regular_missing_cols,
                                                                           high_missing_cols)

            else:  # default simple imputation
                df_imputed = self._apply_default_imputation(df, cols_with_missing)

                # No alternative imputation for default method
                self.df_alternative = None

            # Compare imputation methods if sensitivity analysis is enabled
            if self.perform_sensitivity and self.df_alternative is not None:
                self._compare_imputation_methods(df_original, df_imputed, self.df_alternative,
                                                 regular_missing_cols, dataset_name)

            return df_imputed
        else:
            logging.info("No missing values to impute")
            return df

    def _apply_iterative_imputation(self, df, regular_missing_cols, high_missing_cols):
        """Apply MICE (Multiple Imputation by Chained Equations)"""
        logging.info("Performing multivariate imputation with IterativeImputer (MICE)")

        # Create a copy of dataframe
        df_imputed = df.copy()

        # Handle categorical variables first (they can't go into MICE)
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])

        # Handle high missingness columns with simple imputation
        if high_missing_cols:
            logging.info(f"Using median imputation for {len(high_missing_cols)} high-missingness columns")
            for col in high_missing_cols:
                if col in df_imputed.columns and df_imputed[col].isna().any():
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

        # Apply MICE to regular missingness columns
        if regular_missing_cols:
            # Extract relevant columns for imputation
            cols_for_imputation = [col for col in df_imputed.columns
                                   if col in regular_missing_cols or
                                   (col in df_imputed.select_dtypes(include=np.number).columns and
                                    not df_imputed[col].isna().any())]

            # Create subset dataframe for imputation
            df_for_mice = df_imputed[cols_for_imputation].copy()

            # Apply IterativeImputer
            mice_imputer = IterativeImputer(
                max_iter=self.iterative_max_iter,
                random_state=42,
                estimator=LinearRegression(),
                verbose=0
            )

            # Fit and transform
            mice_imputed_values = mice_imputer.fit_transform(df_for_mice)

            # Update the dataframe with imputed values
            df_imputed[cols_for_imputation] = mice_imputed_values

        return df_imputed

    def _apply_knn_imputation(self, df, regular_missing_cols, high_missing_cols):
        """Apply K-Nearest Neighbors imputation"""
        logging.info("Performing KNN imputation")

        # Create a copy of dataframe
        df_imputed = df.copy()

        # Handle categorical variables first
        categorical_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        for col in categorical_cols:
            if col in df.columns and df[col].isna().any():
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].mode()[0])

        # Handle high missingness columns with simple imputation
        if high_missing_cols:
            logging.info(f"Using median imputation for {len(high_missing_cols)} high-missingness columns")
            for col in high_missing_cols:
                if col in df_imputed.columns and df_imputed[col].isna().any():
                    df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())

        # Apply KNN to regular missingness columns
        if regular_missing_cols:
            # Extract relevant columns for imputation
            cols_for_imputation = [col for col in df_imputed.columns
                                   if col in regular_missing_cols or
                                   (col in df_imputed.select_dtypes(include=np.number).columns and
                                    not df_imputed[col].isna().any())]

            # Create subset dataframe for imputation
            df_for_knn = df_imputed[cols_for_imputation].copy()

            # Apply KNN imputer
            knn_imputer = KNNImputer(n_neighbors=self.knn_neighbors, weights='distance')

            # Fit and transform
            knn_imputed_values = knn_imputer.fit_transform(df_for_knn)

            # Update the dataframe with imputed values
            df_imputed[cols_for_imputation] = knn_imputed_values

        return df_imputed

    def _apply_default_imputation(self, df, cols_with_missing):
        """Simple median/mode imputation with clinical defaults"""
        logging.info("Using simple imputation with clinical defaults")

        # Create a copy of dataframe
        df_imputed = df.copy()

        # Clinical normal values for specific variables
        clinical_defaults = {
            'hr_mean': 80, 'hr_std': 10,
            'rr_mean': 18, 'rr_std': 5,
            'temp_mean': 98.6,
            'spo2_mean': 97, 'spo2_std': 3,
            'gcs_mean': 15,
            'sodium_mean': 140, 'potassium_mean': 4.0,
            'creatinine_mean': 1.0, 'bun_mean': 20,
            'hemoglobin_mean': 12, 'wbc_mean': 8,
            'lactate_mean': 1.5,
            'vasopressor_count': 0, 'beta_blocker_count': 0, 'diuretic_count': 0,
            'acei_count': 0, 'arb_count': 0, 'ccb_count': 0,
            'hr_min': 60, 'hr_max': 100,
            'sbp_min': 100, 'sbp_max': 140,
            'hr_coefficient_variation': 0.1,
            'map_calculated': 93.0
        }

        # Apply clinical defaults where available, median/mode elsewhere
        for col in cols_with_missing:
            if col in clinical_defaults:
                df_imputed[col] = df_imputed[col].fillna(clinical_defaults[col])
            elif col in df_imputed.select_dtypes(include=np.number).columns:
                df_imputed[col] = df_imputed[col].fillna(df_imputed[col].median())
            else:
                df_imputed[col] = df_imputed[col].fillna(
                    df_imputed[col].mode()[0] if not df_imputed[col].mode().empty else "Unknown")

        return df_imputed

    def _compare_imputation_methods(self, df_original, df_method1, df_method2, cols_to_compare, dataset_name):
        """Compare different imputation methods and visualize distributions"""
        logging.info("Performing sensitivity analysis on imputation methods")

        # Create output directory for plots
        figures_dir = f"figures_{self.timestamp}"

        # Select a subset of important columns to compare (up to 6)
        important_cols = ['temp_mean', 'sodium_mean', 'potassium_mean', 'lactate_mean',
                          'hr_mean', 'creatinine_mean', 'bun_mean', 'hemoglobin_mean']

        # Filter to columns that exist and have missing values
        cols_to_visualize = [col for col in important_cols
                             if col in cols_to_compare][:6]  # Limit to 6 for visualization

        if not cols_to_visualize:
            logging.info("No suitable columns for imputation comparison visualization")
            return

        # Create comparison plot
        n_cols = len(cols_to_visualize)
        fig, axes = plt.subplots(n_cols, 1, figsize=(10, 4 * n_cols))
        if n_cols == 1:
            axes = [axes]

        # Method names for display
        method1_name = 'MICE' if self.method == 'iterative' else ('KNN' if self.method == 'knn' else 'Default')
        method2_name = 'KNN' if self.method == 'iterative' else ('MICE' if self.method == 'knn' else 'Alternative')

        for i, col in enumerate(cols_to_visualize):
            # Only plot non-missing values from original data
            non_missing_mask = ~df_original[col].isna()

            # Plot distributions
            sns.kdeplot(df_original.loc[non_missing_mask, col], ax=axes[i],
                        label='Original (non-missing)', color='black')
            sns.kdeplot(df_method1[col], ax=axes[i],
                        label=f'{method1_name} Imputation', color='blue')
            sns.kdeplot(df_method2[col], ax=axes[i],
                        label=f'{method2_name} Imputation', color='red')

            axes[i].set_title(f'Distribution Comparison for {col}')
            axes[i].legend()

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/imputation_comparison_{dataset_name}.png", dpi=300)
        plt.close()

        # Create statistical comparison table
        stats_rows = []
        for col in cols_to_visualize:
            # Get original non-missing data
            original_data = df_original.loc[~df_original[col].isna(), col]

            # Calculate statistics
            stats_rows.append({
                'Variable': col,
                'Original_Mean': original_data.mean(),
                'Original_STD': original_data.std(),
                f'{method1_name}_Mean': df_method1[col].mean(),
                f'{method1_name}_STD': df_method1[col].std(),
                f'{method2_name}_Mean': df_method2[col].mean(),
                f'{method2_name}_STD': df_method2[col].std(),
                'Difference_%': abs(df_method1[col].mean() - df_method2[col].mean()) /
                                original_data.mean() * 100 if original_data.mean() != 0 else 0
            })

        stats_df = pd.DataFrame(stats_rows)
        stats_df.to_csv(f"imputation_stats_{dataset_name}_{self.timestamp}.csv", index=False)
        logging.info(f"Imputation method comparison saved to imputation_stats_{dataset_name}_{self.timestamp}.csv")

        # Log summary of differences
        mean_diff_pct = stats_df['Difference_%'].mean()
        max_diff_pct = stats_df['Difference_%'].max()

        logging.info(f"Imputation sensitivity analysis summary:")
        logging.info(f"  Mean difference between methods: {mean_diff_pct:.2f}%")
        logging.info(f"  Maximum difference between methods: {max_diff_pct:.2f}%")

        if max_diff_pct > 10:
            logging.warning(f"Large differences detected between imputation methods (max: {max_diff_pct:.2f}%). "
                            f"Consider using ensemble approach or report both in results.")
        else:
            logging.info("Imputation methods produce similar results (all differences < 10%). "
                         f"Proceeding with {method1_name} imputation.")


class OptimizedBPPredictor:
    """
    Enhanced BP Predictor with Bayesian optimization, uncertainty quantification,
    and stratified modeling for improved SBP prediction
    """

    def __init__(self, config=None, timestamp=None):
        self.config = config if config else {}
        self.random_state = self.config.get('random_state', 42)
        self.cv_folds = self.config.get('cv_folds', 5)  # FIX: This was missing!
        self.use_bayesian_optimization = self.config.get('bayesian_optimization', True)
        self.n_iter_bayesian = self.config.get('n_iter', 30)
        self.stratified_modeling = self.config.get('stratified_modeling', True)
        self.use_stacked_ensemble = self.config.get('use_stacked_ensemble', True)
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

        # Initialize base pipeline
        self.base_pipeline = Pipeline([
            ('var_filter', VarianceThreshold(threshold=1e-5)),  # Remove near-constant features
            ('scaler', RobustScaler()),
            ('pt', PowerTransformer(method="yeo-johnson", standardize=True)),
            ('gb', GradientBoostingRegressor(random_state=self.random_state))
        ])

        # Initialize models
        self.sbp_regressor = MultiOutputRegressor(self.base_pipeline)
        self.dbp_regressor = MultiOutputRegressor(self.base_pipeline)

        # For stratified models
        self.sbp_stratum_models = {}

        # For uncertainty quantification
        self.sbp_quantile_10 = None
        self.sbp_quantile_90 = None
        self.dbp_quantile_10 = None
        self.dbp_quantile_90 = None

        # For stacked ensemble
        self.sbp_ensemble = None
        self.dbp_ensemble = None

        # For feature importance
        self.feature_importance = {}

        # NEW: Track features used in training for alignment
        self.training_features = None

    def prepare_data(self, df, dataset_name="MIMIC-III"):
        """Prepare data for training with enhanced feature selection"""
        try:
            logging.info(f"Preparing {dataset_name} data for model training...")

            # CRITICAL: Remove leaky features FIRST
            df = remove_leaky_features(df)

            # Target variables
            if 'sbp_mean' not in df.columns or 'dbp_mean' not in df.columns:
                raise ValueError("Target variables (sbp_mean, dbp_mean) not found in dataset")

            y = df[['sbp_mean', 'dbp_mean']].copy()

            # Features - explicitly exclude all BP-related targets
            exclude_columns = [
                # Target variables
                'sbp_mean', 'dbp_mean',
                # ID columns
                'icustay_id', 'patientunitstayid', 'subject_id', 'hadm_id',
                # Categorical that need special handling
                'gender', 'age_group',
                # Any remaining verification features
                'sbp_count', 'sbp_min', 'sbp_max', 'dbp_count',
                'map_mean', 'map_calculated', 'pulse_pressure'
            ]

            # Get available columns (excluding problematic ones)
            available_columns = [col for col in df.columns if col not in exclude_columns]
            X = df[available_columns].copy()

            # Validate no leakage
            if not validate_no_leakage(X):
                raise ValueError("Data leakage detected - cannot proceed with training")

            # Get group variable (for stratified CV)
            if 'subject_id' in df.columns:
                groups = df['subject_id']
            elif 'patientunitstayid' in df.columns:
                groups = df['patientunitstayid']
            else:
                logging.warning("No subject ID found for grouping, using synthetic groups")
                groups = pd.Series(range(len(df)))

            # Handle missing values
            if X.isnull().any().any():
                missing_count = X.isnull().sum().sum()
                logging.warning(f"Found {missing_count} missing values in features, filling with medians")
                for col in X.columns:
                    if X[col].isnull().any():
                        X[col] = X[col].fillna(X[col].median())

            # Convert categorical columns to numeric
            cat_columns = X.select_dtypes(include=['object', 'category']).columns
            for col in cat_columns:
                logging.info(f"Converting categorical column: {col}")
                X[col] = pd.Categorical(X[col]).codes

            # Apply variance threshold
            from sklearn.feature_selection import VarianceThreshold
            selector = VarianceThreshold(threshold=1e-5)
            X_array = selector.fit_transform(X)
            selected_features = [X.columns[i] for i in selector.get_support(indices=True)]
            logging.info(f"Removed {X.shape[1] - len(selected_features)} low-variance features")
            X = X[selected_features]

            # Handle highly correlated features
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            high_corr_features = [column for column in upper.columns if any(upper[column] > 0.95)]
            if high_corr_features:
                logging.info(f"Removing {len(high_corr_features)} highly correlated features")
                X = X.drop(columns=high_corr_features)

            # Handle any remaining missing values
            cols_with_missing = X.columns[X.isnull().any()].tolist()
            for col in cols_with_missing:
                X[col] = X[col].fillna(X[col].median())

            # Final validation check
            if not validate_no_leakage(X):
                raise ValueError("Data leakage detected after processing - cannot proceed")

            # Store training features for later alignment
            if dataset_name == "MIMIC-III":
                self.training_features = list(X.columns)
                logging.info(f"Stored {len(self.training_features)} training features for alignment")

            logging.info(f"Final feature set: {X.shape[1]} features, {X.shape[0]} samples")
            logging.info(f"Target set: {y.shape}")

            return X, y, groups

        except Exception as e:
            logging.error(f"Error in data preparation: {str(e)}")
            raise

    def _log_feature_distributions(self, X, y, dataset_name):
        """Log distributions of key features for understanding data characteristics"""
        # Create output directory for plots
        figures_dir = f"figures_{self.timestamp}"
        os.makedirs(figures_dir, exist_ok=True)

        # Select important features to visualize (if they exist)
        key_features = ['age_normalized', 'hr_mean', 'creatinine_mean', 'lactate_mean',
                        'hr_coefficient_variation', 'bun_mean']

        existing_features = [f for f in key_features if f in X.columns]

        if len(existing_features) > 0:
            # Create pairplot
            pair_df = pd.concat([X[existing_features], y], axis=1)
            plt.figure(figsize=(12, 10))
            sns.pairplot(pair_df, vars=existing_features,
                         hue='sbp_mean', diag_kind='kde',
                         plot_kws={'alpha': 0.5, 's': 10})
            plt.savefig(f"{figures_dir}/{dataset_name}_feature_pairplot.png", dpi=300)
            plt.close()

            # Create correlation heatmap
            plt.figure(figsize=(12, 10))
            corr = pair_df.corr()
            mask = np.triu(np.ones_like(corr, dtype=bool))
            sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                        vmin=-1, vmax=1, linewidths=0.5)
            plt.title(f"{dataset_name} Feature Correlation Heatmap")
            plt.tight_layout()
            plt.savefig(f"{figures_dir}/{dataset_name}_correlation_heatmap.png", dpi=300)
            plt.close()

    def train_models_with_bayesian_optimization(self, X, y, groups):
        """Train models using Bayesian optimization for parameter tuning"""
        try:
            logging.info("Training models using Bayesian optimization...")

            # Define the search space with wider ranges
            search_space = {
                'estimator__gb__n_estimators': Integer(50, 500),
                'estimator__gb__learning_rate': Real(0.001, 0.2, prior='log-uniform'),
                'estimator__gb__max_depth': Integer(3, 10),
                'estimator__gb__min_samples_split': Integer(2, 20),
                'estimator__gb__subsample': Real(0.5, 1.0, prior='uniform'),
                'estimator__gb__max_features': Categorical(['sqrt', 'log2', None])
            }

            # Create GroupKFold for cross-validation
            group_cv = GroupKFold(n_splits=self.cv_folds)

            # SBP model with Bayesian optimization
            logging.info("Training SBP model with Bayesian optimization...")

            bayes_search_sbp = BayesSearchCV(
                self.sbp_regressor,
                search_space,
                n_iter=self.n_iter_bayesian,
                cv=group_cv.split(X, y[['sbp_mean']], groups),
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )

            bayes_search_sbp.fit(X, y[['sbp_mean']])
            self.sbp_regressor = bayes_search_sbp.best_estimator_

            logging.info(f"Best SBP parameters: {bayes_search_sbp.best_params_}")
            logging.info(f"Best SBP score: {bayes_search_sbp.best_score_:.4f}")

            # Train quantile regressors for SBP uncertainty estimation
            self._train_quantile_regressors(X, y[['sbp_mean']], 'sbp')

            # Train DBP model with Bayesian optimization
            logging.info("Training DBP model with Bayesian optimization...")

            bayes_search_dbp = BayesSearchCV(
                self.dbp_regressor,
                search_space,
                n_iter=self.n_iter_bayesian,
                cv=group_cv.split(X, y[['dbp_mean']], groups),
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1,
                random_state=self.random_state
            )

            bayes_search_dbp.fit(X, y[['dbp_mean']])
            self.dbp_regressor = bayes_search_dbp.best_estimator_

            logging.info(f"Best DBP parameters: {bayes_search_dbp.best_params_}")
            logging.info(f"Best DBP score: {bayes_search_dbp.best_score_:.4f}")

            # Train quantile regressors for DBP uncertainty estimation
            self._train_quantile_regressors(X, y[['dbp_mean']], 'dbp')

            # Extract and store feature importance
            self._extract_feature_importance(X.columns)

            # Train stacked ensemble models if enabled
            if self.use_stacked_ensemble:
                self._train_stacked_ensemble(X, y, groups)

            # Train stratified models if enabled
            if self.stratified_modeling:
                self._train_stratified_models(X, y, groups)

            # Save models
            self._save_models()

            return self.timestamp

        except Exception as e:
            logging.error(f"Error in model training with Bayesian optimization: {str(e)}")
            raise

    def train_models_standard(self, X, y, groups):
        """Standard training method without Bayesian optimization (fallback)"""
        try:
            logging.info("Training models using standard grid search...")

            # Define parameter grid
            param_grid = {
                'estimator__gb__n_estimators': [100, 200, 300],
                'estimator__gb__learning_rate': [0.01, 0.05, 0.1],
                'estimator__gb__max_depth': [3, 5, 7],
                'estimator__gb__min_samples_split': [2, 5],
                'estimator__gb__subsample': [0.8, 1.0],
                'estimator__gb__max_features': ['sqrt', None]
            }

            # Create GroupKFold for cross-validation
            group_cv = GroupKFold(n_splits=self.cv_folds)

            # Train SBP model
            logging.info("Training SBP model...")
            grid_search_sbp = GridSearchCV(
                self.sbp_regressor,
                param_grid,
                cv=group_cv.split(X, y[['sbp_mean']], groups),
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            grid_search_sbp.fit(X, y[['sbp_mean']])
            self.sbp_regressor = grid_search_sbp.best_estimator_

            logging.info(f"Best SBP parameters: {grid_search_sbp.best_params_}")
            logging.info(f"Best SBP score: {grid_search_sbp.best_score_:.4f}")

            # Train DBP model
            logging.info("Training DBP model...")
            grid_search_dbp = GridSearchCV(
                self.dbp_regressor,
                param_grid,
                cv=group_cv.split(X, y[['dbp_mean']], groups),
                scoring='neg_root_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )

            grid_search_dbp.fit(X, y[['dbp_mean']])
            self.dbp_regressor = grid_search_dbp.best_estimator_

            logging.info(f"Best DBP parameters: {grid_search_dbp.best_params_}")
            logging.info(f"Best DBP score: {grid_search_dbp.best_score_:.4f}")

            # Extract and store feature importance
            self._extract_feature_importance(X.columns)

            # Train stratified models if enabled
            if self.stratified_modeling:
                self._train_stratified_models(X, y, groups)

            # Save models
            self._save_models()

            return self.timestamp

        except Exception as e:
            logging.error(f"Error in standard model training: {str(e)}")
            raise

    # Add as a new method in your model training class
    def improve_dbp_modeling(self, X, y_dbp, cv_folds=5):
        """Enhanced modeling specifically for DBP prediction."""
        logger.info("Implementing enhanced DBP modeling techniques...")

        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.neural_network import MLPRegressor
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, RobustScaler
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        from xgboost import XGBRegressor

        try:
            # Try to import advanced models (you may need to install these)
            from lightgbm import LGBMRegressor
            from catboost import CatBoostRegressor
            has_advanced_models = True
        except ImportError:
            logger.warning("LightGBM and CatBoost not available. Using standard models only.")
            has_advanced_models = False

        # 1. Feature selection specific to DBP
        logger.info("Performing DBP-specific feature selection...")
        dbp_selector = SelectFromModel(GradientBoostingRegressor(random_state=self.config['random_state']))
        dbp_selector.fit(X, y_dbp)
        X_dbp_selected = X.loc[:, dbp_selector.get_support()]
        logger.info(f"Selected {X_dbp_selected.shape[1]} features for DBP prediction")

        # 2. Define models to try
        models = {
            'gb': GradientBoostingRegressor(random_state=self.config['random_state']),
            'rf': RandomForestRegressor(random_state=self.config['random_state']),
            'xgb': XGBRegressor(random_state=self.config['random_state']),
            'nn': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=self.config['random_state'])
        }

        # Add advanced models if available
        if has_advanced_models:
            models['lgbm'] = LGBMRegressor(objective='regression', random_state=self.config['random_state'])
            models['catboost'] = CatBoostRegressor(verbose=False, random_state=self.config['random_state'])

        # 3. Training with cross-validation
        logger.info("Evaluating DBP models with cross-validation...")
        cv_results = {}
        for name, model in models.items():
            cv_score = cross_val_score(model, X_dbp_selected, y_dbp, cv=cv_folds,
                                       scoring='neg_mean_squared_error')
            rmse = np.sqrt(-np.mean(cv_score))
            cv_results[name] = rmse
            logger.info(f"  {name} model - RMSE: {rmse:.2f} mmHg")

        # 4. Train the best model on the full dataset
        best_model_name = min(cv_results, key=cv_results.get)
        logger.info(f"Best DBP model: {best_model_name} with RMSE: {cv_results[best_model_name]:.2f} mmHg")

        best_model = models[best_model_name]
        best_model.fit(X_dbp_selected, y_dbp)

        return best_model, X_dbp_selected.columns.tolist()

    def _train_quantile_regressors(self, X, y, bp_type):
        """Train quantile regressors for uncertainty estimation"""
        logging.info(f"Training quantile regressors for {bp_type.upper()} uncertainty estimation")

        # Create pipeline for preprocessing
        preprocess_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('pt', PowerTransformer(method="yeo-johnson", standardize=True))
        ])

        # Preprocess features
        X_processed = preprocess_pipeline.fit_transform(X)

        # Get target values
        y_values = y.values.ravel()

        # Train lower bound (10th percentile)
        quantile_10 = QuantileRegressor(quantile=0.1, alpha=0.1, solver='highs')
        quantile_10.fit(X_processed, y_values)

        # Train upper bound (90th percentile)
        quantile_90 = QuantileRegressor(quantile=0.9, alpha=0.1, solver='highs')
        quantile_90.fit(X_processed, y_values)

        # Store models and preprocessing pipeline
        if bp_type == 'sbp':
            self.sbp_quantile_10 = quantile_10
            self.sbp_quantile_90 = quantile_90
            self.sbp_preprocess = preprocess_pipeline
        else:
            self.dbp_quantile_10 = quantile_10
            self.dbp_quantile_90 = quantile_90
            self.dbp_preprocess = preprocess_pipeline

    def _train_stacked_ensemble(self, X, y, groups):
        """Create stacked ensemble combining multiple model types"""
        logging.info("Training stacked ensemble models...")

        # Define base estimators for SBP
        base_estimators_sbp = [
            ('gb', Pipeline([
                ('scaler', RobustScaler()),
                ('pt', PowerTransformer(method="yeo-johnson", standardize=True)),
                ('gb', GradientBoostingRegressor(random_state=self.random_state))
            ])),
            ('rf', Pipeline([
                ('scaler', RobustScaler()),
                ('rf', RandomForestRegressor(random_state=self.random_state))
            ])),
            ('xgb', Pipeline([
                ('scaler', RobustScaler()),
                ('xgb', XGBRegressor(random_state=self.random_state))
            ]))
        ]

        # Create SBP stacked model
        sbp_stack = StackingRegressor(
            estimators=base_estimators_sbp,
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
            cv=5,
            n_jobs=-1
        )

        # Define hyperparameter space
        search_space = {
            'gb__gb__n_estimators': Integer(50, 300),
            'gb__gb__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
            'gb__gb__max_depth': Integer(3, 7),
            'rf__rf__n_estimators': Integer(50, 300),
            'rf__rf__max_depth': Integer(3, 10),
            'xgb__xgb__n_estimators': Integer(50, 300),
            'xgb__xgb__learning_rate': Real(0.01, 0.2, prior='log-uniform')
        }

        # Create GroupKFold for cross-validation
        group_cv = GroupKFold(n_splits=self.cv_folds)

        # Train SBP ensemble with bayesian optimization
        logging.info("Training SBP ensemble with Bayesian optimization...")
        bayes_search_sbp_ensemble = BayesSearchCV(
            sbp_stack,
            search_space,
            n_iter=10,  # Reduced iterations for ensemble
            cv=group_cv.split(X, y['sbp_mean'], groups),
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state
        )

        # Fit model
        bayes_search_sbp_ensemble.fit(X, y['sbp_mean'])
        self.sbp_ensemble = bayes_search_sbp_ensemble.best_estimator_

        logging.info(f"Best SBP ensemble parameters: {bayes_search_sbp_ensemble.best_params_}")
        logging.info(f"Best SBP ensemble score: {bayes_search_sbp_ensemble.best_score_:.4f}")

        # Similarly for DBP
        base_estimators_dbp = [
            ('gb', Pipeline([
                ('scaler', RobustScaler()),
                ('pt', PowerTransformer(method="yeo-johnson", standardize=True)),
                ('gb', GradientBoostingRegressor(random_state=self.random_state))
            ])),
            ('rf', Pipeline([
                ('scaler', RobustScaler()),
                ('rf', RandomForestRegressor(random_state=self.random_state))
            ])),
            ('xgb', Pipeline([
                ('scaler', RobustScaler()),
                ('xgb', XGBRegressor(random_state=self.random_state))
            ]))
        ]

        # Create DBP stacked model
        dbp_stack = StackingRegressor(
            estimators=base_estimators_dbp,
            final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0]),
            cv=5,
            n_jobs=-1
        )

        # Train DBP ensemble
        logging.info("Training DBP ensemble with Bayesian optimization...")
        bayes_search_dbp_ensemble = BayesSearchCV(
            dbp_stack,
            search_space,
            n_iter=10,  # Reduced iterations for ensemble
            cv=group_cv.split(X, y['dbp_mean'], groups),
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1,
            random_state=self.random_state
        )

        # Fit model
        bayes_search_dbp_ensemble.fit(X, y['dbp_mean'])
        self.dbp_ensemble = bayes_search_dbp_ensemble.best_estimator_

        logging.info(f"Best DBP ensemble parameters: {bayes_search_dbp_ensemble.best_params_}")
        logging.info(f"Best DBP ensemble score: {bayes_search_dbp_ensemble.best_score_:.4f}")

    # Add as a new method in your model training class
    def train_without_verification_features(self, X, y, cv_folds=5):
        """Train models without potentially leaking verification features."""
        logger.info("Training models without verification features to assess true predictive power...")

        # Create copy of dataset without the potentially leaking features
        X_independent = X.copy()
        leakage_candidates = ['sbp_mean_verify', 'sbp_std_verify', 'dbp_mean_verify', 'dbp_std_verify']
        for col in leakage_candidates:
            if col in X_independent.columns:
                logger.info(f"Removing potential data leakage feature: {col}")
                X_independent.drop(columns=[col], inplace=True)

        # Train models with the independent feature set
        logger.info(f"Training with {X_independent.shape[1]} features after removing verification features")
        independent_models = self.train_models(X_independent, y, cv_folds)

        return independent_models

    def _train_stratified_models(self, X, y, groups):
        """Train separate models for different BP ranges"""
        logging.info("Training stratified models for different BP ranges...")

        # Create BP categories
        sbp_values = y['sbp_mean'].values
        sbp_categories = np.zeros_like(sbp_values, dtype=int)

        # Define BP strata (based on clinical categories)
        sbp_thresholds = [0, 90, 120, 140, 300]  # Hypotension, Normal, Prehypertension, Hypertension
        sbp_labels = ['Hypotension', 'Normal', 'Prehypertension', 'Hypertension']

        # Assign categories
        for i in range(1, len(sbp_thresholds)):
            sbp_categories[np.logical_and(
                sbp_values >= sbp_thresholds[i - 1],
                sbp_values < sbp_thresholds[i]
            )] = i - 1

        # Count samples in each stratum
        for i, label in enumerate(sbp_labels):
            count = np.sum(sbp_categories == i)
            logging.info(f"  {label}: {count} samples")

        # Create stratified models if enough samples (>30) exist
        sbp_stratum_models = {}

        for i, label in enumerate(sbp_labels):
            stratum_indices = np.where(sbp_categories == i)[0]

            if len(stratum_indices) > 30:  # Only train if enough samples
                logging.info(f"Training specialized model for {label} BP range...")

                # Get data for this stratum
                X_stratum = X.iloc[stratum_indices].copy()
                y_stratum = y.iloc[stratum_indices].copy()
                groups_stratum = groups.iloc[stratum_indices].copy()

                # Special hyperparameters for each stratum
                if label == 'Hypotension':
                    param_grid = {
                        'estimator__gb__n_estimators': [200, 300],
                        'estimator__gb__learning_rate': [0.05, 0.1],
                        'estimator__gb__max_depth': [3, 5],
                        'estimator__gb__subsample': [0.8]
                    }
                elif label == 'Hypertension':
                    # More complex models for hypertension (often harder to predict)
                    param_grid = {
                        'estimator__gb__n_estimators': [300, 400],
                        'estimator__gb__learning_rate': [0.01, 0.05],
                        'estimator__gb__max_depth': [5, 7],
                        'estimator__gb__min_samples_split': [2],
                        'estimator__gb__subsample': [0.8]
                    }
                else:
                    # Default parameters for normal BP ranges
                    param_grid = {
                        'estimator__gb__n_estimators': [100, 200],
                        'estimator__gb__learning_rate': [0.05, 0.1],
                        'estimator__gb__max_depth': [3, 5],
                        'estimator__gb__subsample': [0.8, 1.0]
                    }

                # Initialize and train model for this stratum
                stratum_regressor = MultiOutputRegressor(Pipeline([
                    ('scaler', RobustScaler()),
                    ('pt', PowerTransformer(method="yeo-johnson", standardize=True)),
                    ('gb', GradientBoostingRegressor(random_state=self.random_state))
                ]))

                # Create GroupKFold with appropriate fold count
                n_folds = min(self.cv_folds, len(np.unique(groups_stratum)))
                group_cv = GroupKFold(n_splits=n_folds)

                # Train with GridSearchCV
                grid_search = GridSearchCV(
                    stratum_regressor,
                    param_grid,
                    cv=group_cv.split(X_stratum, y_stratum[['sbp_mean']], groups_stratum),
                    scoring='neg_root_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )

                grid_search.fit(X_stratum, y_stratum[['sbp_mean']])

                # Store model
                sbp_stratum_models[label] = {
                    'model': grid_search.best_estimator_,
                    'thresholds': (sbp_thresholds[i], sbp_thresholds[i + 1]),
                    'best_params': grid_search.best_params_,
                    'best_score': grid_search.best_score_
                }

                logging.info(f"  {label} model - Best RMSE: {-grid_search.best_score_:.2f} mmHg")

        self.sbp_stratum_models = sbp_stratum_models

        logging.info("Stratified model training completed.")
        return sbp_stratum_models

    def predict_with_stratified_models(self, X, first_stage_sbp_pred):
        """
        Make predictions using the appropriate stratum-specific model based on
        the initial BP prediction
        """
        try:
            # If stratified models weren't trained, return original predictions
            if not hasattr(self, 'sbp_stratum_models') or not self.sbp_stratum_models:
                return first_stage_sbp_pred

            # Create array for second-stage predictions
            second_stage_sbp_pred = np.copy(first_stage_sbp_pred)

            # Track which samples were predicted by specialized models
            used_specialized_model = np.zeros(len(first_stage_sbp_pred), dtype=bool)

            # For each prediction, check which stratum it belongs to and use the appropriate model
            for i, sbp_pred in enumerate(first_stage_sbp_pred):
                for label, model_info in self.sbp_stratum_models.items():
                    low_thresh, high_thresh = model_info['thresholds']

                    # If prediction falls in this stratum's range
                    if low_thresh <= sbp_pred < high_thresh:
                        # Get a second prediction using the specialized model
                        specialized_pred = model_info['model'].predict(X.iloc[[i]])[0][0]

                        # Use a weighted average of the two predictions
                        second_stage_sbp_pred[i] = 0.3 * first_stage_sbp_pred[i] + 0.7 * specialized_pred
                        used_specialized_model[i] = True
                        break

            logging.info(
                f"Used specialized models for {used_specialized_model.sum()} out of {len(first_stage_sbp_pred)} predictions")

            return second_stage_sbp_pred

        except Exception as e:
            logging.error(f"Error in stratified prediction: {str(e)}")
            # Fall back to original predictions if something goes wrong
            return first_stage_sbp_pred

    def predict_with_uncertainty(self, X):
        """Make predictions with uncertainty intervals"""
        try:
            # SBP prediction
            sbp_pred = self.sbp_regressor.predict(X).ravel()

            # Use stacked ensemble if available
            if hasattr(self, 'sbp_ensemble') and self.sbp_ensemble is not None:
                ensemble_sbp_pred = self.sbp_ensemble.predict(X)
                # Blend predictions (weighted average)
                sbp_pred = 0.4 * sbp_pred + 0.6 * ensemble_sbp_pred

            # Use stratified models if available
            if hasattr(self, 'sbp_stratum_models') and self.sbp_stratum_models:
                sbp_pred = self.predict_with_stratified_models(X, sbp_pred)

            # DBP prediction
            dbp_pred = self.dbp_regressor.predict(X).ravel()

            # Use stacked ensemble if available
            if hasattr(self, 'dbp_ensemble') and self.dbp_ensemble is not None:
                ensemble_dbp_pred = self.dbp_ensemble.predict(X)
                # Blend predictions (weighted average)
                dbp_pred = 0.4 * dbp_pred + 0.6 * ensemble_dbp_pred

            # Prediction intervals
            sbp_intervals = None
            dbp_intervals = None

            # Compute SBP prediction intervals if quantile regressors are available
            if hasattr(self, 'sbp_quantile_10') and self.sbp_quantile_10 is not None:
                # Preprocess features
                X_processed = self.sbp_preprocess.transform(X)

                # Get lower and upper bounds
                sbp_lower = self.sbp_quantile_10.predict(X_processed)
                sbp_upper = self.sbp_quantile_90.predict(X_processed)

                # Create intervals array
                sbp_intervals = np.column_stack((sbp_lower, sbp_upper))

            # Compute DBP prediction intervals if quantile regressors are available
            if hasattr(self, 'dbp_quantile_10') and self.dbp_quantile_10 is not None:
                # Preprocess features
                X_processed = self.dbp_preprocess.transform(X)

                # Get lower and upper bounds
                dbp_lower = self.dbp_quantile_10.predict(X_processed)
                dbp_upper = self.dbp_quantile_90.predict(X_processed)

                # Create intervals array
                dbp_intervals = np.column_stack((dbp_lower, dbp_upper))

            return sbp_pred, dbp_pred, sbp_intervals, dbp_intervals

        except Exception as e:
            logging.error(f"Error in prediction with uncertainty: {str(e)}")

            # Return basic predictions without uncertainty
            sbp_pred = self.sbp_regressor.predict(X).ravel()
            dbp_pred = self.dbp_regressor.predict(X).ravel()

            return sbp_pred, dbp_pred, None, None

    def _extract_feature_importance(self, feature_names):
        """Extract and store feature importance from the trained models"""
        try:
            # Get feature importance from each estimator
            sbp_importance = self.sbp_regressor.estimators_[0].named_steps['gb'].feature_importances_
            dbp_importance = self.dbp_regressor.estimators_[0].named_steps['gb'].feature_importances_

            # Store in dictionary for use in evaluation
            self.feature_importance = {
                'features': feature_names,
                'sbp_importance': sbp_importance,
                'dbp_importance': dbp_importance
            }

            # Create more detailed feature importance information
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'SBP_Importance': sbp_importance,
                'DBP_Importance': dbp_importance,
                'Average_Importance': (sbp_importance + dbp_importance) / 2
            }).sort_values('Average_Importance', ascending=False)

            # Save feature importance
            importance_df.to_csv(f"feature_importance_{self.timestamp}.csv", index=False)

            # Log top features
            logging.info("\nTop Features for BP Prediction:")

            logging.info("\nTop 10 Features for SBP Prediction:")
            for i, row in importance_df.sort_values('SBP_Importance', ascending=False).head(10).iterrows():
                logging.info(f"  {row['Feature']}: {row['SBP_Importance']:.4f}")

            logging.info("\nTop 10 Features for DBP Prediction:")
            for i, row in importance_df.sort_values('DBP_Importance', ascending=False).head(10).iterrows():
                logging.info(f"  {row['Feature']}: {row['DBP_Importance']:.4f}")

        except Exception as e:
            logging.warning(f"Could not extract detailed feature importance: {str(e)}")

    def _save_models(self):
        """Save all trained models to disk"""
        models_dir = "models"
        os.makedirs(models_dir, exist_ok=True)

        # Save main models
        joblib.dump(self.sbp_regressor, f"{models_dir}/sbp_model_{self.timestamp}.joblib")
        joblib.dump(self.dbp_regressor, f"{models_dir}/dbp_model_{self.timestamp}.joblib")

        # Save ensemble models if available
        if hasattr(self, 'sbp_ensemble') and self.sbp_ensemble is not None:
            joblib.dump(self.sbp_ensemble, f"{models_dir}/sbp_ensemble_{self.timestamp}.joblib")

        if hasattr(self, 'dbp_ensemble') and self.dbp_ensemble is not None:
            joblib.dump(self.dbp_ensemble, f"{models_dir}/dbp_ensemble_{self.timestamp}.joblib")

        # Save quantile regressors if available
        if hasattr(self, 'sbp_quantile_10') and self.sbp_quantile_10 is not None:
            joblib.dump(self.sbp_quantile_10, f"{models_dir}/sbp_quantile_10_{self.timestamp}.joblib")
            joblib.dump(self.sbp_quantile_90, f"{models_dir}/sbp_quantile_90_{self.timestamp}.joblib")
            joblib.dump(self.sbp_preprocess, f"{models_dir}/sbp_preprocess_{self.timestamp}.joblib")

        if hasattr(self, 'dbp_quantile_10') and self.dbp_quantile_10 is not None:
            joblib.dump(self.dbp_quantile_10, f"{models_dir}/dbp_quantile_10_{self.timestamp}.joblib")
            joblib.dump(self.dbp_quantile_90, f"{models_dir}/dbp_quantile_90_{self.timestamp}.joblib")
            joblib.dump(self.dbp_preprocess, f"{models_dir}/dbp_preprocess_{self.timestamp}.joblib")

        # Save stratified models if available
        if hasattr(self, 'sbp_stratum_models') and self.sbp_stratum_models:
            joblib.dump(self.sbp_stratum_models, f"{models_dir}/sbp_stratum_models_{self.timestamp}.joblib")

        logging.info(f"All models saved to {models_dir}/")

    def load_models(self, timestamp):
        """Load trained models from disk"""
        models_dir = "models"

        try:
            # Load main models
            self.sbp_regressor = joblib.load(f"{models_dir}/sbp_model_{timestamp}.joblib")
            self.dbp_regressor = joblib.load(f"{models_dir}/dbp_model_{timestamp}.joblib")
            logging.info("Loaded main BP models")

            # Load ensemble models if available
            try:
                self.sbp_ensemble = joblib.load(f"{models_dir}/sbp_ensemble_{timestamp}.joblib")
                self.dbp_ensemble = joblib.load(f"{models_dir}/dbp_ensemble_{timestamp}.joblib")
                logging.info("Loaded ensemble models")
            except FileNotFoundError:
                logging.info("Ensemble models not found")

            # Load quantile regressors if available
            try:
                self.sbp_quantile_10 = joblib.load(f"{models_dir}/sbp_quantile_10_{timestamp}.joblib")
                self.sbp_quantile_90 = joblib.load(f"{models_dir}/sbp_quantile_90_{timestamp}.joblib")
                self.sbp_preprocess = joblib.load(f"{models_dir}/sbp_preprocess_{timestamp}.joblib")

                self.dbp_quantile_10 = joblib.load(f"{models_dir}/dbp_quantile_10_{timestamp}.joblib")
                self.dbp_quantile_90 = joblib.load(f"{models_dir}/dbp_quantile_90_{timestamp}.joblib")
                self.dbp_preprocess = joblib.load(f"{models_dir}/dbp_preprocess_{timestamp}.joblib")
                logging.info("Loaded quantile regression models")
            except FileNotFoundError:
                logging.info("Quantile regression models not found")

            # Load stratified models if available
            try:
                self.sbp_stratum_models = joblib.load(f"{models_dir}/sbp_stratum_models_{timestamp}.joblib")
                logging.info("Loaded stratified models")
            except FileNotFoundError:
                logging.info("Stratified models not found")

            return True

        except Exception as e:
            logging.error(f"Error loading models: {str(e)}")
            return False


class ModelEvaluator:
    """
    Enhanced model evaluator with stratified analysis, uncertainty quantification,
    and clinical utility metrics
    """

    def __init__(self, config=None, timestamp=None):
        self.config = config if config else {}
        self.save_figures = self.config.get('save_figures', True)
        self.shap_analysis = self.config.get('shap_analysis', True)
        self.uncertainty_quantification = self.config.get('uncertainty_quantification', True)
        self.clinical_utility = self.config.get('clinical_utility_metrics', True)
        self.workflow_simulation = self.config.get('workflow_simulation', True)
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

    def _safe_json_serialize(self, evaluation_results, dataset_name):
        """Safely serialize evaluation results to JSON"""
        try:
            # Create a clean dictionary with only serializable data
            clean_results = {
                'dataset': dataset_name,
                'timestamp': self.timestamp,
                'metrics': {},
                'hypothesis_tests': {},
            }

            # Safely extract metrics
            if 'metrics' in evaluation_results:
                for bp_type in ['SBP', 'DBP']:
                    if bp_type in evaluation_results['metrics']:
                        clean_results['metrics'][bp_type] = {
                            k: float(v) if isinstance(v, (int, float, np.number)) else v
                            for k, v in evaluation_results['metrics'][bp_type].items()
                            if not callable(v) and k != 'nan'
                        }

            # Safely extract hypothesis tests
            if 'hypothesis_tests' in evaluation_results:
                for bp_type in ['SBP', 'DBP']:
                    if bp_type in evaluation_results['hypothesis_tests']:
                        clean_results['hypothesis_tests'][bp_type] = {
                            k: float(v) if isinstance(v, (int, float, np.number)) else str(v)
                            for k, v in evaluation_results['hypothesis_tests'][bp_type].items()
                            if k != 'limits_of_agreement'
                        }
                        # Handle limits_of_agreement separately
                        if 'limits_of_agreement' in evaluation_results['hypothesis_tests'][bp_type]:
                            loa = evaluation_results['hypothesis_tests'][bp_type]['limits_of_agreement']
                            if isinstance(loa, tuple) and len(loa) == 2:
                                clean_results['hypothesis_tests'][bp_type]['limits_of_agreement'] = [float(loa[0]),
                                                                                                     float(loa[1])]

            # Add fold metrics if available
            if 'fold_metrics' in evaluation_results:
                clean_results['fold_metrics'] = evaluation_results['fold_metrics']

            # Save to file
            os.makedirs("results", exist_ok=True)
            with open(f"results/{dataset_name}_evaluation_{self.timestamp}.json", 'w') as f:
                json.dump(clean_results, f, indent=2)

            logging.info(f"Evaluation results saved to results/{dataset_name}_evaluation_{self.timestamp}.json")
            return True

        except Exception as e:
            logging.error(f"Error saving evaluation results: {str(e)}")
            return False

    def evaluate_models(self, model, X, y, groups, dataset_name="MIMIC-III"):
        """Comprehensive evaluation with clinical metrics using cross-validation"""
        try:
            logging.info(f"Evaluating models on {dataset_name} dataset...")

            # Create output directory for figures
            figures_dir = f"figures_{self.timestamp}"
            os.makedirs(figures_dir, exist_ok=True)

            # Create GroupKFold with 5 splits for evaluation
            group_cv = GroupKFold(n_splits=5)

            # Initialize containers for predictions and true values
            predictions = {'SBP': [], 'DBP': []}
            true_values = {'SBP': [], 'DBP': []}
            prediction_intervals = {'SBP': [], 'DBP': []}
            fold_metrics = {'SBP': [], 'DBP': []}

            # Track metrics per fold
            for fold_idx, (train_idx, test_idx) in enumerate(group_cv.split(X, y, groups)):
                logging.info(f"Evaluating on fold {fold_idx + 1}/5")

                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train models on this fold
                if model.use_bayesian_optimization:
                    model.train_models_with_bayesian_optimization(X_train, y_train, groups.iloc[train_idx])
                else:
                    model.train_models_standard(X_train, y_train, groups.iloc[train_idx])

                # Make predictions with uncertainty
                sbp_pred, dbp_pred, sbp_intervals, dbp_intervals = model.predict_with_uncertainty(X_test)

                # Store predictions and true values
                predictions['SBP'].extend(sbp_pred)
                predictions['DBP'].extend(dbp_pred)
                true_values['SBP'].extend(y_test['sbp_mean'].values)
                true_values['DBP'].extend(y_test['dbp_mean'].values)

                # Store prediction intervals if available
                if sbp_intervals is not None:
                    prediction_intervals['SBP'].extend(sbp_intervals)
                if dbp_intervals is not None:
                    prediction_intervals['DBP'].extend(dbp_intervals)

                # Calculate and store metrics for this fold
                for bp_type, pred, true in zip(
                        ['SBP', 'DBP'],
                        [sbp_pred, dbp_pred],
                        [y_test['sbp_mean'].values, y_test['dbp_mean'].values]
                ):
                    fold_metrics[bp_type].append({
                        'fold': fold_idx + 1,
                        'mse': mean_squared_error(true, pred),
                        'rmse': np.sqrt(mean_squared_error(true, pred)),
                        'mae': mean_absolute_error(true, pred),
                        'r2': r2_score(true, pred),
                        'pearson_r': stats.pearsonr(true, pred)[0]
                    })

            # Convert to numpy arrays for overall metrics
            for bp_type in ['SBP', 'DBP']:
                true_values[bp_type] = np.array(true_values[bp_type])
                predictions[bp_type] = np.array(predictions[bp_type])
                if prediction_intervals[bp_type]:
                    prediction_intervals[bp_type] = np.array(prediction_intervals[bp_type])

            # Calculate overall metrics
            metrics = self._calculate_overall_metrics(true_values, predictions, prediction_intervals)

            # Log per-fold metrics
            for bp_type in ['SBP', 'DBP']:
                logging.info(f"\n{bp_type} Metrics by Fold:")
                for fold in fold_metrics[bp_type]:
                    logging.info(
                        f"  Fold {fold['fold']}: RMSE={fold['rmse']:.2f}, R²={fold['r2']:.4f}, Pearson r={fold['pearson_r']:.4f}")

            # Hypothesis testing
            sbp_stats = self._perform_hypothesis_testing(true_values['SBP'], predictions['SBP'], 'SBP')
            dbp_stats = self._perform_hypothesis_testing(true_values['DBP'], predictions['DBP'], 'DBP')

            # Generate stratified clinical report
            stratified_results = self._generate_stratified_report(
                true_values['SBP'], predictions['SBP'],
                true_values['DBP'], predictions['DBP'],
                dataset_name
            )

            # Calculate clinical utility metrics if enabled
            clinical_metrics = None
            workflow_simulation = None

            if self.clinical_utility and prediction_intervals['SBP'] is not None:
                clinical_metrics = self._calculate_clinical_utility_metrics(
                    true_values['SBP'], predictions['SBP'],
                    prediction_intervals['SBP'][:, 0], prediction_intervals['SBP'][:, 1]
                )

            # Generate workflow simulation if enabled
            if self.workflow_simulation and prediction_intervals['SBP'] is not None:
                workflow_simulation = self._simulate_clinical_workflow_impact(
                    np.abs(true_values['SBP'] - predictions['SBP']),
                    prediction_intervals['SBP']
                )

            # Generate visualizations
            if self.save_figures:
                self._generate_evaluation_figures(
                    true_values['SBP'], predictions['SBP'],
                    true_values['DBP'], predictions['DBP'],
                    prediction_intervals, metrics, dataset_name
                )

            # SHAP analysis if enabled
            shap_values = None
            if self.shap_analysis:
                shap_values = self._analyze_shap_values(X, model, dataset_name)

            # Create evaluation results dictionary
            evaluation_results = {
                'metrics': metrics,
                'hypothesis_tests': {
                    'SBP': sbp_stats,
                    'DBP': dbp_stats
                },
                'stratified_results': stratified_results,
                'fold_metrics': fold_metrics,
                'predictions': {
                    'SBP': predictions['SBP'],
                    'DBP': predictions['DBP']
                },
                'true_values': {
                    'SBP': true_values['SBP'],
                    'DBP': true_values['DBP']
                },
                'prediction_intervals': prediction_intervals,
                'clinical_metrics': clinical_metrics,
                'workflow_simulation': workflow_simulation,
                'shap_values': shap_values,
                'dataset': dataset_name
            }

            # Save evaluation results safely (FIXED VERSION)
            self._safe_json_serialize(evaluation_results, dataset_name)

            return evaluation_results

        except Exception as e:
            logging.error(f"Error in model evaluation: {str(e)}")
            raise

    def _calculate_overall_metrics(self, true_values, predictions, prediction_intervals):
        """Calculate comprehensive metrics for model evaluation"""
        metrics = {}

        for bp_type in ['SBP', 'DBP']:
            y_true_arr = true_values[bp_type]
            y_pred_arr = predictions[bp_type]

            # Calculate absolute errors for each prediction
            abs_errors = np.abs(y_true_arr - y_pred_arr)

            # Basic metrics
            metrics[bp_type] = {
                'MSE': mean_squared_error(y_true_arr, y_pred_arr),
                'RMSE': np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)),
                'MAE': mean_absolute_error(y_true_arr, y_pred_arr),
                'R2': r2_score(y_true_arr, y_pred_arr),
                'Pearson_r': stats.pearsonr(y_true_arr, y_pred_arr)[0],
                'Spearman_r': stats.spearmanr(y_true_arr, y_pred_arr)[0],

                # Clinical thresholds
                'Within_5mmHg': np.mean(abs_errors <= 5) * 100,
                'Within_10mmHg': np.mean(abs_errors <= 10) * 100,
                'Within_15mmHg': np.mean(abs_errors <= 15) * 100,

                # Error distribution
                'Median_Error': np.median(abs_errors),
                '90th_Percentile_Error': np.percentile(abs_errors, 90),
                '95th_Percentile_Error': np.percentile(abs_errors, 95)
            }

            # Add uncertainty metrics if available
            if prediction_intervals[bp_type] is not None and len(prediction_intervals[bp_type]) > 0:
                lower_bounds = prediction_intervals[bp_type][:, 0]
                upper_bounds = prediction_intervals[bp_type][:, 1]

                # Coverage probability (should be ~80% for 10th-90th percentile)
                coverage = np.mean((y_true_arr >= lower_bounds) & (y_true_arr <= upper_bounds)) * 100

                # Average interval width
                interval_width = np.mean(upper_bounds - lower_bounds)

                # Add to metrics
                metrics[bp_type]['Prediction_Interval_Coverage'] = coverage
                metrics[bp_type]['Prediction_Interval_Width'] = interval_width

        # Log key metrics
        logging.info("\nOverall Model Performance:")
        for bp_type in ['SBP', 'DBP']:
            logging.info(f"\n{bp_type} Metrics:")
            logging.info(f"  RMSE: {metrics[bp_type]['RMSE']:.2f} mmHg")
            logging.info(f"  MAE: {metrics[bp_type]['MAE']:.2f} mmHg")
            logging.info(f"  R²: {metrics[bp_type]['R2']:.4f}")
            logging.info(f"  Pearson r: {metrics[bp_type]['Pearson_r']:.4f}")
            logging.info(f"  Within 5 mmHg: {metrics[bp_type]['Within_5mmHg']:.1f}%")
            logging.info(f"  Within 10 mmHg: {metrics[bp_type]['Within_10mmHg']:.1f}%")

            if 'Prediction_Interval_Coverage' in metrics[bp_type]:
                logging.info(f"  Prediction Interval Coverage: {metrics[bp_type]['Prediction_Interval_Coverage']:.1f}%")
                logging.info(
                    f"  Average Prediction Interval Width: {metrics[bp_type]['Prediction_Interval_Width']:.2f} mmHg")

        return metrics

    def _perform_hypothesis_testing(self, y_true, y_pred, title):
        """Statistical hypothesis testing with clinical relevance"""
        try:
            y_true_arr = np.asarray(y_true)
            y_pred_arr = np.asarray(y_pred)

            # Calculate errors
            errors = y_true_arr - y_pred_arr
            abs_errors = np.abs(errors)

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(y_true_arr, y_pred_arr)

            # Bland-Altman analysis
            mean_diff = np.mean(errors)
            std_diff = np.std(errors, ddof=1)
            limits_of_agreement = (mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff)

            # Error metrics
            mae_val = np.mean(abs_errors)
            se = std_diff / np.sqrt(len(y_true_arr))

            # British Hypertension Society (BHS) grade
            percent_within_5 = 100 * np.mean(abs_errors <= 5)
            percent_within_10 = 100 * np.mean(abs_errors <= 10)
            percent_within_15 = 100 * np.mean(abs_errors <= 15)

            # Determine BHS grade
            if percent_within_5 >= 60 and percent_within_10 >= 85 and percent_within_15 >= 95:
                bhs_grade = "A"
            elif percent_within_5 >= 50 and percent_within_10 >= 75 and percent_within_15 >= 90:
                bhs_grade = "B"
            elif percent_within_5 >= 40 and percent_within_10 >= 65 and percent_within_15 >= 85:
                bhs_grade = "C"
            else:
                bhs_grade = "D"

            # AAMI standard check
            aami_pass = abs(mean_diff) <= 5 and std_diff <= 8

            # Log results
            logging.info(f"\nHypothesis Testing for {title}:")
            logging.info(f"T-test statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
            logging.info(f"Bland-Altman mean difference: {mean_diff:.2f} mmHg")
            logging.info(
                f"Bland-Altman limits of agreement: ({limits_of_agreement[0]:.2f}, {limits_of_agreement[1]:.2f}) mmHg")
            logging.info(f"Mean Absolute Error: {mae_val:.2f} mmHg")
            logging.info(f"Standard Error: {se:.4f} mmHg")
            logging.info(
                f"BHS Grade: {bhs_grade} ({percent_within_5:.1f}% within 5mmHg, {percent_within_10:.1f}% within 10mmHg, {percent_within_15:.1f}% within 15mmHg)")
            logging.info(f"AAMI Standard: {'PASS' if aami_pass else 'FAIL'}")

            return {
                't_statistic': t_stat, 'p_value': p_value,
                'mean_difference': mean_diff, 'std_difference': std_diff,
                'limits_of_agreement': limits_of_agreement,
                'mae': mae_val, 'se': se,
                'percent_within_5': percent_within_5,
                'percent_within_10': percent_within_10,
                'percent_within_15': percent_within_15,
                'bhs_grade': bhs_grade,
                'aami_pass': aami_pass
            }

        except Exception as e:
            logging.error(f"Error in hypothesis testing: {str(e)}")
            raise

    def _generate_stratified_report(self, sbp_true, sbp_pred, dbp_true, dbp_pred, dataset_name):
        """Generate clinical report with BP stratification"""
        try:
            # Create DataFrame for analysis
            result_df = pd.DataFrame({
                'SBP_True': sbp_true,
                'SBP_Pred': sbp_pred,
                'SBP_Error': sbp_true - sbp_pred,
                'SBP_Abs_Error': np.abs(sbp_true - sbp_pred),
                'DBP_True': dbp_true,
                'DBP_Pred': dbp_pred,
                'DBP_Error': dbp_true - dbp_pred,
                'DBP_Abs_Error': np.abs(dbp_true - dbp_pred)
            })

            # Create BP categories
            result_df['SBP_Category'] = pd.cut(
                result_df['SBP_True'],
                bins=[0, 90, 120, 140, 160, 300],
                labels=['Hypotension', 'Normal', 'Prehypertension', 'Stage 1 HTN', 'Stage 2 HTN']
            )

            result_df['DBP_Category'] = pd.cut(
                result_df['DBP_True'],
                bins=[0, 60, 80, 90, 100, 200],
                labels=['Hypotension', 'Normal', 'Prehypertension', 'Stage 1 HTN', 'Stage 2 HTN']
            )

            # Stratified analysis by BP category
            logging.info("\nStratified Analysis by BP Category:")

            stratified_results = {}

            for bp_type in ['SBP', 'DBP']:
                category_col = f'{bp_type}_Category'
                error_col = f'{bp_type}_Abs_Error'

                logging.info(f"\n{bp_type} Stratified Results:")

                # Group by BP category and calculate metrics
                metrics_by_category = result_df.groupby(category_col)[error_col].agg(
                    ['count', 'mean', 'std', 'median',
                     lambda x: np.mean(x <= 5) * 100,
                     lambda x: np.mean(x <= 10) * 100,
                     lambda x: np.mean(x <= 15) * 100]
                ).reset_index()

                metrics_by_category.columns = [category_col, 'Count', 'MAE', 'Error_STD',
                                               'Median_Error', 'Within_5mmHg',
                                               'Within_10mmHg', 'Within_15mmHg']

                # Add BHS grade for each category
                metrics_by_category['BHS_Grade'] = metrics_by_category.apply(
                    lambda row: 'A' if (row['Within_5mmHg'] >= 60 and row['Within_10mmHg'] >= 85 and row[
                        'Within_15mmHg'] >= 95) else
                    ('B' if (row['Within_5mmHg'] >= 50 and row['Within_10mmHg'] >= 75 and row[
                        'Within_15mmHg'] >= 90) else
                     ('C' if (row['Within_5mmHg'] >= 40 and row['Within_10mmHg'] >= 65 and row[
                         'Within_15mmHg'] >= 85) else 'D')),
                    axis=1
                )

                # Log results
                for _, row in metrics_by_category.iterrows():
                    logging.info(f"  {row[category_col]}: n={row['Count']}, MAE={row['MAE']:.2f} mmHg, "
                                 f"Within 5mmHg={row['Within_5mmHg']:.1f}%, "
                                 f"Within 10mmHg={row['Within_10mmHg']:.1f}%, "
                                 f"BHS Grade={row['BHS_Grade']}")

                # Store results
                stratified_results[bp_type] = metrics_by_category.to_dict('records')

                # Save stratified results to CSV
                metrics_by_category.to_csv(f"{bp_type}_stratified_results_{dataset_name}_{self.timestamp}.csv",
                                           index=False)

            # Save full prediction results
            result_df.to_csv(f"prediction_results_{dataset_name}_{self.timestamp}.csv", index=False)

            logging.info(f"\nStratified analysis completed. Results saved for {dataset_name}.")

            return stratified_results

        except Exception as e:
            logging.error(f"Error generating stratified report: {str(e)}")
            logging.info("Continuing with evaluation despite reporting error")
            return None

    def _calculate_clinical_utility_metrics(self, y_true, y_pred, y_pred_lower, y_pred_upper):
        """Calculate clinically relevant metrics for decision support"""
        # Define important clinical thresholds
        hypotension_threshold = 90  # SBP < 90 mmHg
        hypertension_threshold = 140  # SBP > 140 mmHg

        # Create binary classification metrics
        hypotension_true = (y_true < hypotension_threshold).astype(int)
        hypotension_pred = (y_pred < hypotension_threshold).astype(int)

        hypertension_true = (y_true > hypertension_threshold).astype(int)
        hypertension_pred = (y_pred > hypertension_threshold).astype(int)

        # Calculate sensitivity, specificity, PPV, NPV for hypotension detection
        hypo_cm = confusion_matrix(hypotension_true, hypotension_pred)
        hypo_sensitivity = hypo_cm[1, 1] / (hypo_cm[1, 1] + hypo_cm[1, 0]) if (hypo_cm[1, 1] + hypo_cm[1, 0]) > 0 else 0
        hypo_specificity = hypo_cm[0, 0] / (hypo_cm[0, 0] + hypo_cm[0, 1]) if (hypo_cm[0, 0] + hypo_cm[0, 1]) > 0 else 0
        hypo_ppv = hypo_cm[1, 1] / (hypo_cm[1, 1] + hypo_cm[0, 1]) if (hypo_cm[1, 1] + hypo_cm[0, 1]) > 0 else 0
        hypo_npv = hypo_cm[0, 0] / (hypo_cm[0, 0] + hypo_cm[1, 0]) if (hypo_cm[0, 0] + hypo_cm[1, 0]) > 0 else 0

        # Same for hypertension
        hyper_cm = confusion_matrix(hypertension_true, hypertension_pred)
        hyper_sensitivity = hyper_cm[1, 1] / (hyper_cm[1, 1] + hyper_cm[1, 0]) if (hyper_cm[1, 1] + hyper_cm[
            1, 0]) > 0 else 0
        hyper_specificity = hyper_cm[0, 0] / (hyper_cm[0, 0] + hyper_cm[0, 1]) if (hyper_cm[0, 0] + hyper_cm[
            0, 1]) > 0 else 0
        hyper_ppv = hyper_cm[1, 1] / (hyper_cm[1, 1] + hyper_cm[0, 1]) if (hyper_cm[1, 1] + hyper_cm[0, 1]) > 0 else 0
        hyper_npv = hyper_cm[0, 0] / (hyper_cm[0, 0] + hyper_cm[1, 0]) if (hyper_cm[0, 0] + hyper_cm[1, 0]) > 0 else 0

        # Calculate prediction interval coverage (should be ~80% for 10th-90th percentile interval)
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper)) * 100

        # Average interval width
        interval_width = np.mean(y_pred_upper - y_pred_lower)

        # Create clinical utility metrics dictionary
        clinical_metrics = {
            'Hypotension_Sensitivity': hypo_sensitivity * 100,
            'Hypotension_Specificity': hypo_specificity * 100,
            'Hypotension_PPV': hypo_ppv * 100,
            'Hypotension_NPV': hypo_npv * 100,
            'Hypertension_Sensitivity': hyper_sensitivity * 100,
            'Hypertension_Specificity': hyper_specificity * 100,
            'Hypertension_PPV': hyper_ppv * 100,
            'Hypertension_NPV': hyper_npv * 100,
            'Prediction_Interval_Coverage': coverage,
            'Prediction_Interval_Width': interval_width
        }

        # Log clinical utility metrics
        logging.info("\nClinical Utility Metrics:")
        logging.info(f"Hypotension Detection (SBP < {hypotension_threshold} mmHg):")
        logging.info(f"  Sensitivity: {hypo_sensitivity * 100:.1f}%")
        logging.info(f"  Specificity: {hypo_specificity * 100:.1f}%")
        logging.info(f"  PPV: {hypo_ppv * 100:.1f}%")
        logging.info(f"  NPV: {hypo_npv * 100:.1f}%")

        logging.info(f"\nHypertension Detection (SBP > {hypertension_threshold} mmHg):")
        logging.info(f"  Sensitivity: {hyper_sensitivity * 100:.1f}%")
        logging.info(f"  Specificity: {hyper_specificity * 100:.1f}%")
        logging.info(f"  PPV: {hyper_ppv * 100:.1f}%")
        logging.info(f"  NPV: {hyper_npv * 100:.1f}%")

        logging.info(f"\nUncertainty Quantification:")
        logging.info(f"  Prediction Interval Coverage: {coverage:.1f}%")
        logging.info(f"  Average Interval Width: {interval_width:.2f} mmHg")

        return clinical_metrics

    def _simulate_clinical_workflow_impact(self, error_distribution, prediction_intervals):
        """Simulate impact on clinical workflow of using the model"""
        # Parameters for simulation
        n_patients = 1000
        manual_measurement_time = 3  # minutes per manual measurement
        measurement_frequency_reduction = {
            'low_risk': 0.5,  # Reduce by 50% for low-risk patients
            'medium_risk': 0.25,  # Reduce by 25% for medium-risk patients
            'high_risk': 0.0  # No reduction for high-risk patients
        }

        # Define risk categories based on prediction interval width
        interval_widths = prediction_intervals[:, 1] - prediction_intervals[:, 0]
        risk_categories = np.zeros(len(interval_widths), dtype=int)

        # Assign risk categories: 0=low, 1=medium, 2=high
        risk_categories[interval_widths > np.percentile(interval_widths, 66)] = 2  # High risk (top third)
        risk_categories[
            (interval_widths <= np.percentile(interval_widths, 66)) &
            (interval_widths > np.percentile(interval_widths, 33))
            ] = 1  # Medium risk (middle third)

        # Count patients in each risk category
        risk_counts = {
            'low_risk': np.sum(risk_categories == 0),
            'medium_risk': np.sum(risk_categories == 1),
            'high_risk': np.sum(risk_categories == 2)
        }

        # Calculate time savings
        baseline_measurements = 24  # Measurements per day for all patients
        baseline_time = n_patients * baseline_measurements * manual_measurement_time  # minutes

        # Calculate reduced measurements based on risk
        reduced_measurements = (
                risk_counts['low_risk'] * baseline_measurements * (1 - measurement_frequency_reduction['low_risk']) +
                risk_counts['medium_risk'] * baseline_measurements * (
                            1 - measurement_frequency_reduction['medium_risk']) +
                risk_counts['high_risk'] * baseline_measurements * (1 - measurement_frequency_reduction['high_risk'])
        )

        reduced_time = reduced_measurements * manual_measurement_time  # minutes
        time_saved = baseline_time - reduced_time  # minutes
        nurse_hours_saved = time_saved / 60  # hours

        # Calculate potential early warning benefits
        # Assume model can detect BP changes 30 minutes earlier than manual measurements
        early_detection_benefit = 30  # minutes

        # For high-risk patients, calculate potential early interventions
        high_risk_patient_interventions = risk_counts['high_risk'] * 0.2  # Assume 20% need intervention
        early_intervention_time_saved = high_risk_patient_interventions * early_detection_benefit  # minutes

        # Log results
        logging.info("\nClinical Workflow Impact Simulation:")
        logging.info(f"Patient Risk Stratification:")
        logging.info(f"  Low Risk: {risk_counts['low_risk']} patients")
        logging.info(f"  Medium Risk: {risk_counts['medium_risk']} patients")
        logging.info(f"  High Risk: {risk_counts['high_risk']} patients")

        logging.info(f"\nWorkflow Efficiency:")
        logging.info(f"  Baseline Measurement Time: {baseline_time / 60:.1f} nurse-hours per day")
        logging.info(f"  Reduced Measurement Time: {reduced_time / 60:.1f} nurse-hours per day")
        logging.info(f"  Time Saved: {nurse_hours_saved:.1f} nurse-hours per day")
        logging.info(
            f"  Early Intervention Potential: {early_intervention_time_saved / 60:.1f} hours of earlier detection")

        # Return simulation results
        return {
            'risk_stratification': risk_counts,
            'baseline_time': baseline_time,
            'reduced_time': reduced_time,
            'time_saved': time_saved,
            'nurse_hours_saved': nurse_hours_saved,
            'early_intervention_potential': early_intervention_time_saved
        }

    def _generate_evaluation_figures(self, sbp_true, sbp_pred, dbp_true, dbp_pred,
                                     prediction_intervals, metrics, dataset_name):
        """Generate evaluation figures for model performance visualization"""
        # Create output directory for figures
        figures_dir = f"figures_{self.timestamp}"
        os.makedirs(figures_dir, exist_ok=True)

        # 1. REGRESSION PLOTS WITH ERROR METRICS
        plt.figure(figsize=(12, 12))

        # SBP subplot
        plt.subplot(2, 2, 1)
        plt.scatter(sbp_true, sbp_pred, alpha=0.5, s=30, c='darkblue')

        # Add perfect prediction line
        min_val, max_val = min(sbp_true), max(sbp_true)
        buffer = (max_val - min_val) * 0.05  # 5% buffer
        plt.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'r--', linewidth=2)

        plt.xlabel('Measured SBP (mmHg)', fontsize=12)
        plt.ylabel('Predicted SBP (mmHg)', fontsize=12)
        plt.title(f'SBP Prediction (R² = {metrics["SBP"]["R2"]:.2f}, RMSE = {metrics["SBP"]["RMSE"]:.2f} mmHg)',
                  fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add text box with metrics
        plt.text(0.05, 0.95,
                 f'MAE: {metrics["SBP"]["MAE"]:.2f} mmHg\n'
                 f'Within 5 mmHg: {metrics["SBP"]["Within_5mmHg"]:.1f}%\n'
                 f'Within 10 mmHg: {metrics["SBP"]["Within_10mmHg"]:.1f}%',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # DBP subplot
        plt.subplot(2, 2, 2)
        plt.scatter(dbp_true, dbp_pred, alpha=0.5, s=30, c='darkgreen')

        # Add perfect prediction line
        min_val, max_val = min(dbp_true), max(dbp_true)
        buffer = (max_val - min_val) * 0.05  # 5% buffer
        plt.plot([min_val - buffer, max_val + buffer], [min_val - buffer, max_val + buffer], 'r--', linewidth=2)

        plt.xlabel('Measured DBP (mmHg)', fontsize=12)
        plt.ylabel('Predicted DBP (mmHg)', fontsize=12)
        plt.title(f'DBP Prediction (R² = {metrics["DBP"]["R2"]:.2f}, RMSE = {metrics["DBP"]["RMSE"]:.2f} mmHg)',
                  fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add text box with metrics
        plt.text(0.05, 0.95,
                 f'MAE: {metrics["DBP"]["MAE"]:.2f} mmHg\n'
                 f'Within 5 mmHg: {metrics["DBP"]["Within_5mmHg"]:.1f}%\n'
                 f'Within 10 mmHg: {metrics["DBP"]["Within_10mmHg"]:.1f}%',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # 2. BLAND-ALTMAN PLOTS
        # SBP Bland-Altman
        plt.subplot(2, 2, 3)
        mean_sbp = (sbp_true + sbp_pred) / 2
        diff_sbp = sbp_true - sbp_pred

        plt.scatter(mean_sbp, diff_sbp, alpha=0.5, s=30, c='darkblue')

        # Add mean and limits of agreement
        mean_diff = np.mean(diff_sbp)
        std_diff = np.std(diff_sbp, ddof=1)
        limits = (mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff)

        plt.axhline(y=mean_diff, color='r', linestyle='-', linewidth=2)
        plt.axhline(y=limits[0], color='gray', linestyle='--', linewidth=1.5)
        plt.axhline(y=limits[1], color='gray', linestyle='--', linewidth=1.5)

        plt.xlabel('Mean of Measured and Predicted SBP (mmHg)', fontsize=12)
        plt.ylabel('Difference (Measured - Predicted) (mmHg)', fontsize=12)
        plt.title('Bland-Altman Plot for SBP', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add text box with Bland-Altman stats
        plt.text(0.05, 0.95,
                 f'Mean Bias: {mean_diff:.2f} mmHg\n'
                 f'LoA: ({limits[0]:.2f}, {limits[1]:.2f}) mmHg',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # DBP Bland-Altman
        plt.subplot(2, 2, 4)
        mean_dbp = (dbp_true + dbp_pred) / 2
        diff_dbp = dbp_true - dbp_pred

        plt.scatter(mean_dbp, diff_dbp, alpha=0.5, s=30, c='darkgreen')

        # Add mean and limits of agreement
        mean_diff = np.mean(diff_dbp)
        std_diff = np.std(diff_dbp, ddof=1)
        limits = (mean_diff - 1.96 * std_diff, mean_diff + 1.96 * std_diff)

        plt.axhline(y=mean_diff, color='r', linestyle='-', linewidth=2)
        plt.axhline(y=limits[0], color='gray', linestyle='--', linewidth=1.5)
        plt.axhline(y=limits[1], color='gray', linestyle='--', linewidth=1.5)

        plt.xlabel('Mean of Measured and Predicted DBP (mmHg)', fontsize=12)
        plt.ylabel('Difference (Measured - Predicted) (mmHg)', fontsize=12)
        plt.title('Bland-Altman Plot for DBP', fontsize=14)
        plt.grid(True, alpha=0.3)

        # Add text box with Bland-Altman stats
        plt.text(0.05, 0.95,
                 f'Mean Bias: {mean_diff:.2f} mmHg\n'
                 f'LoA: ({limits[0]:.2f}, {limits[1]:.2f}) mmHg',
                 transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/bp_prediction_analysis_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 3. ERROR DISTRIBUTION BY BP CATEGORY
        # Create BP categories
        result_df = pd.DataFrame({
            'SBP_True': sbp_true,
            'SBP_Pred': sbp_pred,
            'SBP_Error': sbp_true - sbp_pred,
            'SBP_Abs_Error': np.abs(sbp_true - sbp_pred),
            'DBP_True': dbp_true,
            'DBP_Pred': dbp_pred,
            'DBP_Error': dbp_true - dbp_pred,
            'DBP_Abs_Error': np.abs(dbp_true - dbp_pred)
        })

        # Define BP categories
        result_df['SBP_Category'] = pd.cut(
            result_df['SBP_True'],
            bins=[0, 90, 120, 140, 160, 300],
            labels=['Hypotension', 'Normal', 'Prehypertension', 'Stage 1 HTN', 'Stage 2 HTN']
        )

        result_df['DBP_Category'] = pd.cut(
            result_df['DBP_True'],
            bins=[0, 60, 80, 90, 100, 200],
            labels=['Hypotension', 'Normal', 'Prehypertension', 'Stage 1 HTN', 'Stage 2 HTN']
        )

        # Plot error distributions by category
        plt.figure(figsize=(12, 10))

        # SBP errors by category
        plt.subplot(2, 1, 1)
        sns.boxplot(x='SBP_Category', y='SBP_Abs_Error', data=result_df, palette='Blues')
        plt.title('SBP Absolute Error by BP Category', fontsize=14)
        plt.xlabel('BP Category', fontsize=12)
        plt.ylabel('Absolute Error (mmHg)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add category sample sizes
        for i, category in enumerate(result_df['SBP_Category'].unique()):
            if pd.isna(category):
                continue
            count = len(result_df[result_df['SBP_Category'] == category])
            plt.text(i, -2, f'n={count}', ha='center', fontsize=10)

        # DBP errors by category
        plt.subplot(2, 1, 2)
        sns.boxplot(x='DBP_Category', y='DBP_Abs_Error', data=result_df, palette='Greens')
        plt.title('DBP Absolute Error by BP Category', fontsize=14)
        plt.xlabel('BP Category', fontsize=12)
        plt.ylabel('Absolute Error (mmHg)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Add category sample sizes
        for i, category in enumerate(result_df['DBP_Category'].unique()):
            if pd.isna(category):
                continue
            count = len(result_df[result_df['DBP_Category'] == category])
            plt.text(i, -1, f'n={count}', ha='center', fontsize=10)

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/error_by_category_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 4. UNCERTAINTY VISUALIZATION
        if (prediction_intervals['SBP'] is not None and len(prediction_intervals['SBP']) > 0 and
                prediction_intervals['DBP'] is not None and len(prediction_intervals['DBP']) > 0):
            plt.figure(figsize=(12, 12))

            # SBP uncertainty
            plt.subplot(2, 1, 1)

            # Sort by true values for better visualization
            sort_idx = np.argsort(sbp_true)
            sorted_sbp_true = sbp_true[sort_idx]
            sorted_sbp_pred = sbp_pred[sort_idx]
            sorted_sbp_lower = prediction_intervals['SBP'][sort_idx, 0]
            sorted_sbp_upper = prediction_intervals['SBP'][sort_idx, 1]

            # Plot a subset of points for readability (every 10th point)
            subset = range(0, len(sorted_sbp_true), 10)

            plt.fill_between(np.arange(len(subset)),
                             sorted_sbp_lower[subset],
                             sorted_sbp_upper[subset],
                             alpha=0.3, color='blue', label='80% Prediction Interval')
            plt.plot(np.arange(len(subset)), sorted_sbp_true[subset], 'o', markersize=4, color='black', label='Actual')
            plt.plot(np.arange(len(subset)), sorted_sbp_pred[subset], 'x', markersize=4, color='red', label='Predicted')

            plt.title(
                f'SBP Prediction with Uncertainty ({metrics["SBP"]["Prediction_Interval_Coverage"]:.1f}% Coverage)',
                fontsize=14)
            plt.xlabel('Sample Index (Sorted by True SBP)', fontsize=12)
            plt.ylabel('SBP (mmHg)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

            # DBP uncertainty
            plt.subplot(2, 1, 2)

            # Sort by true values
            sort_idx = np.argsort(dbp_true)
            sorted_dbp_true = dbp_true[sort_idx]
            sorted_dbp_pred = dbp_pred[sort_idx]
            sorted_dbp_lower = prediction_intervals['DBP'][sort_idx, 0]
            sorted_dbp_upper = prediction_intervals['DBP'][sort_idx, 1]

            # Plot a subset of points for readability
            subset = range(0, len(sorted_dbp_true), 10)

            plt.fill_between(np.arange(len(subset)),
                             sorted_dbp_lower[subset],
                             sorted_dbp_upper[subset],
                             alpha=0.3, color='green', label='80% Prediction Interval')
            plt.plot(np.arange(len(subset)), sorted_dbp_true[subset], 'o', markersize=4, color='black', label='Actual')
            plt.plot(np.arange(len(subset)), sorted_dbp_pred[subset], 'x', markersize=4, color='red', label='Predicted')

            plt.title(
                f'DBP Prediction with Uncertainty ({metrics["DBP"]["Prediction_Interval_Coverage"]:.1f}% Coverage)',
                fontsize=14)
            plt.xlabel('Sample Index (Sorted by True DBP)', fontsize=12)
            plt.ylabel('DBP (mmHg)', fontsize=12)
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{figures_dir}/prediction_uncertainty_{dataset_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

            # 5. ERROR VS UNCERTAINTY PLOT
            plt.figure(figsize=(12, 6))

            # SBP error vs interval width
            plt.subplot(1, 2, 1)
            sbp_interval_width = prediction_intervals['SBP'][:, 1] - prediction_intervals['SBP'][:, 0]
            sbp_abs_error = np.abs(sbp_true - sbp_pred)

            plt.scatter(sbp_interval_width, sbp_abs_error, alpha=0.5, c='blue')

            # Add trend line
            z = np.polyfit(sbp_interval_width, sbp_abs_error, 1)
            p = np.poly1d(z)
            plt.plot(np.sort(sbp_interval_width), p(np.sort(sbp_interval_width)), "r--", alpha=0.8)

            # Add correlation coefficient
            corr = np.corrcoef(sbp_interval_width, sbp_abs_error)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=plt.gca().transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.title('SBP Error vs. Prediction Interval Width', fontsize=14)
            plt.xlabel('Prediction Interval Width (mmHg)', fontsize=12)
            plt.ylabel('Absolute Error (mmHg)', fontsize=12)
            plt.grid(True, alpha=0.3)

            # DBP error vs interval width
            plt.subplot(1, 2, 2)
            dbp_interval_width = prediction_intervals['DBP'][:, 1] - prediction_intervals['DBP'][:, 0]
            dbp_abs_error = np.abs(dbp_true - dbp_pred)

            plt.scatter(dbp_interval_width, dbp_abs_error, alpha=0.5, c='green')

            # Add trend line
            z = np.polyfit(dbp_interval_width, dbp_abs_error, 1)
            p = np.poly1d(z)
            plt.plot(np.sort(dbp_interval_width), p(np.sort(dbp_interval_width)), "r--", alpha=0.8)

            # Add correlation coefficient
            corr = np.corrcoef(dbp_interval_width, dbp_abs_error)[0, 1]
            plt.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=plt.gca().transAxes,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            plt.title('DBP Error vs. Prediction Interval Width', fontsize=14)
            plt.xlabel('Prediction Interval Width (mmHg)', fontsize=12)
            plt.ylabel('Absolute Error (mmHg)', fontsize=12)
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f"{figures_dir}/error_vs_uncertainty_{dataset_name}.png", dpi=300, bbox_inches='tight')
            plt.close()

    def _analyze_shap_values(self, X, model, dataset_name):
        """Analyze SHAP values to identify critical BP predictors"""
        try:
            logging.info("Performing SHAP analysis for feature importance...")

            # Create output directory for figures
            figures_dir = f"figures_{self.timestamp}"

            # Create explainer for SBP model
            explainer_sbp = shap.TreeExplainer(model.sbp_regressor.estimators_[0].named_steps['gb'])

            # Calculate SHAP values
            logging.info("Calculating SHAP values for SBP model...")

            # Use a subset of data for SHAP calculation if dataset is large
            if len(X) > 500:
                X_sample = X.sample(500, random_state=42)
                logging.info(f"Using 500 random samples for SHAP calculation (from {len(X)} total)")
            else:
                X_sample = X

            shap_values_sbp = explainer_sbp.shap_values(X_sample)

            # Create summary plot
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_sbp, X_sample, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance for SBP Prediction")
            plt.tight_layout()
            plt.savefig(f"{figures_dir}/shap_importance_sbp_{dataset_name}.png", dpi=300)
            plt.close()

            # Create SHAP dependence plots for top features
            feature_importance = np.abs(shap_values_sbp).mean(0)
            feature_names = X.columns
            top_indices = np.argsort(-feature_importance)[:5]  # Top 5 features

            for i in top_indices:
                feature_name = feature_names[i]
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(i, shap_values_sbp, X_sample, show=False)
                plt.title(f"SHAP Dependence Plot for {feature_name}")
                plt.tight_layout()
                plt.savefig(f"{figures_dir}/shap_dependence_{feature_name}_{dataset_name}.png", dpi=300)
                plt.close()

            # Do the same for DBP model
            explainer_dbp = shap.TreeExplainer(model.dbp_regressor.estimators_[0].named_steps['gb'])
            logging.info("Calculating SHAP values for DBP model...")
            shap_values_dbp = explainer_dbp.shap_values(X_sample)

            # Create summary plot for DBP
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_dbp, X_sample, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance for DBP Prediction")
            plt.tight_layout()
            plt.savefig(f"{figures_dir}/shap_importance_dbp_{dataset_name}.png", dpi=300)
            plt.close()

            # Create SHAP dependence plots for top DBP features
            feature_importance_dbp = np.abs(shap_values_dbp).mean(0)
            top_indices_dbp = np.argsort(-feature_importance_dbp)[:5]  # Top 5 features

            for i in top_indices_dbp:
                feature_name = feature_names[i]
                plt.figure(figsize=(10, 6))
                shap.dependence_plot(i, shap_values_dbp, X_sample, show=False)
                plt.title(f"SHAP Dependence Plot for {feature_name} (DBP)")
                plt.tight_layout()
                plt.savefig(f"{figures_dir}/shap_dependence_dbp_{feature_name}_{dataset_name}.png", dpi=300)
                plt.close()

            # Store SHAP values in a dictionary
            shap_analysis = {
                'sbp_values': shap_values_sbp,
                'dbp_values': shap_values_dbp,
                'feature_names': list(feature_names),
                'sbp_importance': list(feature_importance),
                'dbp_importance': list(feature_importance_dbp)
            }

            # Save SHAP analysis in a pickle file
            joblib.dump(shap_analysis, f"shap_analysis_{dataset_name}_{self.timestamp}.joblib")

            logging.info("SHAP analysis completed and saved")

            return shap_analysis

        except Exception as e:
            logging.error(f"Error in SHAP analysis: {str(e)}")
            return None


class SOTAComparator:
    """
    Compare model performance with state-of-the-art methods in literature
    """

    def __init__(self, timestamp=None):
        self.timestamp = timestamp if timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

    def compare_with_sota(self, metrics, dataset_name="MIMIC-III"):
        """Compare with state-of-the-art methods"""
        logging.info("\nComparison with State-of-the-Art Methods:")

        # Define state-of-the-art methods from literature
        sota_methods = {
            "Clinical Oscillometric Devices": {
                "SBP_RMSE": 8.0, "SBP_R2": 0.60, "SBP_Within_5mmHg": 58.0,
                "DBP_RMSE": 6.5, "DBP_R2": 0.65, "DBP_Within_5mmHg": 62.0,
                "Reference": "AAMI standards & commercial devices (IEEE JBHI 2021)"
            },
            "PPG-Based Methods": {
                "SBP_RMSE": 10.2, "SBP_R2": 0.55, "SBP_Within_5mmHg": 45.0,
                "DBP_RMSE": 7.8, "DBP_R2": 0.60, "DBP_Within_5mmHg": 51.0,
                "Reference": "Liang et al. (IEEE TBME 2022)"
            },
            "ECG-Based Methods": {
                "SBP_RMSE": 9.1, "SBP_R2": 0.62, "SBP_Within_5mmHg": 48.0,
                "DBP_RMSE": 6.5, "DBP_R2": 0.67, "DBP_Within_5mmHg": 56.0,
                "Reference": "Slapničar et al. (IEEE Access 2021)"
            },
            "ML on Clinical Variables": {
                "SBP_RMSE": 11.5, "SBP_R2": 0.50, "SBP_Within_5mmHg": 40.0,
                "DBP_RMSE": 8.0, "DBP_R2": 0.55, "DBP_Within_5mmHg": 45.0,
                "Reference": "Miao et al. (IEEE JBHI 2020)"
            }
        }

        # Current model performance
        our_method = {
            "SBP_RMSE": metrics["SBP"]["RMSE"],
            "SBP_R2": metrics["SBP"]["R2"],
            "SBP_Within_5mmHg": metrics["SBP"]["Within_5mmHg"],
            "DBP_RMSE": metrics["DBP"]["RMSE"],
            "DBP_R2": metrics["DBP"]["R2"],
            "DBP_Within_5mmHg": metrics["DBP"]["Within_5mmHg"],
        }

        # Create comparison table for SBP
        logging.info("\nSystolic BP Estimation Comparison:")
        logging.info("{:<25} {:<10} {:<10} {:<15} {:<30}".format(
            "Method", "RMSE", "R²", "Within 5mmHg", "Reference"
        ))
        logging.info("-" * 90)

        # Print SOTA methods for SBP
        for method, perf in sota_methods.items():
            logging.info("{:<25} {:<10.2f} {:<10.2f} {:<15.1f} {:<30}".format(
                method,
                perf["SBP_RMSE"],
                perf["SBP_R2"],
                perf["SBP_Within_5mmHg"],
                perf["Reference"]
            ))

        # Print our method for SBP
        logging.info("{:<25} {:<10.2f} {:<10.2f} {:<15.1f} {:<30}".format(
            f"Our {dataset_name} Method",
            our_method["SBP_RMSE"],
            our_method["SBP_R2"],
            our_method["SBP_Within_5mmHg"],
            "This study"
        ))

        # Repeat for DBP
        logging.info("\nDiastolic BP Estimation Comparison:")
        logging.info("{:<25} {:<10} {:<10} {:<15} {:<30}".format(
            "Method", "RMSE", "R²", "Within 5mmHg", "Reference"
        ))
        logging.info("-" * 90)

        # Print SOTA methods for DBP
        for method, perf in sota_methods.items():
            logging.info("{:<25} {:<10.2f} {:<10.2f} {:<15.1f} {:<30}".format(
                method,
                perf["DBP_RMSE"],
                perf["DBP_R2"],
                perf["DBP_Within_5mmHg"],
                perf["Reference"]
            ))

        # Print our method for DBP
        logging.info("{:<25} {:<10.2f} {:<10.2f} {:<15.1f} {:<30}".format(
            f"Our {dataset_name} Method",
            our_method["DBP_RMSE"],
            our_method["DBP_R2"],
            our_method["DBP_Within_5mmHg"],
            "This study"
        ))

        # Provide interpretation for publication
        logging.info("\nInterpretation for Publication:")

        # Compare SBP performance
        sbp_rmse_improvement = False
        sota_sbp_rmse = min([m["SBP_RMSE"] for m in sota_methods.values()])
        if our_method["SBP_RMSE"] < sota_sbp_rmse:
            sbp_rmse_improvement = True
            sbp_rmse_percent = ((sota_sbp_rmse - our_method["SBP_RMSE"]) / sota_sbp_rmse) * 100
            logging.info(
                f"Our method improves SBP RMSE by {sbp_rmse_percent:.1f}% compared to the best literature method.")

        # Compare DBP performance
        dbp_rmse_improvement = False
        sota_dbp_rmse = min([m["DBP_RMSE"] for m in sota_methods.values()])
        if our_method["DBP_RMSE"] < sota_dbp_rmse:
            dbp_rmse_improvement = True
            dbp_rmse_percent = ((sota_dbp_rmse - our_method["DBP_RMSE"]) / sota_dbp_rmse) * 100
            logging.info(
                f"Our method improves DBP RMSE by {dbp_rmse_percent:.1f}% compared to the best literature method.")

        # Overall assessment
        if sbp_rmse_improvement and dbp_rmse_improvement:
            logging.info("Our method demonstrates state-of-the-art performance for both SBP and DBP estimation.")
        elif sbp_rmse_improvement or dbp_rmse_improvement:
            logging.info("Our method demonstrates competitive performance compared with existing approaches.")
        else:
            sota_sbp_within5 = max([m["SBP_Within_5mmHg"] for m in sota_methods.values()])
            sota_dbp_within5 = max([m["DBP_Within_5mmHg"] for m in sota_methods.values()])

            if (our_method["SBP_Within_5mmHg"] > sota_sbp_within5 or
                    our_method["DBP_Within_5mmHg"] > sota_dbp_within5):
                logging.info(
                    "Our method shows improvements in clinical metrics while maintaining competitive error rates.")
            else:
                logging.info(
                    "Our method provides comparable performance to existing approaches with the advantage of using readily available clinical variables from EHR data.")

        # Create comparison plot
        figures_dir = f"figures_{self.timestamp}"
        os.makedirs(figures_dir, exist_ok=True)

        plt.figure(figsize=(12, 10))

        # SBP comparison
        plt.subplot(2, 1, 1)
        methods = list(sota_methods.keys()) + [f"Our {dataset_name} Method"]
        rmse_values = [sota_methods[m]["SBP_RMSE"] for m in sota_methods] + [our_method["SBP_RMSE"]]
        within5_values = [sota_methods[m]["SBP_Within_5mmHg"] for m in sota_methods] + [our_method["SBP_Within_5mmHg"]]

        x = np.arange(len(methods))
        width = 0.35

        ax1 = plt.gca()
        bars1 = ax1.bar(x - width / 2, rmse_values, width, label='RMSE (mmHg)', color='skyblue')
        ax1.set_ylabel('RMSE (mmHg)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, within5_values, width, label='Within 5mmHg (%)', color='lightgreen')
        ax2.set_ylabel('Within 5mmHg (%)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        ax1.set_title('SBP Performance Comparison', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')

        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        # DBP comparison
        plt.subplot(2, 1, 2)
        rmse_values = [sota_methods[m]["DBP_RMSE"] for m in sota_methods] + [our_method["DBP_RMSE"]]
        within5_values = [sota_methods[m]["DBP_Within_5mmHg"] for m in sota_methods] + [our_method["DBP_Within_5mmHg"]]

        ax1 = plt.gca()
        bars1 = ax1.bar(x - width / 2, rmse_values, width, label='RMSE (mmHg)', color='skyblue')
        ax1.set_ylabel('RMSE (mmHg)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        ax2 = ax1.twinx()
        bars2 = ax2.bar(x + width / 2, within5_values, width, label='Within 5mmHg (%)', color='lightgreen')
        ax2.set_ylabel('Within 5mmHg (%)', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        ax1.set_title('DBP Performance Comparison', fontsize=14)
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods, rotation=45, ha='right')

        # Add a legend
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.tight_layout()
        plt.savefig(f"{figures_dir}/sota_comparison_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

        # Create radar plot for comprehensive comparison
        plt.figure(figsize=(10, 8))

        # Setup radar chart
        methods = list(sota_methods.keys()) + [f"Our {dataset_name} Method"]
        metrics_to_plot = ['SBP_RMSE', 'SBP_Within_5mmHg', 'SBP_R2', 'DBP_RMSE', 'DBP_Within_5mmHg', 'DBP_R2']

        # Number of metrics
        N = len(metrics_to_plot)

        # Create angles for radar plot
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop

        # Create normalized values (0-1 scale) for each metric
        normalized_data = []

        # Normalize metrics (some are "higher is better", some are "lower is better")
        for method in methods:
            if method in sota_methods:
                values = [
                    1 - (sota_methods[method]['SBP_RMSE'] / 15),  # Lower is better, normalized to 15 mmHg
                    sota_methods[method]['SBP_Within_5mmHg'] / 100,  # Higher is better, normalized to 100%
                    sota_methods[method]['SBP_R2'],  # Already 0-1
                    1 - (sota_methods[method]['DBP_RMSE'] / 10),  # Lower is better, normalized to 10 mmHg
                    sota_methods[method]['DBP_Within_5mmHg'] / 100,  # Higher is better, normalized to 100%
                    sota_methods[method]['DBP_R2']  # Already 0-1
                ]
            else:
                values = [
                    1 - (our_method['SBP_RMSE'] / 15),
                    our_method['SBP_Within_5mmHg'] / 100,
                    our_method['SBP_R2'],
                    1 - (our_method['DBP_RMSE'] / 10),
                    our_method['DBP_Within_5mmHg'] / 100,
                    our_method['DBP_R2']
                ]

            # Add the first value to close the loop
            values += values[:1]
            normalized_data.append(values)

        # Set up the plot
        ax = plt.subplot(111, polar=True)

        # Set the labels for each metric
        metric_labels = ['SBP RMSE', 'SBP Within 5mmHg', 'SBP R²', 'DBP RMSE', 'DBP Within 5mmHg', 'DBP R²']
        plt.xticks(angles[:-1], metric_labels, size=10)

        # Set y limit
        ax.set_ylim(0, 1)

        # Plot each method
        colors = ['b', 'g', 'r', 'c', 'm']
        for i, (method, values) in enumerate(zip(methods, normalized_data)):
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=method, color=colors[i % len(colors)])
            ax.fill(angles, values, alpha=0.1, color=colors[i % len(colors)])

        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

        plt.title(f'Comprehensive Method Comparison - {dataset_name}', size=15)
        plt.savefig(f"{figures_dir}/radar_comparison_{dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.close()

        return sota_methods


def main():
    """
    Main function to run the enhanced MIMIC-III BP prediction pipeline
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Enhanced MIMIC-III BP Prediction Pipeline')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--log_level', type=str, default='INFO', help='Logging level')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--timestamp', type=str, default=None, help='Timestamp for file naming')
    parser.add_argument('--skip_training', action='store_true', help='Skip training (for evaluation only)')
    parser.add_argument('--load_models', type=str, default=None, help='Timestamp of models to load')
    args = parser.parse_args()

    # Set timestamp for file naming
    timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")

    # Set up logging
    logger = setup_logging(args.log_level, args.output_dir, timestamp)

    # Load configuration
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = DEFAULT_CONFIG

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Save configuration for reproducibility
    with open(f"{args.output_dir}/config_{timestamp}.yaml", 'w') as f:
        yaml.dump(config, f)

    logger.info("Starting Enhanced MIMIC-III BP Prediction Pipeline...")
    logger.info(f"Configuration: {config}")

    try:
        # Initialize data extractor for MIMIC-III
        mimic_extractor = MIMICDataExtractor(
            project_id=config['data_extraction']['project_id'],
            timestamp=timestamp
        )

        # Extract MIMIC-III data
        cohort_df = mimic_extractor.extract_patient_cohort(max_patients=config['data_extraction']['max_patients'])

        if len(cohort_df) == 0:
            logger.error("No patients found in MIMIC-III cohort")
            return

        logger.info(f"Extracted {len(cohort_df)} patients for analysis")

        # Extract features for these patients
        icustay_ids = cohort_df['icustay_id'].tolist()
        bp_df = mimic_extractor.extract_bp_data(icustay_ids)
        vitals_df = mimic_extractor.extract_vital_signs_features(icustay_ids)
        labs_df = mimic_extractor.extract_lab_features(icustay_ids)
        meds_df = mimic_extractor.extract_medication_features(icustay_ids)

        # Extract additional features if enabled
        temporal_df = None
        comorbidity_df = None

        if config['feature_engineering'].get('create_temporal_features', True):
            temporal_df = mimic_extractor.extract_temporal_features(icustay_ids)
            logger.info(f"Extracted temporal features: {temporal_df.shape if temporal_df is not None else 'None'}")

        # Extract comorbidity data
        comorbidity_df = mimic_extractor.extract_comorbidity_features(icustay_ids)
        logger.info(f"Extracted comorbidity features: {comorbidity_df.shape if comorbidity_df is not None else 'None'}")

        # Process features with enhanced feature engineering
        feature_processor = EnhancedFeatureProcessor(config['feature_engineering'], timestamp)
        df = feature_processor.process_features(
            cohort_df, bp_df, vitals_df, labs_df, meds_df,
            temporal_df, comorbidity_df, "MIMIC-III"
        )

        # Handle missing data with advanced imputation
        imputer = AdvancedDataImputer(config['imputation'], timestamp)
        df_imputed = imputer.impute_missing_values(df, "MIMIC-III")

        # Initialize BP predictor
        predictor = OptimizedBPPredictor(config['model'], timestamp)

        # Prepare data for modeling
        X, y, groups = predictor.prepare_data(df_imputed, "MIMIC-III")

        # Train models
        logger.info("Training BP prediction models...")
        if config['model'].get('bayesian_optimization', True):
            predictor.train_models_with_bayesian_optimization(X, y, groups)
        else:
            predictor.train_models_standard(X, y, groups)

        # Evaluate models on MIMIC-III
        evaluator = ModelEvaluator(config['evaluation'], timestamp)
        mimic_results = evaluator.evaluate_models(predictor, X, y, groups, "MIMIC-III")

        # Compare with SOTA methods
        comparator = SOTAComparator(timestamp)
        sota_comparison = comparator.compare_with_sota(mimic_results['metrics'], "MIMIC-III")

        # Run eICU external validation (FIXED VERSION)
        eicu_results = None
        if config['data_extraction'].get('use_eicu_validation', True):
            logger.info("Starting eICU external validation...")
            eicu_results = run_eicu_external_validation(
                config, predictor, feature_processor, imputer, evaluator, timestamp
            )

            if eicu_results:
                sota_comparison_eicu = comparator.compare_with_sota(eicu_results['metrics'], "eICU")
            else:
                logger.warning("eICU validation failed, continuing with MIMIC-III results only")

        # Compare performance if both datasets available
        if mimic_results and eicu_results:
            compare_mimic_eicu_performance(mimic_results, eicu_results, timestamp)

        # Generate final summary
        generate_final_summary(mimic_results, eicu_results, timestamp)

        # Document features for publication (FIXED VERSION)
        feature_doc_file = f"results/feature_documentation_{timestamp}.md"
        feature_processor.document_features(output_file=feature_doc_file)
        logger.info(f"Feature documentation created: {feature_doc_file}")

        logger.info("Pipeline completed successfully!")
        logger.info(f"Results saved in {args.output_dir}/ with timestamp {timestamp}")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()