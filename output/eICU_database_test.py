#!/usr/bin/env python3
"""
Complete eICU Testing Script
Tests eICU database access and data extraction before full pipeline integration
"""

import numpy as np
import pandas as pd
from google.cloud import bigquery
import logging
from datetime import datetime
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')


# Setup logging
def setup_logging():
    """Set up detailed logging for eICU testing"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create logs directory
    os.makedirs('logs', exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/eicu_test_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(f"eICU Testing Log initialized - {timestamp}")
    return logger


class eICUDataExtractor:
    """
    FIXED eICU Data Extractor for testing - based on actual schema analysis
    """

    def __init__(self, project_id='d1namo-ecg-processing'):
        self.project_id = project_id
        self.client = bigquery.Client(project=project_id)
        logging.info(f"BigQuery client initialized for project: {project_id}")

    def test_basic_access(self):
        """Test basic access to eICU tables"""
        logger = logging.getLogger(__name__)

        try:
            logger.info("=" * 60)
            logger.info("TESTING BASIC eICU ACCESS")
            logger.info("=" * 60)

            # Test 1: Patient table access
            query = """
            SELECT COUNT(*) as total_patients
            FROM `physionet-data.eicu_crd.patient`
            """
            result = self.client.query(query).to_dataframe()
            logger.info(f"✓ Total eICU patients: {result['total_patients'].iloc[0]:,}")

            # Test 2: Vital signs table access
            query = """
            SELECT COUNT(*) as total_vitals
            FROM `physionet-data.eicu_crd.vitalperiodic`
            """
            result = self.client.query(query).to_dataframe()
            logger.info(f"✓ Total vital sign records: {result['total_vitals'].iloc[0]:,}")

            # Test 3: Lab table access
            query = """
            SELECT COUNT(*) as total_labs
            FROM `physionet-data.eicu_crd.lab`
            """
            result = self.client.query(query).to_dataframe()
            logger.info(f"✓ Total lab records: {result['total_labs'].iloc[0]:,}")

            # Test 4: Check age distribution
            query = """
            SELECT 
                age,
                COUNT(*) as count
            FROM `physionet-data.eicu_crd.patient`
            WHERE age IS NOT NULL AND age != ''
            GROUP BY age
            ORDER BY count DESC
            LIMIT 10
            """
            result = self.client.query(query).to_dataframe()
            logger.info("✓ Age distribution (top 10):")
            for _, row in result.iterrows():
                logger.info(f"   Age '{row['age']}': {row['count']:,} patients")

            # Test 5: BP data availability
            query = """
            SELECT 
                COUNT(DISTINCT patientunitstayid) as patients_with_bp,
                COUNT(*) as total_bp_measurements,
                MIN(systemicsystolic) as min_sbp,
                MAX(systemicsystolic) as max_sbp,
                AVG(systemicsystolic) as avg_sbp
            FROM `physionet-data.eicu_crd.vitalperiodic`
            WHERE systemicsystolic IS NOT NULL 
            AND systemicdiastolic IS NOT NULL
            AND systemicsystolic > 0
            AND systemicdiastolic > 0
            """
            result = self.client.query(query).to_dataframe()
            logger.info(f"✓ Patients with BP data: {result['patients_with_bp'].iloc[0]:,}")
            logger.info(f"✓ Total BP measurements: {result['total_bp_measurements'].iloc[0]:,}")
            logger.info(
                f"✓ SBP range: {result['min_sbp'].iloc[0]:.0f} - {result['max_sbp'].iloc[0]:.0f} (avg: {result['avg_sbp'].iloc[0]:.1f})")

            # Test 6: Sample lab names
            query = """
            SELECT 
                labname,
                COUNT(*) as count
            FROM `physionet-data.eicu_crd.lab`
            WHERE labname IS NOT NULL
            GROUP BY labname
            ORDER BY count DESC
            LIMIT 15
            """
            result = self.client.query(query).to_dataframe()
            logger.info("✓ Top 15 lab test names:")
            for _, row in result.iterrows():
                logger.info(f"   {row['labname']}: {row['count']:,} tests")

            logger.info("✓ Basic eICU access test PASSED")
            return True

        except Exception as e:
            logger.error(f"Basic access test FAILED: {str(e)}")
            return False

    def extract_patient_cohort(self, max_patients=100):
        """Extract patient cohort from eICU - TESTING VERSION"""
        logger = logging.getLogger(__name__)

        query = f"""
        WITH patient_cohort AS (
            SELECT DISTINCT
                p.patientunitstayid,
                p.patienthealthsystemstayid,
                p.gender,
                -- Handle age as STRING with special values
                CASE 
                    WHEN p.age = '> 89' THEN 90
                    WHEN p.age = '' OR p.age IS NULL THEN NULL
                    ELSE SAFE_CAST(p.age AS INT64)
                END as age,
                -- Use actual discharge status fields
                CASE WHEN p.unitdischargestatus = 'Expired' THEN 1 ELSE 0 END as hospital_mortality,
                p.unittype as first_careunit,
                -- Calculate LOS properly
                CASE 
                    WHEN p.unitdischargeoffset > 0 THEN p.unitdischargeoffset / 60.0
                    ELSE NULL 
                END as icu_los_hours,
                p.ethnicity,
                p.hospitaladmitsource,
                p.admissionheight as height,
                p.admissionweight as weight,
                -- Add missing fields with defaults
                'Unknown' as insurance,
                'Unknown' as language,
                -- Additional eICU-specific fields
                p.hospitalid,
                p.wardid
            FROM `physionet-data.eicu_crd.patient` p
            WHERE (
                CASE 
                    WHEN p.age = '> 89' THEN 90
                    WHEN p.age = '' OR p.age IS NULL THEN NULL
                    ELSE SAFE_CAST(p.age AS INT64)
                END
            ) BETWEEN 18 AND 89
            AND p.unitdischargeoffset > 1440  -- At least 24 hours
            AND p.unitdischargeoffset IS NOT NULL
        )
        SELECT * FROM patient_cohort
        WHERE age IS NOT NULL
        ORDER BY patientunitstayid
        LIMIT {max_patients}
        """

        logger.info(f"Extracting eICU patient cohort (max {max_patients})...")
        result = self.client.query(query).to_dataframe()
        logger.info(f"✓ Extracted {len(result)} eICU patients")

        if len(result) > 0:
            logger.info("Sample patient data:")
            for i in range(min(3, len(result))):
                row = result.iloc[i]
                logger.info(
                    f"   Patient {row['patientunitstayid']}: {row['gender']}, age {row['age']}, LOS {row['icu_los_hours']:.1f}h")

        return result

    def extract_bp_data(self, patient_ids):
        """Extract BP measurements from eICU - TESTING VERSION"""
        logger = logging.getLogger(__name__)

        if len(patient_ids) == 0:
            logger.warning("No patient IDs provided for BP extraction")
            return pd.DataFrame()

        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH bp_measurements AS (
            SELECT 
                patientunitstayid,
                -- Use correct column names from actual schema
                AVG(CASE WHEN systemicsystolic > 30 AND systemicsystolic < 300 THEN systemicsystolic END) as sbp_mean,
                STDDEV(CASE WHEN systemicsystolic > 30 AND systemicsystolic < 300 THEN systemicsystolic END) as sbp_std,
                COUNT(CASE WHEN systemicsystolic > 30 AND systemicsystolic < 300 THEN 1 END) as sbp_count,
                MIN(CASE WHEN systemicsystolic > 30 AND systemicsystolic < 300 THEN systemicsystolic END) as sbp_min,
                MAX(CASE WHEN systemicsystolic > 30 AND systemicsystolic < 300 THEN systemicsystolic END) as sbp_max,

                AVG(CASE WHEN systemicdiastolic > 15 AND systemicdiastolic < 200 THEN systemicdiastolic END) as dbp_mean,
                STDDEV(CASE WHEN systemicdiastolic > 15 AND systemicdiastolic < 200 THEN systemicdiastolic END) as dbp_std,
                COUNT(CASE WHEN systemicdiastolic > 15 AND systemicdiastolic < 200 THEN 1 END) as dbp_count,

                AVG(CASE WHEN systemicmean > 20 AND systemicmean < 200 THEN systemicmean END) as map_mean

            FROM `physionet-data.eicu_crd.vitalperiodic`
            WHERE patientunitstayid IN ({patient_list})
            GROUP BY patientunitstayid
        )
        SELECT * FROM bp_measurements
        WHERE sbp_mean IS NOT NULL AND dbp_mean IS NOT NULL
        AND sbp_count >= 2 AND dbp_count >= 2  -- Ensure multiple measurements
        """

        logger.info(f"Extracting BP data for {len(patient_ids)} patients...")
        result = self.client.query(query).to_dataframe()
        logger.info(f"✓ Extracted BP data for {len(result)} patients")

        if len(result) > 0:
            logger.info("BP data summary:")
            logger.info(
                f"   SBP range: {result['sbp_mean'].min():.1f} - {result['sbp_mean'].max():.1f} (mean: {result['sbp_mean'].mean():.1f})")
            logger.info(
                f"   DBP range: {result['dbp_mean'].min():.1f} - {result['dbp_mean'].max():.1f} (mean: {result['dbp_mean'].mean():.1f})")
            logger.info(
                f"   Avg measurements per patient: SBP={result['sbp_count'].mean():.1f}, DBP={result['dbp_count'].mean():.1f}")

        return result

    def extract_vital_signs_features(self, patient_ids):
        """Extract vital signs from eICU - TESTING VERSION"""
        logger = logging.getLogger(__name__)

        if len(patient_ids) == 0:
            logger.warning("No patient IDs provided for vital signs extraction")
            return pd.DataFrame()

        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH vital_stats AS (
            SELECT 
                patientunitstayid,
                -- Heart Rate
                AVG(CASE WHEN heartrate > 20 AND heartrate < 250 THEN heartrate END) as hr_mean,
                STDDEV(CASE WHEN heartrate > 20 AND heartrate < 250 THEN heartrate END) as hr_std,
                MIN(CASE WHEN heartrate > 20 AND heartrate < 250 THEN heartrate END) as hr_min,
                MAX(CASE WHEN heartrate > 20 AND heartrate < 250 THEN heartrate END) as hr_max,

                -- Respiratory Rate
                AVG(CASE WHEN respiration > 5 AND respiration < 60 THEN respiration END) as rr_mean,
                STDDEV(CASE WHEN respiration > 5 AND respiration < 60 THEN respiration END) as rr_std,

                -- Temperature (already in Fahrenheit in eICU)
                AVG(CASE WHEN temperature > 80 AND temperature < 110 THEN temperature END) as temp_mean,

                -- SpO2
                AVG(CASE WHEN sao2 > 50 AND sao2 <= 100 THEN sao2 END) as spo2_mean,
                STDDEV(CASE WHEN sao2 > 50 AND sao2 <= 100 THEN sao2 END) as spo2_std,

                -- Additional eICU vitals
                AVG(CASE WHEN cvp > 0 AND cvp < 50 THEN cvp END) as cvp_mean,
                AVG(CASE WHEN etco2 > 10 AND etco2 < 80 THEN etco2 END) as etco2_mean,

                -- Count valid measurements
                COUNT(CASE WHEN heartrate > 20 AND heartrate < 250 THEN 1 END) as hr_count,
                COUNT(CASE WHEN respiration > 5 AND respiration < 60 THEN 1 END) as rr_count,
                COUNT(CASE WHEN sao2 > 50 AND sao2 <= 100 THEN 1 END) as spo2_count

            FROM `physionet-data.eicu_crd.vitalperiodic`
            WHERE patientunitstayid IN ({patient_list})
            GROUP BY patientunitstayid
        )
        SELECT * FROM vital_stats
        WHERE hr_mean IS NOT NULL OR rr_mean IS NOT NULL OR spo2_mean IS NOT NULL
        """

        logger.info(f"Extracting vital signs for {len(patient_ids)} patients...")
        result = self.client.query(query).to_dataframe()
        logger.info(f"✓ Extracted vital signs for {len(result)} patients")

        if len(result) > 0:
            logger.info("Vital signs summary:")
            hr_valid = result['hr_mean'].notna().sum()
            rr_valid = result['rr_mean'].notna().sum()
            temp_valid = result['temp_mean'].notna().sum()
            spo2_valid = result['spo2_mean'].notna().sum()

            logger.info(f"   HR data: {hr_valid}/{len(result)} patients")
            logger.info(f"   RR data: {rr_valid}/{len(result)} patients")
            logger.info(f"   Temp data: {temp_valid}/{len(result)} patients")
            logger.info(f"   SpO2 data: {spo2_valid}/{len(result)} patients")

            if hr_valid > 0:
                hr_data = result['hr_mean'].dropna()
                logger.info(f"   HR range: {hr_data.min():.1f} - {hr_data.max():.1f} (mean: {hr_data.mean():.1f})")

        return result

    def extract_lab_features(self, patient_ids):
        """Extract lab features from eICU - TESTING VERSION"""
        logger = logging.getLogger(__name__)

        if len(patient_ids) == 0:
            logger.warning("No patient IDs provided for lab extraction")
            return pd.DataFrame()

        patient_list = ','.join([str(id) for id in patient_ids])

        query = f"""
        WITH lab_stats AS (
            SELECT 
                patientunitstayid,
                -- Use flexible lab name matching
                AVG(CASE WHEN LOWER(labname) LIKE '%sodium%' AND labresult BETWEEN 120 AND 180 THEN labresult END) as sodium_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%potassium%' AND labresult BETWEEN 2.0 AND 8.0 THEN labresult END) as potassium_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%creatinine%' AND labresult BETWEEN 0.3 AND 15.0 THEN labresult END) as creatinine_mean,
                AVG(CASE WHEN (LOWER(labname) LIKE '%bun%' OR LOWER(labname) LIKE '%urea%') AND labresult BETWEEN 5 AND 200 THEN labresult END) as bun_mean,
                AVG(CASE WHEN (LOWER(labname) LIKE '%hemoglobin%' OR LOWER(labname) LIKE '%hgb%') AND labresult BETWEEN 3 AND 20 THEN labresult END) as hemoglobin_mean,
                AVG(CASE WHEN (LOWER(labname) LIKE '%wbc%' OR LOWER(labname) LIKE '%white blood%') AND labresult BETWEEN 1 AND 100 THEN labresult END) as wbc_mean,
                AVG(CASE WHEN LOWER(labname) LIKE '%lactate%' AND labresult BETWEEN 0.3 AND 30.0 THEN labresult END) as lactate_mean,

                -- Count available labs
                COUNT(DISTINCT CASE WHEN LOWER(labname) LIKE '%sodium%' THEN labname END) as sodium_tests,
                COUNT(DISTINCT CASE WHEN LOWER(labname) LIKE '%potassium%' THEN labname END) as potassium_tests,
                COUNT(DISTINCT CASE WHEN LOWER(labname) LIKE '%creatinine%' THEN labname END) as creatinine_tests,
                COUNT(DISTINCT CASE WHEN LOWER(labname) LIKE '%lactate%' THEN labname END) as lactate_tests

            FROM `physionet-data.eicu_crd.lab`
            WHERE patientunitstayid IN ({patient_list})
            AND labresult IS NOT NULL
            AND labresult > 0
            GROUP BY patientunitstayid
        )
        SELECT * FROM lab_stats
        """

        logger.info(f"Extracting lab data for {len(patient_ids)} patients...")
        result = self.client.query(query).to_dataframe()
        logger.info(f"✓ Extracted lab data for {len(result)} patients")

        if len(result) > 0:
            logger.info("Lab data availability:")
            sodium_valid = result['sodium_mean'].notna().sum()
            creatinine_valid = result['creatinine_mean'].notna().sum()
            lactate_valid = result['lactate_mean'].notna().sum()

            logger.info(f"   Sodium: {sodium_valid}/{len(result)} patients")
            logger.info(f"   Creatinine: {creatinine_valid}/{len(result)} patients")
            logger.info(f"   Lactate: {lactate_valid}/{len(result)} patients")

        return result

    def run_comprehensive_test(self, max_patients=50):
        """Run comprehensive eICU extraction test"""
        logger = logging.getLogger(__name__)

        try:
            logger.info("=" * 60)
            logger.info("COMPREHENSIVE eICU EXTRACTION TEST")
            logger.info("=" * 60)

            # Step 1: Basic access test
            if not self.test_basic_access():
                logger.error("Basic access test failed - stopping")
                return False

            # Step 2: Extract patient cohort
            logger.info(f"\nStep 2: Extracting patient cohort (max {max_patients})...")
            cohort_df = self.extract_patient_cohort(max_patients)

            if len(cohort_df) == 0:
                logger.error("No patients extracted - stopping")
                return False

            logger.info(f"✓ Cohort extraction successful: {len(cohort_df)} patients")

            # Step 3: Extract BP data
            logger.info("\nStep 3: Extracting BP data...")
            patient_ids = cohort_df['patientunitstayid'].tolist()
            bp_df = self.extract_bp_data(patient_ids)

            if len(bp_df) == 0:
                logger.error("No BP data extracted - this is critical")
                return False

            logger.info(f"✓ BP extraction successful: {len(bp_df)} patients with BP data")

            # Step 4: Extract vital signs
            logger.info("\nStep 4: Extracting vital signs...")
            vitals_df = self.extract_vital_signs_features(patient_ids)
            logger.info(f"✓ Vitals extraction: {len(vitals_df)} patients")

            # Step 5: Extract lab data
            logger.info("\nStep 5: Extracting lab data...")
            labs_df = self.extract_lab_features(patient_ids)
            logger.info(f"✓ Lab extraction: {len(labs_df)} patients")

            # Step 6: Summary
            logger.info("\n" + "=" * 60)
            logger.info("EXTRACTION SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Patient cohort: {len(cohort_df)} patients")
            logger.info(f"BP data: {len(bp_df)} patients ({len(bp_df) / len(cohort_df) * 100:.1f}%)")
            logger.info(f"Vital signs: {len(vitals_df)} patients ({len(vitals_df) / len(cohort_df) * 100:.1f}%)")
            logger.info(f"Lab data: {len(labs_df)} patients ({len(labs_df) / len(cohort_df) * 100:.1f}%)")

            # Step 7: Test data merging
            logger.info("\nStep 7: Testing data merging...")

            # Merge BP data with cohort
            merged_df = cohort_df.merge(bp_df, on='patientunitstayid', how='inner')
            logger.info(f"✓ After BP merge: {len(merged_df)} patients")

            # Merge vital signs
            merged_df = merged_df.merge(vitals_df, on='patientunitstayid', how='left')
            logger.info(f"✓ After vitals merge: {len(merged_df)} patients")

            # Merge lab data
            merged_df = merged_df.merge(labs_df, on='patientunitstayid', how='left')
            logger.info(f"✓ After lab merge: {len(merged_df)} patients")

            logger.info(f"\nFinal merged dataset: {merged_df.shape}")
            logger.info("Column summary:")
            logger.info(f"   Demographics: patientunitstayid, gender, age, ethnicity")
            logger.info(f"   BP variables: sbp_mean, sbp_std, dbp_mean, dbp_std")
            logger.info(f"   Vital signs: hr_mean, rr_mean, temp_mean, spo2_mean")
            logger.info(f"   Lab values: sodium_mean, creatinine_mean, lactate_mean, etc.")

            # Save test results
            logger.info("\nSaving test results...")
            merged_df.to_csv('eicu_test_results.csv', index=False)
            logger.info("✓ Test data saved to 'eicu_test_results.csv'")

            logger.info("\n" + "=" * 60)
            logger.info("✅ COMPREHENSIVE eICU TEST PASSED!")
            logger.info("eICU integration is ready for full pipeline")
            logger.info("=" * 60)

            return True

        except Exception as e:
            logger.error(f"Comprehensive test FAILED: {str(e)}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return False


def main():
    """Main function to run eICU testing"""

    # Setup logging
    logger = setup_logging()

    try:
        logger.info("Starting eICU Database Testing")
        logger.info("=" * 60)

        # Initialize extractor
        extractor = eICUDataExtractor(project_id='d1namo-ecg-processing')

        # Run comprehensive test
        success = extractor.run_comprehensive_test(max_patients=100)

        if success:
            logger.info("\n🎉 eICU testing completed successfully!")
            logger.info("You can now proceed with full pipeline integration")
        else:
            logger.error("\n❌ eICU testing failed!")
            logger.error("Check the errors above and fix before proceeding")

    except Exception as e:
        logger.error(f"Test script failed: {str(e)}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()