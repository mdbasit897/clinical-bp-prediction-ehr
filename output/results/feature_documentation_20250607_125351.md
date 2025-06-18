# Feature Documentation for BP Prediction Models

This document describes the features used in the blood pressure prediction models.

## Demographics

Patient demographic information including age, gender, ethnicity, and primary language.

Features in this category:

- `age`
- `gender`
- `ethnicity`
- `language`

## Vital Signs

Time-averaged vital sign measurements including heart rate, respiratory rate, oxygen saturation, and temperature.

Features in this category:

- `hr_mean`
- `hr_std`
- `rr_mean`
- `rr_std`
- `spo2_mean`
- `temp_mean`

## Blood Pressure

Blood pressure measurements from verification sources. **Note: These features should be used with caution due to potential data leakage.**

Features in this category:

- `sbp_mean_verify`
- `sbp_std_verify`
- `dbp_mean_verify`
- `dbp_std_verify`

## Laboratory

Common laboratory values with clinical relevance to hemodynamics and overall patient status.

Features in this category:

- `sodium_mean`
- `potassium_mean`
- `hemoglobin_mean`
- `wbc_mean`
- `lactate_mean`

## Blood Gas

Arterial blood gas measurements reflecting acid-base status and oxygenation.

Features in this category:

- `ph_mean`
- `pco2_mean`
- `po2_mean`

## Liver Function

Laboratory markers of liver function.

Features in this category:

- `alp_mean`
- `ast_mean`
- `alt_mean`

## Temporal

Features capturing the temporal dynamics of physiological measurements.

Features in this category:

- `hr_trend`
- `sbp_lability_events`

## Interaction

Engineered features representing clinically meaningful interactions between variables.

Features in this category:

- `age_creatinine_product`
- `vasopressor_lactate_interaction`

## Derived

Features mathematically derived from other measurements to capture non-linear relationships.

Features in this category:

- `bun_creatinine_ratio`
- `perfusion_index`
- `lactate_mean_squared`
- `lactate_mean_sqrt`

## Feature Importance Warning

The features `sbp_mean_verify` and `dbp_mean_verify` show extremely high importance in the model, which may indicate potential data leakage. Models should be evaluated both with and without these features to ensure robust performance.
