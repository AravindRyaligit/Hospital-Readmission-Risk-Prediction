# Data Directory

This directory contains the dataset files for the Hospital Readmission Prediction System.

## Required Files

Place your dataset file here:
- `hospital_readmissions.csv` - Main dataset with patient records

## Data Source

The recommended dataset is the **Diabetes 130-US Hospitals Dataset** from the UCI Machine Learning Repository:
- URL: https://archive.ics.uci.edu/ml/datasets/Diabetes+130-US+hospitals+for+years+1999-2008

## Data Format

The CSV file should contain the following columns:
- `age` - Age bracket
- `time_in_hospital` - Days in hospital (1-14)
- `n_lab_procedures` - Number of lab procedures
- `n_procedures` - Number of procedures
- `n_medications` - Number of medications
- `n_outpatient` - Outpatient visits in past year
- `n_inpatient` - Inpatient visits in past year
- `n_emergency` - Emergency visits in past year
- `medical_specialty` - Medical specialty
- `diag_1` - Primary diagnosis
- `diag_2` - Secondary diagnosis
- `diag_3` - Additional diagnosis
- `glucose_test` - Glucose test result
- `A1Ctest` - A1C test result
- `change` - Medication change
- `diabetes_med` - Diabetes medication
- `readmitted` - Target variable (yes/no)

## Note

The actual data files are excluded from Git for privacy and size reasons. Download the dataset from the source above and place it in this directory before running the preprocessing pipeline.
