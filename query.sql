WITH base AS (
  -- Basic ICU + patient demographics and admission info
  SELECT
    i.subject_id AS patient_id,
    i.hadm_id AS admission_id,
    i.stay_id AS icu_stay_id,
    i.first_careunit AS first_icu_careunit,
    i.last_careunit AS last_icu_careunit,
    i.intime AS icu_intime,
    i.outtime AS icu_outtime,
    i.los AS icu_length_of_stay_in_days,
    TIMESTAMP_DIFF(i.outtime, i.intime, MINUTE) AS icu_length_of_stay_in_minutes,
    a.hospital_expire_flag AS patient_expired_in_hospital, 
    p.gender AS patient_gender, 
    p.anchor_age AS patient_age, 
    p.anchor_year AS patient_year, 
    p.anchor_year_group AS patient_year_group, 
    p.dod AS patient_date_of_death,
    a.admission_type as admission_type, 
    a.admission_location as admission_location, 
    a.discharge_location as discharge_location, 
    a.insurance as insurance, 
    a.language as native_language, 
    a.marital_status as marital_status, 
    a.race as patient_race, 
    a.deathtime AS patient_death_time
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
    ON i.subject_id = p.subject_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON i.hadm_id = a.hadm_id
),

meds AS (
  SELECT
    i.stay_id,
    STRING_AGG(
      CONCAT(
        'Start Time:', COALESCE(CAST(p.starttime AS STRING), ''),
        '|Stop Time:', COALESCE(CAST(p.stoptime AS STRING), ''),
        '|Drug Type:', COALESCE(p.drug_type, ''),
        '|Drug:', COALESCE(p.drug, ''),
        '|Drug Ontology:', COALESCE(p.formulary_drug_cd, ''),
        '|Composition Strength of Prescribed Medicine:', COALESCE(p.prod_strength, ''),
        '|Drug Form:', COALESCE(p.form_rx, ''),
        '|Prescribed Dose for Patient:', COALESCE(p.dose_val_rx, ''),
        '|Unit Measurement of Prescribed Dose:', COALESCE(p.dose_unit_rx, ''),
        '|Amount of Medicine in Single Formulary Dose:', COALESCE(p.form_val_disp, ''),
        '|Amount of Medicine in Formulary Dose:', COALESCE(p.form_unit_disp, ''),
        '|Number of Doses per 24 Hours:', COALESCE(CAST(p.doses_per_24_hrs AS STRING), ''),
        '|Route of Administration:', COALESCE(p.route, ''),
        '|Administered in ICU:', CASE 
                   WHEN p.starttime BETWEEN i.intime AND i.outtime THEN 'True'
                   ELSE 'False'
                 END
      ),
      '\n' ORDER BY p.starttime
    ) AS chronology_of_medications
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.prescriptions` p
    ON i.subject_id = p.subject_id
  GROUP BY i.stay_id
),

diagnoses AS (
  SELECT
    icu.stay_id,
    STRING_AGG(DISTINCT dd.long_title, '\n' ORDER BY dd.long_title) AS diagnosis
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` di
    ON icu.hadm_id = di.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_diagnoses` dd
    ON di.icd_code = dd.icd_code
   AND di.icd_version = dd.icd_version
  GROUP BY icu.stay_id
),

procedures AS (
  SELECT
    icu.stay_id,
    STRING_AGG(
      CONCAT(
        COALESCE(dp.long_title, 'Unknown Procedure'),
        ' (', CAST(pi.chartdate AS STRING), ')'
      ),
      '\n' ORDER BY pi.chartdate
    ) AS procedure_list
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.procedures_icd` pi
    ON icu.subject_id = pi.subject_id
   AND icu.hadm_id = pi.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.d_icd_procedures` dp
    ON pi.icd_code = dp.icd_code
   AND pi.icd_version = dp.icd_version
  GROUP BY icu.stay_id
),

procedure_events_cte AS (
  SELECT
    icu.stay_id,
    STRING_AGG(
      CONCAT(
        'Start Time:', COALESCE(CAST(pe.starttime AS STRING), ''),
        '|End Time:', COALESCE(CAST(pe.endtime AS STRING), ''),
        '|Event Recorded Time in Patient File:', COALESCE(CAST(pe.storetime AS STRING), ''),
        '|Duration of Procedure:', COALESCE(CAST(pe.value AS STRING), ''),
        '|Unit of Measurement of Duration of Procedure:', COALESCE(pe.valueuom, ''),
        '|Location of Procedure on Patient Body:', COALESCE(pe.location, ''),
        '|Location Category of Procedure:', COALESCE(pe.locationcategory, ''),
        '|Patient Weight (in Kilograms):', COALESCE(CAST(pe.patientweight AS STRING), ''),
        '|Ultimate Procedure Status:', COALESCE(pe.statusdescription, '')
      ),
      '\n' ORDER BY pe.starttime
    ) AS procedure_event_list
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_icu.procedureevents` pe
    ON icu.stay_id = pe.stay_id
   AND icu.subject_id = pe.subject_id
   AND icu.hadm_id = pe.hadm_id
  GROUP BY icu.stay_id
),

microbiology AS (
  SELECT
    icu.stay_id,
    STRING_AGG(
      CONCAT(
        'Sample Collection Time:', COALESCE(CAST(m.chartdate AS STRING), CAST(m.charttime AS STRING)),
        '|Specimen Type:', COALESCE(m.spec_type_desc, ''),
        '|Sample Received Time:', COALESCE(CAST(m.storedate AS STRING), CAST(m.storetime AS STRING)),
        '|Microbiology Test Name:', COALESCE(m.test_name, ''),
        '|Organism Name:', COALESCE(m.org_name, ''),
        '|Antibiotic Name (Isolated Colony):', COALESCE(m.ab_name, ''),
        '|Interpretation:', COALESCE(m.interpretation, '')
      ),
      '\n' ORDER BY m.charttime
    ) AS microbiology_cultures
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
    ON icu.subject_id = m.subject_id
   AND icu.hadm_id = m.hadm_id
  GROUP BY icu.stay_id
),

labevents_cte AS (
  SELECT
    icu.stay_id,
    STRING_AGG(
      CONCAT(
        'Sample Collection Time:', COALESCE(CAST(le.charttime AS STRING), ''),
        '|Sample Received Time:', COALESCE(CAST(le.storetime AS STRING), ''),
        '|Value:', COALESCE(CAST(le.value AS STRING), CAST(le.valuenum AS STRING), ''),
        '|Unit of Measurement of Value:', COALESCE(le.valueuom, ''),
        '|Reference Range Lower:', COALESCE(CAST(le.ref_range_lower AS STRING), ''),
        '|Reference Range Upper:', COALESCE(CAST(le.ref_range_upper AS STRING), ''),
        '|Abnormal Flag:', CASE WHEN le.flag IS NOT NULL AND le.flag <> '' THEN 'True' ELSE 'False' END,
        '|Priority:', COALESCE(le.priority, ''),
        '|Comments:', COALESCE(le.comments, '')
      ),
      '\n' ORDER BY le.charttime
    ) AS labevents
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.labevents` le
    ON icu.subject_id = le.subject_id
   AND icu.hadm_id = le.hadm_id
  GROUP BY icu.stay_id
),

ingredient_events_cte AS (
  SELECT
    icu.stay_id,
    STRING_AGG(
      CONCAT(
        'Start Time:', COALESCE(CAST(ie.starttime AS STRING), ''),
        '|End Time:', COALESCE(CAST(ie.endtime AS STRING), ''),
        '|Input Ingredient:', 
            CASE 
              WHEN ie.amount IS NOT NULL THEN CONCAT(CAST(ie.amount AS STRING), ' ', COALESCE(ie.amountuom, ''))
              ELSE ''
            END,
        '|Rate:', 
            CASE 
              WHEN ie.rate IS NOT NULL THEN CONCAT(CAST(ie.rate AS STRING), ' ', COALESCE(ie.rateuom, ''))
              ELSE 'N/A'
            END,
        '|Status Description:', COALESCE(ie.statusdescription, '')
      ),
      '\n' ORDER BY ie.starttime
    ) AS ingredient_events
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_icu.ingredientevents` ie
    ON icu.subject_id = ie.subject_id
   AND icu.stay_id = ie.stay_id
   AND icu.hadm_id = ie.hadm_id
  GROUP BY icu.stay_id
)

SELECT
  b.*,
  m.chronology_of_medications,
  d.diagnosis,
  pr.procedure_list,
  pe.procedure_event_list,
  mc.microbiology_cultures,
  lab.labevents,
  ing.ingredient_events
FROM base b
LEFT JOIN meds m
  ON b.icu_stay_id = m.stay_id
LEFT JOIN diagnoses d
  ON b.icu_stay_id = d.stay_id
LEFT JOIN procedures pr
  ON b.icu_stay_id = pr.stay_id
LEFT JOIN procedure_events_cte pe
  ON b.icu_stay_id = pe.stay_id
LEFT JOIN microbiology mc
  ON b.icu_stay_id = mc.stay_id
LEFT JOIN labevents_cte lab
  ON b.icu_stay_id = lab.stay_id
LEFT JOIN ingredient_events_cte ing
  ON b.icu_stay_id = ing.stay_id;


____________________________________________________________________________________

-- STRUCTURED per-stay export: each stay has an ARRAY of ordered event STRUCTs
WITH base AS (
  SELECT
    i.subject_id        AS patient_id,
    i.hadm_id           AS admission_id,
    i.stay_id           AS icu_stay_id,
    i.first_careunit    AS first_icu_careunit,
    i.last_careunit     AS last_icu_careunit,
    i.intime            AS icu_intime,
    i.outtime           AS icu_outtime,
    i.los               AS icu_length_of_stay_in_days,
    TIMESTAMP_DIFF(i.outtime, i.intime, MINUTE) AS icu_length_of_stay_in_minutes,
    a.hospital_expire_flag AS patient_expired_in_hospital, 
    p.gender            AS patient_gender, 
    p.anchor_age        AS patient_age, 
    p.anchor_year       AS patient_year, 
    p.anchor_year_group AS patient_year_group, 
    p.dod               AS patient_date_of_death,
    a.admission_type    AS admission_type, 
    a.admission_location AS admission_location, 
    a.discharge_location AS discharge_location, 
    a.insurance         AS insurance, 
    a.language          AS native_language, 
    a.marital_status    AS marital_status, 
    a.race              AS patient_race, 
    a.deathtime         AS patient_death_time,
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` p
    ON i.subject_id = p.subject_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON i.hadm_id = a.hadm_id
),

-- union all event sources into one canonical events table
events_union AS (
  -- prescriptions
  SELECT
    i.stay_id as icu_stay_id, 
    p.starttime AS event_time,
    'prescription' AS event_type,
    TRIM(CONCAT(
    'Drug:', IFNULL(p.drug, ''), 
    ' |Strength:', IFNULL(p.prod_strength, ''), 
    ' |Form:', IFNULL(p.form_rx, ''), 
    ' |Ontology:', IFNULL(p.formulary_drug_cd, ''), 
    ' |Dose:', IFNULL(p.dose_val_rx, '')
    )) AS event_text,
    TIMESTAMP_DIFF(
      CASE WHEN a.hospital_expire_flag = 1 THEN a.deathtime ELSE i.outtime END,
      p.starttime,
      MINUTE
    ) AS relative_time_to_final_event
  FROM `physionet-data.mimiciv_3_1_icu.icustays` i
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.prescriptions` p
    ON i.subject_id = p.subject_id
   AND i.hadm_id = p.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a
    ON i.hadm_id = a.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` ptd
    ON i.subject_id = ptd.subject_id
  WHERE p.starttime IS NOT NULL

  UNION ALL

  -- procedure events (procedureevents table)
  SELECT
    icu.stay_id as icu_stay_id, 
    pe.starttime AS event_time,
    'procedure_event' AS event_type,
    TRIM(CONCAT(
    'Start Time:', IFNULL(CAST(pe.starttime AS STRING), ''), 
    ' |End Time:', IFNULL(CAST(pe.endtime AS STRING), ''), 
    ' |Event Recorded Time in Patient File:', IFNULL(CAST(pe.storetime AS STRING), ''), 
    ' |Duration of Procedure:', IFNULL(CAST(pe.value AS STRING), ''), 
    ' |Unit of Measurement of Duration of Procedure:', IFNULL(pe.valueuom, ''), 
    ' |Location of Procedure on Patient Body:', IFNULL(pe.location, ''), 
    ' |Location Category of Procedure:', IFNULL(pe.locationcategory, ''), 
    ' |Patient Weight (in Kilograms):', IFNULL(CAST(pe.patientweight AS STRING), ''), 
    ' |Ultimate Procedure Status:', IFNULL(pe.statusdescription, '')
  )) AS event_text,
    TIMESTAMP_DIFF(
      CASE WHEN adm.hospital_expire_flag = 1 THEN adm.deathtime ELSE icu.outtime END,
      pe.starttime,
      MINUTE
    ) AS relative_time_to_final_event
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_icu.procedureevents` pe
    ON icu.subject_id = pe.subject_id
   AND icu.stay_id = pe.stay_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm
    ON icu.hadm_id = adm.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` pt
    ON icu.subject_id = pt.subject_id
  WHERE pe.starttime IS NOT NULL

  UNION ALL

  -- lab events
  SELECT
    icu.stay_id as icu_stay_id, 
    le.charttime AS event_time,
    'labevent' AS event_type,
     TRIM(CONCAT(
    'Sample Collection Time:', COALESCE(CAST(le.charttime AS STRING), ''),
    '|Sample Received Time:', COALESCE(CAST(le.storetime AS STRING), ''),
    '|Lab Result Value:', COALESCE(CAST(le.value AS STRING), CAST(le.valuenum AS STRING), ''),
    '|Unit of Measurement:', COALESCE(le.valueuom, ''),
    '|Reference Range Lower Bound:', COALESCE(CAST(le.ref_range_lower AS STRING), ''),
    '|Reference Range Upper Bound:', COALESCE(CAST(le.ref_range_upper AS STRING), ''),
    '|Abnormal Result Flag:', CASE WHEN le.flag IS NOT NULL AND le.flag <> '' THEN 'True' ELSE 'False' END,
    '|Result Priority:', COALESCE(le.priority, ''),
    '|Clinician Comments:', COALESCE(le.comments, '')
  )) AS event_text,
    TIMESTAMP_DIFF(
      CASE WHEN adm.hospital_expire_flag = 1 THEN adm.deathtime  ELSE icu.outtime END,
      le.charttime,
      MINUTE
    ) AS relative_time_to_final_event
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.labevents` le
    ON icu.subject_id = le.subject_id
   AND icu.hadm_id = le.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm
    ON icu.hadm_id = adm.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` pt
    ON icu.subject_id = pt.subject_id
  WHERE le.charttime IS NOT NULL

  UNION ALL

  -- microbiology events
  SELECT
    icu.stay_id as icu_stay_id, 
    COALESCE(m.chartdate, m.charttime) AS event_time,
    'microbiology' AS event_type,
    TRIM(
      CONCAT(
        'Sample Collection Time:', COALESCE(CAST(m.chartdate AS STRING), CAST(m.charttime AS STRING)),
        '|Specimen Type:', COALESCE(m.spec_type_desc, ''),
        '|Sample Received Time:', COALESCE(CAST(m.storedate AS STRING), CAST(m.storetime AS STRING)),
        '|Microbiology Test Name:', COALESCE(m.test_name, ''),
        '|Organism Name:', COALESCE(m.org_name, ''),
        '|Antibiotic Name (Isolated Colony):', COALESCE(m.ab_name, ''),
        '|Interpretation:', COALESCE(m.interpretation, '')
      )
    ) AS event_text,
    TIMESTAMP_DIFF(
      CASE WHEN adm.hospital_expire_flag = 1 THEN adm.deathtime ELSE icu.outtime END,
      COALESCE(m.chartdate, m.charttime),
      MINUTE
    ) AS relative_time_to_final_event
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.microbiologyevents` m
    ON icu.subject_id = m.subject_id
   AND icu.hadm_id = m.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm
    ON icu.hadm_id = adm.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` pt
    ON icu.subject_id = pt.subject_id
  WHERE COALESCE(m.chartdate, m.charttime) IS NOT NULL

  UNION ALL

  -- ingredientevents (IV fluids, nutrition)
  SELECT
    icu.stay_id as icu_stay_id, 
    ie.starttime AS event_time,
    'ingredient' AS event_type,
    TRIM(
    CONCAT(
        'Start Time:', COALESCE(CAST(ie.starttime AS STRING), ''),
        '|End Time:', COALESCE(CAST(ie.endtime AS STRING), ''),
        '|Input Ingredient:', 
            CASE 
              WHEN ie.amount IS NOT NULL THEN CONCAT(CAST(ie.amount AS STRING), ' ', COALESCE(ie.amountuom, ''))
              ELSE ''
            END,
        '|Rate:', 
            CASE 
              WHEN ie.rate IS NOT NULL THEN CONCAT(CAST(ie.rate AS STRING), ' ', COALESCE(ie.rateuom, ''))
              ELSE 'N/A'
            END,
        '|Status Description:', COALESCE(ie.statusdescription, '')
      )) AS event_text,
    TIMESTAMP_DIFF(
      CASE WHEN adm.hospital_expire_flag = 1 THEN adm.deathtime ELSE icu.outtime END,
      ie.starttime,
      MINUTE
    ) AS relative_time_to_final_event
  FROM `physionet-data.mimiciv_3_1_icu.icustays` icu
  LEFT JOIN `physionet-data.mimiciv_3_1_icu.ingredientevents` ie
    ON icu.subject_id = ie.subject_id
   AND icu.stay_id = ie.stay_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.admissions` adm
    ON icu.hadm_id = adm.hadm_id
  LEFT JOIN `physionet-data.mimiciv_3_1_hosp.patients` pt
    ON icu.subject_id = pt.subject_id
  WHERE ie.starttime IS NOT NULL
),

-- aggregate events into ordered ARRAY<STRUCT(...)> per stay
events_agg AS (
  SELECT
    icu_stay_id,
    ARRAY_AGG(
      STRUCT(
        event_time,
        event_type,
        event_text,
        relative_time_to_final_event
      )
      ORDER BY event_time
    ) AS events
  FROM events_union
  GROUP BY icu_stay_id
)

SELECT
  b.*,
  e.events
FROM base b
LEFT JOIN events_agg e
  ON b.icu_stay_id = e.icu_stay_id;
