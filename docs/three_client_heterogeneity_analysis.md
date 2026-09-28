# Three-Client Heterogeneity Analysis and Validation Report: NIH, CheXpert, and PadChest

**Project Title:** Adaptive Federated Aggregation Algorithm for Disease Prediction using Heterogeneous Healthcare Data  
**Target Disease:** Pleural Effusion  
**Federated Clients:**
1. `client_01_NIH`: NIH ChestX-ray14 (Bethesda, MD, USA)
2. `client_02_CheXpert`: Stanford Hospital (Stanford, CA, USA)
3. `client_03_PadChest`: Hospital Universitario de San Juan de Alicante (Alicante, Spain)

---

## Executive Summary

This document provides both the **preserved initial comparative analysis** and the **authoritative validated audit** for the three clinical nodes in our federated learning network. All three datasets have been standardized into a unified, leak-free **17-column client schema**, encompassing a total of **255,738 chest radiographs across 100,344 unique patients** and **143,295+ unique clinical studies**.

To ensure scientific rigor prior to federated model training, this report explicitly audits the initial comparative claims, flags naive or misleading metrics, separates empirical observations from unsupported causal assertions, and establishes the definitive statistical baseline for federated learning experiment design.

```
Federated Testbed Summary:
  Total Clean Images:   255,738
  Total Patients:       100,344
  Total Studies:        143,295+
  Federation Partition: Train: 211,235 (82.6%) | Val: 23,824 (9.3%) | Test: 20,679 (8.1%)
  Participating Nodes:  NIH (28.81%) | CheXpert (47.63%) | PadChest (23.56%)
```

---

# PART I: Preserved Original Three-Client Comparative Analysis (Initial Baseline)

> [!NOTE]
> **Context:** The analysis below reflects the initial comparative report generated immediately following standardized client preprocessing. It is preserved here in full for institutional continuity and auditability. Specific naive metrics and unverified causal claims are marked with audit warning callouts.

### 1. Global Sizing & Partition Distribution (Original)

| Metric | NIH (Client 01) | CheXpert (Client 02) | PadChest (Client 03) | Total Federation |
| :--- | :---: | :---: | :---: | :---: |
| **Institution / Origin** | NIH Clinical Center (USA) | Stanford Hospital (USA) | Hosp. Univ. San Juan (Spain) | 3 Health Systems |
| **Total Images** | 73,678 | 121,817 | 60,243 | **255,738** |
| **Unique Patients** | 25,923 | 41,009 | 33,412 | **100,344** |
| **Unique Studies** | *Unavailable (NaN)* | 100,728 | 42,567 | **143,295+** |
| **Images / Patient Ratio** | 2.84 | 2.97 | 1.80 | **2.55** |
| **Federation Share (%)** | 28.81% | 47.63% | 23.56% | 100.0% |

#### Partition Breakdown (Original)
* **NIH:** Preserves the official NIH test split (`test_list.txt`); the official train/val portion is split 90/10 at the patient level.
  * Train: **53,323** images (20,977 patients)
  * Validation: **5,836** images (2,331 patients)
  * Test: **14,519** images (2,615 patients)
* **CheXpert:** Preserves Stanford's official radiologist-consensus validation benchmark (`valid.csv`) as held-out test; training data split 90/10 at the patient level.
  * Train: **109,572** images (36,728 patients)
  * Validation: **12,011** images (4,081 patients)
  * Test: **234** images (200 patients)
* **PadChest:** Patient-level 80/10/10 split with joint patient-study encounter clustering.
  * Train: **48,340** images (26,729 patients)
  * Validation: **5,977** images (3,341 patients)
  * Test: **5,926** images (3,342 patients)

---

### 2. Label Distribution & Class Imbalance Comparison (Original)

| Split / Client | Total Images | Positive (`target=1`) | Negative (`target=0`) | Positive Ratio (%) |
| :--- | :---: | :---: | :---: | :---: |
| **NIH — Train** | 53,323 | 7,842 | 45,481 | 14.71% |
| **NIH — Val** | 5,836 | 817 | 5,019 | 14.00% |
| **NIH — Test** | 14,519 | 4,658 | 9,861 | 32.08% |
| **NIH — Overall** | **73,678** | **13,317** | **60,361** | **18.07%** |
| **CheXpert — Train** | 109,572 | 77,719 | 31,853 | 70.93% |
| **CheXpert — Val** | 12,011 | 8,468 | 3,543 | 70.50% |
| **CheXpert — Test** | 234 | 67 | 167 | 28.63% |
| **CheXpert — Overall** | **121,817** | **86,254** | **35,563** | **70.81%** |
| **PadChest — Train** | 48,340 | 7,934 | 40,406 | 16.41% |
| **PadChest — Val** | 5,977 | 965 | 5,012 | 16.15% |
| **PadChest — Test** | 5,926 | 954 | 4,972 | 16.10% |
| **PadChest — Overall** | **60,243** | **9,853** | **50,390** | **16.36%** |

> [!WARNING]
> **Audit Note on Initial Causal Claim (AUDIT-01):** The initial analysis claimed that CheXpert's ~71% positive prevalence was *"driven by high-acuity inpatient/ICU sampling."* While Stanford Medical Center does treat high-acuity inpatients, Part II demonstrates that this 70.81% rate is primarily an artifact of our filtering protocol (excluding 90,203 unmentioned cases and 11,628 uncertain cases).

---

### 3. Patient Demographics: Age and Sex (Original)

| Demographic Feature | NIH | CheXpert | PadChest |
| :--- | :---: | :---: | :---: |
| **Age Mean $\pm$ Std** | $46.49 \pm 16.78$ years | $61.03 \pm 17.52$ years | $51.24 \pm 20.03$ years |
| **Age Median [IQR]** | 48.0 [34.0, 59.0] | 62.0 [50.0, 74.0] | 52.0 [38.0, 66.0] |
| **Age Range** | 1.0 – 95.0+* | 18.0 – 90.0 | 0.0 – 104.0 |
| **Sex: Male (%)** | **56.13%** (41,357) | **59.11%** (72,002) | **44.91%** (27,053) |
| **Sex: Female (%)** | **43.87%** (32,321) | **40.89%** (49,814) | **55.08%** (33,180) |
| **Sex: Unknown / Other** | 0.00% (0) | <0.01% (1) | 0.01% (6) |

*\*Note: NIH metadata contains a known public typo artifact with an upper bound of 413, but the 99th percentile is 86 years.*

> [!WARNING]
> **Audit Note on Initial Demographic Claims (AUDIT-03):** Describing CheXpert patients as *"predominantly geriatric hospital admissions"* was partially unsupported (the median age is 62.0, below the standard geriatric threshold of 65). Furthermore, naive mean age comparisons were confounded by CheXpert strictly excluding pediatric patients ($<18$), whereas NIH (4.75%) and PadChest (5.85%) included pediatrics and neonates.

---

### 4. Radiological Views, Projections, and Hardware (Original)

| Attribute | NIH | CheXpert | PadChest |
| :--- | :--- | :--- | :--- |
| **View Orientation** | 100% Frontal (0% Lateral) | 84.06% Frontal<br>**15.94% Lateral** | 59.57% Documented Frontal<br>**15.33% Lateral / LL**<br>*(40.43% View NaN)* |
| **Projections** | PA: **62.29%**<br>AP: **37.71%** | AP: **69.53%**<br>PA: **14.53%**<br>*(15.94% Lateral)* | PA: **62.30%**<br>L: **28.43%**<br>AP_horizontal: **6.85%**<br>AP: **2.04%** |
| **Scanner Manufacturers** | *100% Unavailable (NaN)* | *100% Unavailable (NaN)* | **ImagingDynamicsCo: 52.18%**<br>**Philips Medical: 47.82%** |
| **Image Pixel Dimensions** | *100% Unavailable (NaN)* | *100% Unavailable (NaN)* | **100% Populated**<br>Mean width: 2,293.5 px<br>Range: [616, 4,280] px |
| **Pixel Spacing** | *Unavailable (NaN)* | *Unavailable (NaN)* | *Unavailable (NaN)* |

> [!WARNING]
> **Audit Note on View Missingness and AP Causality (AUDIT-02, AUDIT-04, AUDIT-05):**
> 1. PadChest was naively reported as having "40.43% View NaN". In reality, PadChest has a 100% complete radiologist-verified `Projection` column, while `ViewPosition_DICOM` was merely an optional DICOM tag.
> 2. Asserting that CheXpert AP predominance was strictly *"due to bedbound intensive care patients"* is an unverified clinical assumption, as CheXpert includes ambulatory and outpatient imaging.
> 3. Claiming scanner manufacturer tags were *"stripped for public release anonymization"* in NIH and CheXpert is an unverified assumption regarding institutional release policies.

---

### 5. Schema Completeness & Missing Metadata Matrix (Original)

| Standard Column | NIH Missing (%) | CheXpert Missing (%) | PadChest Missing (%) | Alignment Status |
| :--- | :---: | :---: | :---: | :--- |
| `image_id` | **0.0%** | **0.0%** | **0.0%** | Fully populated across all clients |
| `patient_id` | **0.0%** | **0.0%** | **0.0%** | Fully populated across all clients |
| `study_id` | **100.0%** | **0.0%** | **0.0%** | Available in CheXpert & PadChest |
| `source` | **0.0%** | **0.0%** | **0.0%** | Identified (`NIH`, `CheXpert`, `PadChest`) |
| `target` | **0.0%** | **0.0%** | **0.0%** | Binary ground truth (`1` vs `0`) |
| `age` | **0.0%** | **0.0%** | **<0.01%** | Available across all clients |
| `sex` | **0.0%** | **0.0%** | **<0.01%** | Available across all clients |
| `view_position` | **0.0%** | **0.0%** | **40.4%** | Frontal vs Lateral |
| `projection` | **0.0%** | **15.9%** | **0.0%** | PA / AP / Lateral designations |
| `image_width` | **100.0%** | **100.0%** | **0.0%** | Native DICOM dimensions in PadChest |
| `image_height` | **100.0%** | **100.0%** | **0.0%** | Native DICOM dimensions in PadChest |
| `pixel_spacing_x/y` | **100.0%** | **100.0%** | **100.0%** | Not provided in release metadata CSVs |
| `scanner_manufacturer`| **100.0%** | **100.0%** | **0.0%** | Native DICOM vendor in PadChest |
| `image_path` | **0.0%** | **0.0%** | **0.0%** | Standardized, relative forward-slash paths |
| `finding_labels` | **0.0%** | **100.0%** | **0.0%** | Raw diagnostic finding lists |
| `follow_up` | **0.0%** | **100.0%** | **100.0%** | Longitudinal encounter index in NIH |

---

### 6. Identified Dimensions of Heterogeneity (Original Baseline)

```
                                  Heterogeneity Landscape (Original)
                                                 │
       ┌──────────────────┬──────────────────────┼──────────────────────┬──────────────────┐
       ▼                  ▼                      ▼                      ▼                  ▼
 1. Label Skew       2. Covariate            3. Concept /           4. Client           5. Domain &
  (Prior Shift)         Shift               Feature Shift             Scale              Hardware
 P(Y): 71% vs 16%    Age: 61y vs 46y        AP vs PA views;        122K vs 60K          Philips vs
  CheXpert vs Pad    Sex: 59%M vs 55%F      Frontal vs Lateral     CheXpert vs Pad     IDC vs Unknown
```

1. **Label Distribution Skew (Prior Probability Shift, $P(Y)$):** CheXpert was reported as 70.8% positive vs. PadChest (16.4%) and NIH (18.1%).
2. **Covariate / Demographic Shift ($P(X_{\text{demo}})$):** Patient mean age differed by ~14.5 years between CheXpert (61.0y) and NIH (46.5y); biological sex inverted between US cohorts (56–59% Male) and Spain (55.1% Female).
3. **Concept & Viewpoint Shift ($P(X_{\text{image}} \mid Y)$):** CheXpert had 69.5% AP projections, compared to 62.3% PA in NIH and PadChest. NIH strictly contained frontal images, whereas CheXpert and PadChest contained lateral images.
4. **Data Volume & Participation Imbalance:** CheXpert contributed 47.6% of all images, NIH 28.8%, and PadChest 23.6%.
5. **Acquisition Domain & Scanner Heterogeneity:** Images originated across three health systems in two countries with disparate equipment.

---

# PART II: Authoritative Validated Audit & Methodological Corrections

> [!IMPORTANT]
> **Authoritative Baseline:** The findings in Part II are derived from a direct re-verification of the processed split files (`train.csv`, `val.csv`, `test.csv`) for NIH, CheXpert, and PadChest. All statistics below supersede naive initial metrics and serve as the validated ground truth for federated experiment design.

---

## 1. Audit Table of Flagged Claims, Root Causes, and Authoritative Corrections

The following table itemizes all flagged inconsistencies, misleading direct column comparisons, and unsupported causal claims from the initial analysis:

| Audit ID | Dimension | Initial Naive Claim / Metric | Root Cause & Empirical Flaw | Authoritative Correction |
| :--- | :--- | :--- | :--- | :--- |
| **AUDIT-01** | **Label Prior Skew Causality** | CheXpert's ~71% positive rate is *"driven by high-acuity inpatient/ICU sampling."* | **Filtering protocol artifact:** CheXpert raw data contains 90,203 unmentioned (`NaN`) and 11,628 uncertain (`-1.0`) cases. Filtering them out leaves 86,254 positive and 35,563 negative. If unmentioned cases were counted as negative, prevalence would be 38.6%. | **Empirical protocol effect:** CheXpert's 70.81% positive rate is an empirical label distribution skew driven jointly by clinical cohort factors and explicit positive-vs-negative filtering. It must not be cited as direct clinical incidence. |
| **AUDIT-02** | **View Position vs. Projection** | PadChest had *"40.4% missing view positions"*, while CheXpert was Frontal/Lateral and NIH was PA/AP. | **Semantic column conflation:** Anatomical orientation (*Frontal vs Lateral*) was conflated with beam trajectory (*PA vs AP*). PadChest's raw `ViewPosition_DICOM` was an optional PACS tag, but its `Projection` column is **100% complete and validated**. | **Harmonized reporting:**<br>• Orientation: NIH 100% Frontal; CheXpert 84.06% Frontal / 15.94% Lateral; PadChest 71.57% Frontal / 28.43% Lateral.<br>• Frontal Trajectory: NIH 62.29% PA / 37.71% AP; CheXpert 17.28% PA / 82.72% AP; PadChest 87.51% PA / 12.49% AP (excl. costal). |
| **AUDIT-03** | **Age Demographics & Outliers** | Raw NIH age ranged from 1 to 413; CheXpert was described as *"predominantly geriatric."* | **Typo artifacts & pediatric mismatch:** NIH has 9 public typo entries ($>100$, up to 413). CheXpert excludes pediatrics ($<18$) and caps at 90.0 (HIPAA). NIH has 4.75% pediatrics; PadChest has 5.85% pediatrics down to neonates (age 0). CheXpert median adult age is 62.0 (below geriatric threshold of 65). | **Adult cohort ($18 \le \text{Age} \le 100$):**<br>• NIH Adults: $48.18 \pm 14.98$ yrs (Median 49.0)<br>• CheXpert Adults: $61.03 \pm 17.52$ yrs (Median 62.0)<br>• PadChest Adults: $53.86 \pm 17.50$ yrs (Median 53.0)<br>CheXpert adults are genuinely 12.85 yrs older than NIH and 7.17 yrs older than PadChest. |
| **AUDIT-04** | **AP Projection Causality** | CheXpert AP predominance was caused strictly by *"bedbound intensive care patients."* | **Unmeasured clinical status:** While AP is standard portable technique when patients cannot stand, CheXpert includes outpatient, emergency, and inpatient examinations (Irvin et al., 2019). Bedbound status is unmeasured in metadata. | **Empirical acquisition profile:** CheXpert comprises a high proportion of portable/AP examinations (82.72% of frontal views). This creates an authentic imaging physics/concept shift against upright PA cohorts. |
| **AUDIT-05** | **Hardware Metadata Omission** | Scanner metadata was *"omitted during public release anonymization"* in NIH and CheXpert. | **Unverified motivation:** Public CSV tables for NIH and CheXpert simply do not include manufacturer or matrix columns. Asserting specific institutional intent is an unverified assumption. | **Fact-based reporting:** Manufacturer and pixel dimensions are absent in NIH and CheXpert CSV releases, but 100% documented in PadChest (52.18% IDC, 47.82% Philips). |
| **AUDIT-06** | **Intra-Client Prior Inversion** | Client test sets evaluate performance under client training priors. | **Internal train-to-test prior shifts:** CheXpert Train is **70.93% positive**, but its Test benchmark is **28.63% positive**. NIH Train is **14.71% positive**, but its Test set is **32.08% positive** (enriched). PadChest Train is **16.41% positive** and Test is **16.10% positive**. | **Algorithmic consequence:** Local models trained on CheXpert will suffer severe calibration drift when evaluated on their own test set. Threshold-free metrics (**AUROC**, **PR-AUC**) are required. |
| **AUDIT-07** | **Evaluation Sample Disparity** | Local client test sets are equally reliable indicators of model generalization. | **Sample size asymmetry:** CheXpert contributes 51.9% of all training data (109,572 images), but its gold-standard test set contains only **234 images** (67 positive, 167 negative), compared to 14,519 in NIH and 5,926 in PadChest. | **Statistical power impact:** CheXpert test AUROC has a 95% confidence interval width of $\sim \pm 6.5\%$, vs. $\sim \pm 0.8\%$ on NIH. Client test set scores cannot be averaged naively without variance weighting. |

---

## 2. Authoritative Client Profiles & Sizing

The federation contains **255,738 clean chest radiographs across 100,344 unique patients**. Whole-patient separation across splits is strictly enforced with zero patient or study leakage:

| Client ID | Institution | Country | Total Clean Images | Unique Patients | Unique Studies | Images / Patient | Federation Share (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`client_01_NIH`** | NIH Clinical Center | Bethesda, MD, USA | **73,678** | **25,923** | *Unrecorded* | 2.84 | 28.81% |
| **`client_02_CheXpert`** | Stanford Hospital | Stanford, CA, USA | **121,817** | **41,009** | **100,728** | 2.97 | 47.63% |
| **`client_03_PadChest`** | Hosp. Univ. San Juan | Alicante, Spain | **60,243** | **33,412** | **42,567** | 1.80 | 23.56% |
| **Total Federation** | **3 Health Systems** | **2 Nations** | **255,738** | **100,344** | **143,295+** | **2.55** | **100.0%** |

### Validated Partition Split Distribution

| Client | Split | Images | Patients | Studies | Positives (`target=1`) | Negatives (`target=0`) | Positive Rate (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **NIH** | **Train** | 53,323 | 20,977 | *Unrecorded* | 7,842 | 45,481 | 14.71% |
| | **Validation** | 5,836 | 2,331 | *Unrecorded* | 817 | 5,019 | 14.00% |
| | **Test** | 14,519 | 2,615 | *Unrecorded* | 4,658 | 9,861 | 32.08% |
| | *Client Total* | *73,678* | *25,923* | *Unrecorded* | *13,317* | *60,361* | *18.07%* |
| **CheXpert** | **Train** | 109,572 | 36,728 | 90,675 | 77,719 | 31,853 | 70.93% |
| | **Validation** | 12,011 | 4,081 | 9,853 | 8,468 | 3,543 | 70.50% |
| | **Test** | 234 | 200 | 200 | 67 | 167 | 28.63% |
| | *Client Total* | *121,817* | *41,009* | *100,728* | *86,254* | *35,563* | *70.81%* |
| **PadChest** | **Train** | 48,340 | 26,729 | 34,092 | 7,934 | 40,406 | 16.41% |
| | **Validation** | 5,977 | 3,341 | 4,246 | 965 | 5,012 | 16.15% |
| | **Test** | 5,926 | 3,342 | 4,229 | 954 | 4,972 | 16.10% |
| | *Client Total* | *60,243* | *33,412* | *42,567* | *9,853* | *50,390* | *16.36%* |
| **Federation** | **All Splits** | **255,738** | **100,344** | **143,295+** | **109,424** | **146,314** | **42.79%** |

---

## 3. Authoritative Demographics & Radiological Acquisition

### Demographic Distributions
* **Adult Cohort ($18 \le \text{Age} \le 100$):**
  * **NIH (70,167 images):** Mean $48.18 \pm 14.98$ years | Median 49.0 | IQR [36.0, 59.0]
  * **CheXpert (121,817 images):** Mean $61.03 \pm 17.52$ years | Median 62.0 | IQR [50.0, 74.0]
  * **PadChest (56,714 images):** Mean $53.86 \pm 17.50$ years | Median 53.0 | IQR [41.0, 67.0]
* **Pediatric Proportions ($<18$ years):**
  * NIH: **3,502 images (4.75%)**
  * CheXpert: **0 images (0.00%)** (adult-only cohort, capped at 90.0)
  * PadChest: **3,522 images (5.85%)** (neonates and pediatrics, min age 0.0)
* **Biological Sex Composition:**
  * NIH: **56.13% Male** (41,357) | **43.87% Female** (32,321)
  * CheXpert: **59.11% Male** (72,002) | **40.89% Female** (49,814) | <0.01% Unknown (1)
  * PadChest: **44.91% Male** (27,053) | **55.08% Female** (33,180) | 0.01% Other (6) | 4 Nulls

### Harmonized Radiological Views & Beam Projections
* **Anatomical Orientation (Frontal vs Lateral):**
  * NIH: **100.0% Frontal** (73,678) | **0.0% Lateral** (0)
  * CheXpert: **84.06% Frontal** (102,400) | **15.94% Lateral** (19,417)
  * PadChest: **71.57% Frontal** (43,113) | **28.43% Lateral** (17,128)
* **Frontal Beam Trajectory (PA vs AP among Frontal Images):**
  * NIH: **62.29% PA** (45,891) vs. **37.71% AP** (27,787)
  * CheXpert: **17.28% PA** (17,694) vs. **82.72% AP** (84,696)
  * PadChest: **87.51% PA** (37,530) vs. **12.49% AP / AP_horizontal** (5,358) *(excluding 225 costal views)*

### Scanner Hardware & Pixel Metrics
* **NIH & CheXpert:** Scanner manufacturer and native matrix dimensions are unrecorded in official public CSV tables.
* **PadChest:** 100% recorded across two digital vendors:
  * `ImagingDynamicsCompanyLtd`: **52.18%** (31,432 images)
  * `PhilipsMedicalSystems`: **47.82%** (28,811 images)
  * Native matrix resolution: Mean width $2,293.5$ px (range $[616, 4,280]$); Mean height $2,382.4$ px (range $[512, 4,280]$).

---

## 4. Authoritative Schema Completeness Matrix

The completeness percentage across the 17 standardized columns is summarized below:

| Column Name | NIH Populated | CheXpert Populated | PadChest Populated | Description & Standard Representation |
| :--- | :---: | :---: | :---: | :--- |
| `image_id` | **100.0%** | **100.0%** | **100.0%** | Unique image file identifier string |
| `patient_id` | **100.0%** | **100.0%** | **100.0%** | Patient identifier string |
| `study_id` | 0.0% (NaN) | **100.0%** | **100.0%** | Clinical study identifier (`patient_study`) |
| `source` | **100.0%** | **100.0%** | **100.0%** | Client origin tag: `NIH`, `CheXpert`, `PadChest` |
| `target` | **100.0%** | **100.0%** | **100.0%** | Binary target: `1` (Positive), `0` (Negative) |
| `age` | **100.0%** | **100.0%** | **99.99%** | Patient age in years |
| `sex` | **100.0%** | **100.0%** | **99.99%** | Patient biological sex (`M`, `F`, `O`) |
| `view_position` | **100.0%** | **100.0%** | 59.57% | Frontal vs Lateral orientation string |
| `projection` | **100.0%** | 84.06% | **100.0%** | PA, AP, Lateral beam projections |
| `image_width` | 0.0% (NaN) | 0.0% (NaN) | **100.0%** | Native DICOM image width in pixels |
| `image_height` | 0.0% (NaN) | 0.0% (NaN) | **100.0%** | Native DICOM image height in pixels |
| `pixel_spacing_x` | 0.0% (NaN) | 0.0% (NaN) | 0.0% (NaN) | Horizontal pixel spacing (mm) |
| `pixel_spacing_y` | 0.0% (NaN) | 0.0% (NaN) | 0.0% (NaN) | Vertical pixel spacing (mm) |
| `scanner_manufacturer`| 0.0% (NaN) | 0.0% (NaN) | **100.0%** | Digital radiography equipment vendor |
| `image_path` | **100.0%** | **100.0%** | **100.0%** | Standardized forward-slash relative image path |
| `finding_labels` | **100.0%** | 0.0% (NaN) | **100.0%** | Native diagnostic finding string/list |
| `follow_up` | **100.0%** | 0.0% (NaN) | 0.0% (NaN) | Longitudinal encounter follow-up index |

---

## 5. Revised Heterogeneity Taxonomy for Federated Learning

```
                               Federated Heterogeneity Architecture
                                                │
         ┌───────────────────┬──────────────────┴───────────────────┬───────────────────┐
         ▼                   ▼                                      ▼                   ▼
   1. Label Skew       2. Feature & Concept                   3. Demographic       4. Client Scale &
   Prior Shift P(Y)       Shift P(X | Y)                      Covariate Shift       Evaluation
  Train P(Y=1):       Frontal PA: NIH (62%), Pad (88%)       Adult Age:           Training Share:
  CheXpert: 70.9%     Frontal AP: CheXpert (83%)             CheXpert: 61.0 yrs   CheXpert: 51.9%
  PadChest: 16.4%     Lateral Views:                         NIH:      48.2 yrs   NIH:      25.3%
  NIH:      14.7%     NIH (0%) vs CheX (16%) vs Pad (28%)    PadChest: 53.9 yrs   PadChest: 22.9%
```

1. **Extreme Label Distribution Skew (Prior Shift, $P(Y)$):**
   * CheXpert training data has an empirical prevalence of **70.93% positive**, whereas NIH (**14.71%**) and PadChest (**16.41%**) are negative-dominated.
   * Under standard Federated Averaging (FedAvg), CheXpert's local gradient updates pull the global classifier bias term strongly positive, causing severe false-positive rates when evaluated on NIH and PadChest.
2. **Concept & Imaging Acquisition Shift ($P(X_{\text{image}} \mid Y)$):**
   * In upright PA radiographs (dominant in NIH and PadChest), pleural effusion gravitates into dependent costophrenic angles, creating a sharp concave meniscus sign.
   * In portable AP radiographs (dominant in CheXpert), fluid layers across the posterior thoracic space, producing diffuse hemithorax haziness.
   * Convolutional filters trained on one projection family face an authentic visual representation shift when evaluated on another.
3. **Viewpoint Heterogeneity:**
   * NIH contains 0% lateral views, while CheXpert contains 15.94% and PadChest contains 28.43% lateral views. Multi-view or lateral-dependent architectures cannot run on NIH without synthetic imputation or view masking.
4. **Demographic Covariate Shift ($P(X_{\text{demo}})$):**
   * CheXpert adult patients are on average **~12.9 years older** than NIH patients and **~7.2 years older** than PadChest patients. Anatomical characteristics (cardiothoracic ratio, aortic tortuosity, parenchymal elasticity) correlate with age.
   * Biological sex inverts from male-dominant in the US cohorts (~56–59% M) to female-dominant in Spain (55.1% F).
5. **System Scale & Evaluation Asymmetry:**
   * In sample-weighted FedAvg ($w_k = n_k / N$), CheXpert receives **51.9% of the aggregation weight**, allowing Stanford's distribution to dominate global gradient trajectories.
   * CheXpert's official gold-standard test benchmark contains only **234 images**, creating an evaluation asymmetry compared to NIH (14,519 images) and PadChest (5,926 images).

---

## 6. Strategic Recommendations for FL Experiment Design

1. **Input View Standardization (Frontal Benchmark):**
   * **Recommendation:** For the primary cross-institutional federated benchmark, restrict training and evaluation to **Frontal radiographs only** (`projection.isin(['PA', 'AP', 'AP_horizontal'])`).
   * This yields a 100% interoperable domain across all three nodes (NIH: 53,323 train; CheXpert: 92,173 train; PadChest: 34,367 train), while eliminating lateral view out-of-distribution failure on NIH.
2. **Client Loss Balancing:**
   * Local models should employ client-specific **Focal Loss** or **effective class weighting** ($w_c = (1 - \beta) / (1 - \beta^{n_c})$) to prevent local gradient bias collapse under the 71% vs. 15% prior disparity.
3. **Adaptive Federated Aggregation:**
   * Naive FedAvg will overfit to CheXpert due to its 51.9% sample volume and high positive bias.
   * The project's **Adaptive Federated Aggregation Algorithm** should dynamically down-weight divergent gradient directions (e.g. angle-based gradient projection or normalized client weighting) to ensure fair, robust generalization across all clinical centers.
4. **Threshold-Free Evaluation & Cross-Client Generalization Matrix:**
   * Primary discrimination metrics must be threshold-independent: **AUROC** and **macro PR-AUC**, preventing distortions caused by intra-client prior shifts.
   * Report a full $3 \times 3$ **Cross-Client Generalization Matrix** (training on Client $i$, evaluating on Client $j$) alongside the global federated model to directly quantify out-of-domain transferability.
