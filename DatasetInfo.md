## 📊 Dataset Description

The dataset used for **Phase 1: Solar PV Efficiency Prediction** consists of two files:

### 🔹 `train.csv`
Contains historical solar plant data used for training the machine learning model.  
Each row represents a time-stamped measurement of plant/system performance along with environmental conditions.

**Key columns (indicative):**
- `id` — Unique record identifier
- `efficiency` — Target variable: measured solar PV efficiency
- `temperature` — Temperature at the panel/inverter location
- `irradiance` — Solar irradiance level
- `humidity` — Ambient humidity
- `string_id` — Categorical string identifier for sub-system (encoded)
- `error_code` — Categorical error/event code from system sensors
- `installation_type` — Categorical type of installation
- *(Other engineered/system metrics)*

### 🔹 `test.csv`
Contains similar features as `train.csv` but **without the target `efficiency` column**.  
This data is used for generating predictions.

---

### ⚠ Note
> The dataset was provided by **Zelestra x AWS ML Ascend Challenge** organizers for competition use only.  
> We **do not redistribute the dataset** here — it must be obtained from the official source.

---

## 📂 Phase1 Folder Contents

```text
Phase1/
├── train.csv           # Solar plant training data (local copy for dev only)
├── test.csv            # Test data for predictions (local copy for dev only)
├── phase1_model.ipynb  # Notebook with full ML pipeline (feature engg, modeling, stacking)
├── submission.csv      # Example output predictions file
├── DATASET_INFO.md     # This dataset description and folder overview

---

## 📝 Usage

Our code reads the dataset as:
```python
train = pd.read_csv("Phase1/train.csv")
test = pd.read_csv("Phase1/test.csv")

---
