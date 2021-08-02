# Prepare Dataset
You can Directly use the data csv in [ISOLET](./ISOLET)

or 
1. Download [here](https://datahub.io/machine-learning/isolet#data) 
2. Store all csv in [data_csv](./ISOLET/data_csv)
3. Sort the data if you want to use IID
  ```bash
  cd "data preprocess src"
  python sort_data.py
  ```
# Usage
```bash
cd FLHDC_binaryAM_new
```
Then, follow the instructions

# Experiments
Dir | Purpose |
|-------------------|-------------------|
FLHDC | one-shot Federated HDC |
FLHDC_binaryAM_new | Federated HDC with retrained updates and BinaryAM (Final Version) |
Centralized HDC | One-shot and Retrained HDC trained by centralized dataset. |
data preprocess src | Download datasets and preprocess them into csv files |
ISOLET | Storage for preprocessed data csv  |
