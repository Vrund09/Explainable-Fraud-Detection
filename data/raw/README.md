# Raw Data Directory

## PaySim Dataset

This directory should contain the PaySim dataset file:

- **paysim.csv** - The main dataset containing synthetic financial transaction data

### Dataset Information

The PaySim dataset is a synthetic dataset based on a sample of real transactions extracted from one month of financial logs from a mobile money service implemented in an African country.

**Download Instructions:**
1. Download the PaySim dataset from Kaggle: [PaySim Synthetic Financial Fraud Detection Dataset](https://www.kaggle.com/ntnu-testimon/paysim1)
2. Place the `paysim.csv` file in this directory
3. The file should contain columns: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

**File Structure:**
```
data/raw/
├── README.md (this file)
└── paysim.csv (to be added by user)
```

