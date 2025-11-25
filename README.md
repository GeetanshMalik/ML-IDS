# ML-IDS: Machine Learning Intrusion Detection System

A Machine Learning-based Intrusion Detection System using the AWID CLS-R WiFi security dataset and XGBoost classifier.

## Features

- Supervised ML intrusion detection using XGBoost
- Preprocessing and data balancing pipeline
- Classification metrics and confusion matrix visualization
- Real-time intrusion detection demo
- Tunable detection threshold

## Project Structure

```
ML-IDS/
├── data/
│   ├── raw/                    # Place AWID dataset here
│   └── processed/              # Generated .npz files
├── models/
│   └── xgb_awid.pkl           # Trained model (27MB)
├── src/
│   ├── preprocess/
│   │   ├── awid_preprocess.py
│   │   └── awid_balance.py
│   ├── supervised/
│   │   ├── train_xgb.py
│   │   └── evaluate_xgb.py
│   └── utils/
│       └── check_distribution.py
├── realtime_demo.py
├── requirements.txt
└── README.md
```

## Setup

1. **Install dependencies:**
```bash
pip install numpy pandas scikit-learn xgboost matplotlib

```

2. **Download dataset:**
   - Get AWID CLS-R from [Kaggle](https://www.kaggle.com/datasets/zhiqingcui/awidclsr)
   - Place files in `data/raw/`:
     - `AWID-CLS-R-Trn.csv`
     - `AWID-CLS-R-Tst.csv`

3. **Preprocess data:**
```bash
cd src/preprocess
python awid_preprocess.py
python awid_balance.py
```

4. **Train model:**
```bash
cd src/supervised
python train_xgb.py
```

5. **Evaluate performance:**
```bash
python evaluate_xgb.py
```

6. **Run demo:**
```bash
python realtime_demo.py
```

## Results

The model achieves strong performance with optimized threshold (0.001):
- High accuracy and recall
- Low false negative rate
- Effective detection of WiFi attacks (Deauth, Evil Twin, Rogue AP, Flooding)

## Requirements

- Python 3.10+
- xgboost
- numpy
- pandas
- scikit-learn
- matplotlib

## Author

Geetansh Malik  
GitHub: [GeetanshMalik](https://github.com/GeetanshMalik)
