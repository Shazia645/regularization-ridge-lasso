# Regularization: Ridge vs Lasso

project on the **50 Startups** dataset to understand how L1 (Lasso) and L2 (Ridge) regularization work and how they affect model coefficients.

## Dataset
`50_Startups.csv`: R&D Spend, Administration, Marketing Spend, State, Profit.
Target: **Profit**. The categorical `State` column is one-hot encoded.

## What I did
- Split data into train/test sets and scaled the features
- Trained Ridge and Lasso regression
- Compared Lasso coefficients to see feature importance
- Used `RidgeCV` and `LassoCV` (5-fold CV) to find the best alpha

## Results

| Model | Best params | Test R2 |
|-------|-------------|---------|
| Ridge | alpha=0.1 | 0.8985 |
| Lasso | alpha=1000 | 0.9166 |
| ElasticNet | alpha=1000, l1_ratio=1.0 | 0.9166 |

## Key Learnings
- R&D Spend is the strongest predictor of profit
- Lasso shrank the `State_New York` coefficient to exactly 0, which removes the feature (feature selection)
- Ridge shrinks coefficients but never makes them exactly zero
- Both models perform similarly on this small dataset
- ElasticNetCV selected l1_ratio = 1.0, meaning pure Lasso performed best on this dataset
- With a larger alpha (1000), Lasso set both State dummy variables to zero, showing that State has little effect on Profit
- R&D Spend is the dominant predictor of Profit
- Lasso/ElasticNet (R2 0.9166) outperformed Ridge (R2 0.8985)

## Tech Stack
Python, Pandas, NumPy, Scikit-learn, Jupyter Notebook

## How to Run
1. Clone the repo
2. Install requirements: `pip install pandas numpy scikit-learn jupyter`
3. Open `regularization.ipynb` and run all cells