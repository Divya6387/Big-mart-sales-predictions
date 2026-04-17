# BigMart Sales Prediction

Predict item-level sales across BigMart outlets using machine learning.

Analytics Vidhya Hackathon — Big Mart Sales III

---

## Problem Statement

The data scientists at BigMart collected 2013 sales data for 1,559 products across 10 stores in different cities. The goal is to build a predictive model that estimates the sales of each product at a particular outlet.

- **Train set:** 8,523 rows (includes target column `Item_Outlet_Sales`)
- **Test set:** 5,681 rows (predict `Item_Outlet_Sales`)
- **Evaluation metric:** Root Mean Squared Error (RMSE)


## Data Dictionary

| Variable | Description |

| Item_Identifier | Unique product ID |
| Item_Weight | Weight of product |
| Item_Fat_Content | Low Fat or Regular |
| Item_Visibility | Percentage of total display area in store |
| Item_Type | Category of product |
| Item_MRP | Maximum Retail Price |
| Outlet_Identifier | Unique store ID |
| Outlet_Establishment_Year | Year store was established |
| Outlet_Size | Store size — Small, Medium, High |
| Outlet_Location_Type | Tier 1, Tier 2, or Tier 3 city |
| Outlet_Type | Grocery Store or Supermarket type |
| Item_Outlet_Sales | TARGET — Sales value to predict |

## Setup and Usage

### Step 1 — Clone the repository

bash
git clone https://github.com/your-username/bigmart-sales-prediction.git
cd bigmart-sales-prediction


### Step 2 — Install dependencies

bash
pip install -r requirements.txt


### Step 3 — Add data files

Place `train_v9rqX0R.csv` and `test_AbJTz2l.csv` inside the `data/` folder.

Update `TRAIN_PATH` and `TEST_PATH` at the top of `solution.py` if needed.

### Step 4 — Run the pipeline


python solution.py


This will automatically:

1. Load and explore the data
2. Clean and impute missing values
3. Engineer new features
4. Train XGBoost, LightGBM, and Ridge models with 5-Fold CV
5. Save the ensemble predictions to `outputs/submission.csv`


## Data Cleaning

| Problem | Fix Applied |

| Item_Weight missing 17% | Filled with per-item mean across all rows |
| Outlet_Size missing 29% | Filled with mode per Outlet_Type |
| Item_Visibility equals zero | Replaced with per-item mean since zero is physically impossible |
| Item_Fat_Content inconsistent labels | Standardised LF and low fat to Low Fat, reg to Regular |
| Non-edible items had fat content labels | Re-labeled as Non-Edible |

## Feature Engineering

| New Feature | Description |

| Outlet_Age | 2013 minus Outlet_Establishment_Year |
| Item_Category | First 2 characters of Item_Identifier — FD for Food, DR for Drink, NC for Non-consumable |
| Visibility_Ratio | Item visibility divided by that items mean visibility |
| MRP_Bucket | Item_MRP cut into 4 bins — Low, Medium, High, Very High |
| Is_Grocery | Binary flag — 1 if Grocery Store, 0 otherwise |

## Models and Results

| Model | CV RMSE log-space |
| XGBoost | 0.5615 |
| LightGBM | 0.5454 |
| Ridge | 0.5430 |
| Ensemble 45 / 45 / 10 | Best overall |

### Why log-transform the target?

'Item_Outlet_Sales' is right-skewed. Applying 'log1p' during training and 'expm1' at prediction time reduces the impact of extreme values and helps all three models perform better.

### Ensemble Strategy

Final predictions are a weighted average of all three models.

- 45% XGBoost
- 45% LightGBM
- 10% Ridge


## Key Insights

- `Item_MRP` is the strongest predictor of sales by a large margin
- `Outlet_Type` has the biggest categorical impact on sales volume
- `Outlet_Age` adds useful signal — older stores have more stable, predictable sales
- Fixing zero-visibility values meaningfully improves model accuracy
- Standardising fat content labels prevents the model from treating duplicate categories as separate signals

## Requirements
pandas
numpy
scikit-learn
xgboost
lightgbm
matplotlib
seaborn



