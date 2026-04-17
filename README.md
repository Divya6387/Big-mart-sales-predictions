 BigMart Sales Prediction

Analytics Vidhya Hackathon – Predict item-level sales across BigMart outlets.


Problem Statement
The data scientists at BigMart have collected 2013 sales data for 1,559 products across 10 stores in different cities. The goal is to build a predictive model that estimates the sales of each product at a particular outlet — helping BigMart understand what properties of products and stores drive sales.

Train set: 8,523 rows (with Item_Outlet_Sales as target)
Test set: 5,681 rows (predict Item_Outlet_Sales)
Evaluation metric: Root Mean Squared Error (RMSE)

Data Dictionary
VariableDescriptionItem_IdentifierUnique product IDItem_WeightWeight of productItem_Fat_ContentLow Fat / RegularItem_Visibility% of total display area in storeItem_TypeCategory of productItem_MRPMaximum Retail PriceOutlet_IdentifierUnique store IDOutlet_Establishment_YearYear store was establishedOutlet_SizeStore size (Small / Medium / High)Outlet_Location_TypeTier 1 / 2 / 3 cityOutlet_TypeGrocery Store or Supermarket typeItem_Outlet_SalesTARGET – Sales value to predict

Setup & Usage
1. Clone the repository
bashgit clone https://github.com/<your-username>/bigmart-sales-prediction.git
cd bigmart-sales-prediction
2. Install dependencies
bashpip install -r requirements.txt
3. Add data files
Place train_v9rqX0R.csv and test_AbJTz2l.csv inside the data/ folder.
Update TRAIN_PATH and TEST_PATH at the top of solution.py if needed.
4. Run the pipeline
bashpython solution.py
This will:

Clean and impute missing values
Engineer new features
Train XGBoost, LightGBM, and Ridge models with 5-Fold CV
Save the ensemble submission to outputs/submission.csv


Data Cleaning & Imputation
IssueFixItem_Weight missing (~17%)Filled with per-item mean across all rowsOutlet_Size missing (~29%)Filled with mode per Outlet_TypeItem_Visibility = 0 (impossible)Replaced with per-item mean visibilityItem_Fat_Content inconsistent labelsStandardised: LF / low fat → Low Fat, reg → RegularNon-edible items with fat labelsRe-labeled as Non-Edible

Feature Engineering
New FeatureDescriptionOutlet_Age2013 − Outlet_Establishment_YearItem_CategoryFirst 2 chars of Item_Identifier (FD = Food, NC = Non-consumable, DR = Drink)Visibility_RatioItem visibility ÷ item's mean visibilityMRP_BucketCut Item_MRP into 4 bins: Low / Medium / High / Very HighIs_GroceryBinary flag: 1 if Grocery Store, else 0

Models & Results
ModelCV RMSE (log-space)XGBoost~0.561LightGBM~0.545Ridge~0.543Ensemble (45/45/10)Best

The target Item_Outlet_Sales is log1p-transformed during training and expm1-transformed back for submission (reduces skew, helps tree models).

Ensemble Strategy
Predictions are weighted-averaged:
45% XGBoost
45% LightGBM
10% Ridge


Key Insights ->
Item_MRP is by far the strongest predictor of sales.
Outlet_Type (Grocery vs Supermarket) drastically affects sales volume.
Outlet_Age adds signal — older, established stores tend to have steadier sales.
Fixing zero-visibility values meaningfully improves model performance.
Standardising fat content labels prevents the model from treating duplicates as separate categories.


 Requirements ->
See requirements.txt. Main packages:
pandas, numpy
scikit-learn
xgboost
lightgbm
matplotlib, seaborn


