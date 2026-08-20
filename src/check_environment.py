import sys

import numpy as np
import pandas as pd
import scipy
import sklearn
import xgboost
import shap
import imblearn
import fastapi

print("=" * 55)
print("FEDFAIRCHAIN ENVIRONMENT CHECK")
print("=" * 55)

print("Python          :", sys.version.split()[0])
print("NumPy           :", np.__version__)
print("Pandas          :", pd.__version__)
print("SciPy           :", scipy.__version__)
print("Scikit-learn    :", sklearn.__version__)
print("XGBoost         :", xgboost.__version__)
print("SHAP            :", shap.__version__)
print("Imbalanced-learn:", imblearn.__version__)
print("FastAPI         :", fastapi.__version__)

print("\nALL CORE LIBRARIES IMPORTED SUCCESSFULLY.")