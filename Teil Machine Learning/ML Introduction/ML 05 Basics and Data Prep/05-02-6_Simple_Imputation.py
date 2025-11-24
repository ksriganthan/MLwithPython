"""

05-02-6_Simple_Imputation.py
Simple imputation of missing values.

"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

myDict = {
        "A": [1.0, 2.0, np.nan, 4.0, 5.0],
        "B": [10.0, np.nan, 30.0, 40.0, 50.0],
        "C": [np.nan, 0.5, 0.7, np.nan, 1.0]
         }
#np.nan = kein Wert

type(myDict)
df = pd.DataFrame(myDict)
print(df)
print(df.iloc[:,0:])
# Simple Imputer from sklearn is a univariate imputation method.
# That means that each feature is imputed independently of the others.
# For example, if a value is missing in column A, the imputer will only use the values in column A to impute the missing value.

# Initialize the SimpleImputer with the desired strategy
imputer = SimpleImputer(strategy="mean") # Man könnte hier auch den Median oder bei kategorischen den Modus verwenden
# Fit the imputer on the DataFrame and transform the data (fit & transform)
imputed_array = imputer.fit_transform(df) # Gibt ein Array zurück !!!
type(imputed_array)

# Convert the resulting array back to a DataFrame
imputed_df = pd.DataFrame(imputed_array, columns=df.columns)
print(imputed_df)