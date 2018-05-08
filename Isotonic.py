#coding: utf-8

import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression

import gc


dtypes = {
    'click_id':'uint32',
    'predicted':'float32',
    'is_attributed':'float32',
}
target_file = "../train_v22_click_id.csv"
pred_file = "../newdata/Output/cv_pred.Feat@stack8_Learner@reg_lgb_native.csv"
test_file = "../newdata/Output/Subm/sub_pred.Feat@stack8_Learner@reg_lgb_native.[Mean0.018597]_[Std0.003297].csv"
print("Loading training data")
df_X = pd.read_csv(pred_file,dtype=dtypes)
X = df_X["predicted"].values
del(df_X)
print("len_X={}".format(len(X)))
gc.collect()

df_y = pd.read_csv(target_file,skiprows=range(1,24903889),dtype=dtypes)
y = df_y["click_id"].values
del(df_y)
y = y[-40000000:]
print("len_y={}".format(len(y)))
gc.collect()

print("Loading test data")
test_df = pd.read_csv(test_file,dtype=dtypes)
click_id = test_df["click_id"].values
X_test = test_df["is_attributed"].values
del(test_df)
print("len_X_test={}".format(len(X_test)))
gc.collect()

print("Fitting data")
ir = IsotonicRegression()
ir.fit_transform(X, y)

print("Predict data")
y_test = ir.predict(X_test)
print("Save prediction")

import config
fname = "%s/sub_pred.%s.[Mean%.6f]_[Std%.6f].csv"%(
    config.SUBM_DIR, "isotonic_stack8", 0.0, 0.0)
print("Writing to file:{}".format(fname))
pd.DataFrame({"click_id": click_id, "is_attributed": y_test}).fillna(0).to_csv(fname, index=False)

