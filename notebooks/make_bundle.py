import joblib
import pandas as pd
import pm_core as pm

bundle = pm.train_bundle(pd.read_csv(pm.find_dataset()))
joblib.dump(bundle, "bundle.joblib", compress=3)
print("saved")