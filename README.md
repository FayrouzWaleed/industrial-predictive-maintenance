# Industrial Predictive Maintenance

A machine learning project that predicts **whether a machine will fail and which type of failure it will be**, using operational sensor readings. The main goal is not accuracy, but keeping **false alarms low**, because every false alarm stops production unnecessarily.

**Live dashboard:** https://industrial-predictive-maintenance-98k9yrenks9nyi5qyubt2e.streamlit.app/

## Project Goals

* Predict the failure type (Tool Wear, Heat Dissipation, Power Failure, Overstrain) or "No Failure" from sensor readings.
* Handle the strong class imbalance (about 3.3% of machines fail).
* Evaluate models with the **False Discovery Rate (FDR)**, the share of alarms that are wrong, and tune the decision threshold to minimize it.
* Compare the FDR-based threshold with a **cost-based** threshold.
* Compare supervised models with an **anomaly detector**.
* Find which sensor is the lead indicator of each failure type.
* Provide an **interactive dashboard** where a user can explore the results and enter sensor readings to get a prediction.

## Dataset

AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository), 10,000 machine observations.

Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Citation: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020. Licensed under CC BY 4.0.

The dataset is synthetic, and its failures follow fixed rules based on the same sensors used as features. The results therefore show how the methods behave, not how they would perform in a real factory.

## Approach

1. **Exploratory data analysis:** sensor distributions, class imbalance, failure rate by product type.
2. **Target construction:** the original `Machine failure` column is binary. A 5-class `Failure Type` target was built from the failure mode columns (TWF, HDF, PWF, OSF).
   * 9 rows with a failure but no recorded failure mode were removed.
   * 19 random failure (RNF) rows were removed, because they occur by chance and cannot be predicted from sensors.
   * A row with several failure modes at once is assigned to one type (the first in the order TWF, HDF, PWF, OSF).
3. **Feature engineering:** features based on the physics of each failure mode:
   * Temperature difference (process minus air temperature), for Heat Dissipation.
   * Power (torque x rotational speed), for Power Failure.
   * Tool wear x torque, for Overstrain.
4. **Models:** Dummy baseline, Logistic Regression, Random Forest and XGBoost, with class weights to handle the imbalance.
5. **Threshold tuning:** the model predicts "No Failure" unless its most likely failure type reaches a probability threshold. The threshold is chosen on a validation set (lowest FDR with at least 70% recall) and evaluated once on the untouched test set.
6. **Cost-sensitive learning:** each type of mistake gets a cost, and the threshold with the lowest total cost is selected, followed by a sensitivity analysis over different cost ratios.
7. **Anomaly detection:** an Isolation Forest trained only on normal machines.
8. **Correlation analysis:** correlation of each sensor with each failure type, to name the lead sensor.
9. **Dashboard:** the same pipeline (cleaning, features, seeds and split) wrapped in a Streamlit app.

## Key Results

Test set: 1,995 machines, of which 66 are real failures.

| Model | Threshold | False Alarms | Missed Failures | FDR | Recall |
|---|---|---|---|---|---|
| Random Forest (selected) | 0.6 | 3 | 21 | 6.3% | 68.2% |
| XGBoost | 0.9 | 10 | 12 | 15.6% | 81.8% |

* **Random Forest is the final model**, because it produces the fewest false alarms, which is the main requirement. XGBoost detects more failures, so it is the better choice if missed failures are more costly than false alarms for the factory.
* **Cost-sensitive view:** with assumed costs (false alarm = 1, wrong type = 2, missed failure = 10), Random Forest at a lower threshold (0.2) has the lowest total cost, and it beats XGBoost for every cost ratio from 1 to 20. The cost values are assumptions and should be replaced with real ones.
* **Anomaly detection:** the Isolation Forest is much weaker than the supervised models (ROC-AUC 0.79 compared with 0.99) and has an FDR of about 92%, so it cannot meet the FDR requirement. It could only serve as a second opinion that needs no failure labels.
* **Lead sensor:** Torque has the strongest average correlation with failures, slightly ahead of Rotational speed. All raw correlations are weak, because failures are caused by combinations of sensors. The engineered features are stronger than the raw sensors they come from.

## Interactive Dashboard

A Streamlit app built on the same pipeline as the notebook. It is deployed on Streamlit Community Cloud (see the link at the top).

| Tab | What it shows |
|---|---|
| Overview | Class imbalance, failure rate by product type, sensor distributions |
| Lead indicator | Correlation of each sensor with each failure type |
| Models & threshold | FDR, recall, confusion matrix, and default vs tuned accuracy side by side |
| Cost-sensitive | Editable costs, FDR rule vs cost rule, sensitivity to the cost ratio |
| Anomaly detection | Isolation Forest scores and comparison with the supervised models |
| Live prediction | Enter sensor readings and get a failure-type prediction, with class probabilities, an anomaly score, a physical rule check and batch CSV upload |

The sidebar controls the model (Random Forest or XGBoost), the minimum recall, the costs and the decision threshold. Thresholds are chosen on a validation set, and all reported metrics come from the untouched test set.

The **physical rule check** in the Live prediction tab applies the documented failure rules of the dataset, independently of the model. When the model and the rules disagree (for example for Tool Wear), the app says so.

## Limitations

* **Tool Wear failures are not detected** (0 of 9 in the test set). In the dataset, these failures happen at a randomly chosen tool wear time between 200 and 240 minutes, and the class has only 45 examples. The sensors can show elevated risk but not the moment of failure.
* The test set contains only 66 failures, so a difference of one or two failures changes recall by 1.5 to 3 percentage points.
* The thresholds were chosen on a small validation set, so they could change with a different split.
* The costs used for the cost-sensitive step are assumed, not measured.
* The dataset has no time series, so "lead sensor" means the strongest association, not the sensor that changes first before a failure.
* The dataset is synthetic, so real-world performance is likely to be lower.

## Project Structure

```
.
├── data/
│   └── ai4i2020.csv
├── images/                            # figures generated by the notebook
├── notebooks/
│   ├── predictive_maintenance.ipynb   # analysis and modelling
│   ├── streamlit_dashboard.ipynb      # writes and launches the dashboard
│   ├── app.py                         # Streamlit app (user interface)
│   ├── pm_core.py                     # cleaning, features, models, metrics
│   ├── make_bundle.py                 # saves the pre-trained models
│   └── bundle.joblib                  # pre-trained models loaded by the app
├── requirements.txt
└── README.md
```

## How to Run

1. Clone the repository and create a virtual environment.
2. Install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure `ai4i2020.csv` is in the `data/` folder. If it is missing, download it from the UCI link above.
4. **Analysis notebook:** open `notebooks/predictive_maintenance.ipynb` and run all cells from the top. The notebook reads the data from `../data/` and saves figures to `../images/`.
5. **Dashboard:** run it locally with

   ```bash
   cd notebooks
   streamlit run app.py
   ```

   The app loads `bundle.joblib` if it exists. To rebuild it after changing the models, run `python make_bundle.py` from the same folder. If the CSV is somewhere else, upload it from the dashboard sidebar.

## Technologies

* Python
* pandas and NumPy
* Matplotlib and Seaborn (notebook figures)
* scikit-learn
* XGBoost
* Streamlit and Plotly (dashboard)
* joblib
* Git and GitHub

## Project Status

The analysis notebook and the Streamlit dashboard are complete, and the dashboard is deployed on Streamlit Community Cloud.

## Next Steps

* Per-class thresholds, for example a lower threshold for Tool Wear, to trade some false alarms for more detected failures.
* Time-to-Failure regression to estimate remaining useful life (optional extension). The dataset has no time series, so this would need a proxy target based on tool wear.
* Replace the assumed costs with real maintenance and downtime costs.

## Disclaimer

This is an educational machine learning project. Predictions are experimental and must not be used as the sole basis for real industrial safety or maintenance decisions.