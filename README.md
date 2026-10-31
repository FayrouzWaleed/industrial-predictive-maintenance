# Industrial Predictive Maintenance

A machine learning project that predicts **whether a machine will fail and which type of failure it will be**, using operational sensor readings. The main goal is not accuracy, but keeping **false alarms low**, because every false alarm stops production unnecessarily.

## Project Goals

* Predict the failure type (Tool Wear, Heat Dissipation, Power Failure, Overstrain) or "No Failure" from sensor readings.
* Handle the strong class imbalance (about 3.3% of machines fail).
* Evaluate models with the **False Discovery Rate (FDR)**, the share of alarms that are wrong, and tune the decision threshold to minimize it.
* Compare the FDR-based threshold with a **cost-based** threshold.
* Compare supervised models with an **anomaly detector**.
* Find which sensor is the lead indicator of each failure type.

## Dataset

AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository), 10,000 machine observations.

Source: https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Citation: S. Matzka, "Explainable Artificial Intelligence for Predictive Maintenance Applications," 2020 Third International Conference on Artificial Intelligence for Industries (AI4I), 2020. Licensed under CC BY 4.0.

The dataset is synthetic, so the results show how the methods behave, not how they would perform on a real factory.

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

## Limitations

* **Tool Wear failures are not detected** (0 of 9 in the test set). In the dataset, these failures happen at a randomly chosen tool wear time, and the class has only 45 examples.
* The test set contains only 66 failures, so a difference of one or two failures changes recall by 1.5 to 3 percentage points.
* The thresholds were chosen on a small validation set, so they could change with a different split.
* The costs used for the cost-sensitive step are assumed, not measured.
* The dataset has no time series, so "lead sensor" means the strongest association, not the sensor that changes first before a failure.

## Project Structure

```
.
├── data/
│   └── ai4i2020.csv
├── images/                      # figures generated by the notebook
├── notebooks/
│   └── predictive_maintenance.ipynb
└── README.md
```

## How to Run

1. Clone the repository and create a virtual environment.
2. Install the required packages:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
   ```

3. Make sure `ai4i2020.csv` is in the `data/` folder. If it is missing, download it from the UCI link above.
4. Open `notebooks/predictive_maintenance.ipynb` and run all cells from the top. The notebook reads the data from `../data/` and saves figures to `../images/`.

## Technologies

* Python
* pandas and NumPy
* Matplotlib and Seaborn
* scikit-learn
* XGBoost
* Git and GitHub

## Project Status

The analysis and modelling notebook is complete.

## Next Steps

* Time-to-Failure regression to estimate remaining useful life (bonus task). The dataset has no time series, so this would need a proxy target based on tool wear.
* Interactive Streamlit dashboard for exploring the data and testing predictions.
* Replace the assumed costs with real maintenance and downtime costs.

## Disclaimer

This is an educational machine learning project. Predictions are experimental and must not be used as the sole basis for real industrial safety or maintenance decisions.
