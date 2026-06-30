# Agricultural Supply Chain Loss Prediction

Predicting crop loss percentage (`loss_percent`) for a shipment based on storage
conditions, transport mode, and travel details — a simplified, end-to-end
version of a supply-chain spoilage prediction project.

## Problem

In India, 10–30% of crops spoil before reaching market due to poor storage and
transport. This project builds a model that predicts how much loss (%) a given
shipment is likely to experience, based on factors like storage duration,
temperature, humidity, distance traveled, and transport mode.

## About the data

Real shipment-level spoilage data isn't publicly available at the scale needed
for this project, so the dataset (`data/shipments.csv`, ~2,000 rows) is
**synthetically generated** using domain-informed rules — see
[`generate_data.py`](generate_data.py). Longer storage, higher humidity, higher
temperature, and longer delays all increase simulated loss, with random noise
added so the relationships aren't perfectly clean. This is disclosed openly
rather than presented as real-world data; the goal is to demonstrate the full
analysis and modeling workflow.

| Column | Type | Description |
|---|---|---|
| shipment_id | int | unique shipment ID |
| crop_type | categorical | e.g. Tomato, Rice, Mango |
| storage_days | numeric | days spent in storage |
| temperature | numeric | °C |
| humidity | numeric | % |
| distance_km | numeric | shipment distance |
| transport_mode | categorical | Truck, Rail, Refrigerated Truck, Open Cart |
| delay_hours | numeric | transport delay |
| region | categorical | North/South/East/West/Central |
| loss_percent | numeric (target) | % of shipment lost to spoilage |

## Project structure

```
agri_project/
├── data/
│   └── shipments.csv          # generated dataset
├── images/                    # exported charts (used below)
├── notebooks/
│   └── agri_loss_prediction.ipynb   # full analysis notebook
├── generate_data.py            # creates the synthetic dataset
├── build_notebook.py           # generates the notebook file (optional, for reference)
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
python generate_data.py                 # creates data/shipments.csv
jupyter notebook notebooks/agri_loss_prediction.ipynb
```

## Workflow

1. **Data cleaning** — handle missing values, duplicates, invalid ranges.
2. **SQL-style analysis** — average loss by crop, region, and transport mode
   (done in pandas `groupby`, with equivalent SQL shown as comments).
3. **EDA** — scatter plots, histograms, and a correlation heatmap to check
   relationships between storage/humidity/temperature and loss.
4. **Feature engineering** — added `risk_score = temperature × humidity`.
5. **Modeling** — Linear Regression (baseline) vs. Random Forest (final model),
   80/20 train-test split.

## Results

| Model | R² | MAE |
|---|---|---|
| Linear Regression (baseline) | 0.83 | 2.98 |
| Random Forest (final) | 0.84 | 2.88 |

*(Numbers come from this run; results vary slightly between runs of the random
data generator.)*

### Key charts

**Storage days & humidity vs loss**

![Storage and Humidity](images/scatter_storage_humidity.png)

**Correlation heatmap**

![Correlation Heatmap](images/correlation_heatmap.png)

**Average loss by crop type**

![Loss by Crop](images/loss_by_crop.png)

**Predicted vs actual loss (Random Forest)**

![Actual vs Predicted](images/actual_vs_predicted.png)

**Feature importance**

![Feature Importance](images/feature_importance.png)

## Key insights

- Loss % increases with storage duration and humidity, matching real-world
  spoilage behavior.
- Perishable crops (tomato, banana, mango) show higher average loss than
  grains (wheat, rice).
- Refrigerated transport reduces average loss compared to open-cart transport.

## Limitations & next steps

This is a learning/demonstration project built on synthetic data. To make it
production-relevant, the simulation rules should be replaced or validated
against real shipment/spoilage records — e.g. from
[FAOSTAT](https://www.fao.org/faostat/), India's
[Open Government Data Platform](https://data.gov.in/), or a cold-chain
logistics partner.

## Tools used

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, Jupyter
