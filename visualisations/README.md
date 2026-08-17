# Multi-Location Temporal Forecasting of Temperature and Relative Humidity Using a Stacked Ensemble Framework

## IndianWeatherCities

## Overview
This repository contains the implementation of a machine learning framework for forecasting near-surface temperature and relative humidity across multiple Indian cities.

The project applies a stacked ensemble learning approach to capture temporal patterns in climate data and improve forecasting accuracy across diverse climatic regions.

---

## Key Features
- Multi-location climate forecasting
- Stacked ensemble modelling framework (ARIMA, LSTM, and hybrid models)
- Time series preprocessing and feature engineering
- Model evaluation and performance comparison
- SHAP-based interpretability analysis
- Visualisation of forecast outputs

---

## Dataset
The dataset used in this study is publicly available:

**Indian Weather Repository – Live**  
https://www.kaggle.com/code/nelgiriyewithana/indian-weather-repository-live

---

## Repository Structure
The repository is organised as follows:

- `analysis/`  
  Contains the main analysis scripts, model development, and forecasting workflow.

- `archive/`  
  Includes earlier versions of the modelling scripts retained for transparency, version tracking, and reproducibility.

- `Charts/`  
  Visualisations and output plots generated during the analysis and forecasting process.

- `ne_110m_admin_0_countries/`  
  Geospatial data used for mapping and visual representation.

- `IndianWeatherRepository.csv`  
  Primary dataset used for model training and evaluation.

- `*.png` files  
  Forecast outputs and visual results.

- `*.html` files  
  Interactive visualisations (e.g., maps).

- `README.md`  
  Documentation describing the project, methodology, and usage instructions.

## How to Run

1. Clone the repository:
```bash
git clone https://github.com/Bilqis1424/IndianWeatherCities.git
cd IndianWeatherCities
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the model:
```bash
python india_weather_rev6.py
```

## Methodology
The project integrates classical statistical models and deep learning techniques within a stacked ensemble framework. It combines temporal feature extraction with machine learning to improve predictive performance across different climatic zones.

## Results
The model demonstrates improved forecasting accuracy compared to individual models, with consistent performance across multiple locations and time horizons.
