# ChloroFish
Erdos Deep Learning Bootcamp Team Project: Modeling predictive linkage between chlorophyll and fishery measures


This project focuses on modeling catch data using environmental and chlorophyll measurements from the Southeast Area Monitoring and Assessment Program (SEAMAP). The data, preprocessing and models are organized in the `data`, `src` and `models` directories, respectively.

## Key Data Source

The **Southeast Area Monitoring and Assessment Program (SEAMAP)** is a cooperative program that collects, manages, and disseminates fishery-independent data from the southeastern United States. SEAMAP operates across three regions—Gulf, South Atlantic, and Caribbean—conducting surveys that provide critical data for fisheries management.

## Data

This directory contains a number of csv folders which were used in the project. Any dataframe created in this project is stored in this folder.


## Preprocessing

- /src/data_preprocessing/Shrimp_preprocessing
    - Merges data files based on `STATIONID` (the project's method for tracking samples) and eliminating repeated `STATIONID`s.
    - Creates a dataframe from this merged data

- /src/data_preprocessing/Exploratory_Data_Analysis
    - Filters out incorrect measurements (e.g., temperatures over 100°C).
    - Identifies features that were highly correlated or largely missing.
    - Fills blank shrimp data with zeros (as per the Trawling Operations Manual).
    - Performs exploratory data analysis to identify predictive variables.

-src/visualizations/scatterplots
    - Generates scatterplots to understand the spatial and spatial-temporal structure of the shrimp dataset.

## Modeling

- /models/Model_comparison
    - Runs and evaluates a number of models for total shrimp biomass
        - Mean catch (baseline): MSE = 2.4884
        - XGBoost: MSE of 1.6099.
        - **Histogram Gradient Boosting:** MSE of 1.3901.
        - LinearRegressor (with imputation): MSE = 2.1474
        - SVMRegressor (with imputation): MSE = 2.0916
        - KNeighborsRegressor (with imputation): MSE = 1.9059
        - RandomForestRegressor (with imputation): MSE = 1.5493
        - AdaBoostRegressor (with imputation): MSE = 3.3975
        - GradientBoostingRegressor (with imputation): MSE = 1.6334
        - HistGradientBoostingRegressor (with imputation): MSE = 1.4342
    - Performs hyperparameter tuning for the histogram gradient boosting (HGB) model
    - Performs Shapley analysis to interpret the predictions of the HGB model
- /models/Neural_networks
    - Runs and evaluates two neural network models and a convolutional neural network for total shrimp biomass
        - Three layers with two hidden layers (64 and 32 neurons with ReLU activation) and an output layer with a single neuron: MSE = 1.8331.
        - Similar structure but using hyperbolic tangent as the activation function: MSE = 1.8194.
- /models/HistGradBoost_Shrimp
    - Experiments with different features for an HGB model
    - Uses HGB with a quantile loss function and a transformed target variable to obtain prediction intervals

## Conclusions

The **HistGradientBoostingRegressor** consistently performed the best, achieving an MSE of 1.3901 without imputation. Attempts to fine-tune and rescale the model did not yield significant improvements.

During a demo presentation, prediction intervals were requested for shrimp catch. The target was transformed using `f(x) = log(1+x)` to smooth the data, and prediction intervals were generated using quantile loss. Although the 90% prediction interval only contained 81% of the data, the results were reasonably accurate.

Shapley values were used to identify the most predictive features, revealing that location and time are key indicators, with environmental factors such as chlorophyll levels, temperature, and bottom oxygen also playing significant roles. High temperatures and low oxygen levels were found to be detrimental to shrimp populations.

## Repository Structure

- **data/**: Contains dataframes stored as csvs.
- **src/**: Contains data preprocessing scripts, visualizations, and exploratory data analysis.
- **models/**: Includes various models tested in this project, along with their performance metrics.

