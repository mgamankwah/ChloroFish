# ChloroFish
This repository was created as part of the Erdos Institute Deep Learning Bootcamp Team Project and New Atlantis Labs Ocean Data Fellowship. The research question was to model the predictive linkage between chlorophyll and fishery measures. To address this question, we studied catch data and environmental/chlorophyll measurements from the Southeast Area Monitoring and Assessment Program (SEAMAP). The data, preprocessing and models are organized in the `data`, `src` and `models` directories, respectively.

## Repository Structure

- **data/**: Contains dataframes stored as csvs.
- **src/**: Contains data preprocessing scripts, visualizations, and exploratory data analysis.
- **models/**: Includes various models tested in this project, along with their performance metrics.

## Key Data Source

The **Southeast Area Monitoring and Assessment Program (SEAMAP)** is a cooperative program that collects, manages, and disseminates fishery-independent data from the southeastern United States. SEAMAP operates across three regions—Gulf, South Atlantic, and Caribbean—conducting surveys that provide critical data for fisheries management.

# Shrimp Population Modeling

## Preprocessing

- /src/data_preprocessing/Shrimp_preprocessing
    - Merges data files based on `STATIONID` (the project's method for tracking samples) and eliminating repeated `STATIONID`s.
    - Creates a dataframe from this merged data

- /src/data_preprocessing/Exploratory_Data_Analysis
    - Filters out incorrect measurements (e.g., temperatures over 100°C).
    - Identifies features that were highly correlated or largely missing.
    - Fills blank shrimp data with zeros (as per the Trawling Operations Manual).
    - Performs exploratory data analysis to identify predictive variables.
- /src/visualizations/scatterplots
    - Generates scatterplots to understand the spatial and spatial-temporal structure of the shrimp dataset.
- /src/data_preprocessing/BIO_DIV
    - 
## Modeling

- /models/Model_comparison
    - Runs and evaluates a number of models for total shrimp biomass
        - Mean catch (baseline): MSE = 2.4884
        - XGBoost: MSE of 1.6099
        - **Histogram Gradient Boosting:** MSE of 1.3901
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
        - Three layers with two hidden layers (64 and 32 neurons with ReLU activation) and an output layer with a single neuron: MSE = 1.8331
        - Similar structure but using hyperbolic tangent as the activation function: MSE = 1.8194
- /models/HistGradBoost_Shrimp
    - Experiments with different features for an HGB model
    - Uses HGB with a quantile loss function and a transformed target variable to obtain prediction intervals
    - Trains the HGB model on the full training set and tests it on the test set to obtain the final mean squared error

## Conclusions

The **HistGradientBoostingRegressor** consistently performed the best, achieving an MSE of 1.3901 without imputation on the validation set. Attempts to fine-tune and rescale the model did not yield significant improvements. When we trained the model on the full training set and evaluated it on the testing set, this model obtain a MSE of 1.3514.

During a demo presentation, prediction intervals were requested for shrimp catch. The target was transformed using `f(x) = log(1+x)` to smooth the data, and prediction intervals were generated using quantile loss. The results were reasonably accurate, but the prediction intervals were a bit narrow, with the 90% prediction interval containing 84% of the data.

Shapley values were used to identify the most predictive features, revealing that location and time are key indicators, with environmental factors such as chlorophyll levels, temperature, and bottom oxygen also playing significant roles. High temperatures and low oxygen levels were found to be detrimental to shrimp populations.




# Biodiversity Population Modeling

Typically, a more healthy ecosystem has a high variety of species and is “well-balanced” (i.e. no invasive species takes over the ecosystem). Therefore, another approach with the data was to model the species diversity by using species IDs (unique tag for each species) and their extrapolated counts during the survey expeditions. The main challenges with this approach are:
- Creating a good metric, or metrics, for a diversity ecosystem
- Gathering the data in such a way that preserves useful information


## Preprocessing

- /src/data_preprocessing/BIO_DIV
    - Filters out incorrect measurements (e.g., temperatures over 100°C).
    - Groups entries by day to get unique species caught daily
    - Calculates different species diversity health metrics

<img src="https://github.com/mgamankwah/ChloroFish/blob/main/images/shannon_entropy.png" width="512">

## Modeling

- /models/BIODIV_models
    - Runs and evaluates a number of models for predicting biodiversity marker of Shannon entropy (values range from 0 - 4)
        - Linear model (with imputation): RMSE = 0.637
        - XGBoost: RMSE = 0.515
        - 2 layer Neural Network (with imputation): RMSE = 0.538
        - **2 layer CNN (with imputation):** RMSE = 0.507
        - MultiRegressor XGBoost: RMSE = 0.525
    - Performed XGBoost tree pruning and network exploration to prevent overfitting


<img src="https://github.com/mgamankwah/ChloroFish/blob/main/images/Linear_predictions.png" width="256"><img src="https://github.com/mgamankwah/ChloroFish/blob/main/images/CNN_predictions.png" width="256">


## Conclusions

The **CNN** was the best performing model by a small margin but overall all models seem to have the same overall performance and hit the same RMSE barrier limited that couls be caused by:
- Low amount of ~7000 data points
- Naive averaging of spatiotemporal quantities that may introduce error


The complex nature of the Shannon entropy through time (but not through space) may be too abrupt to accurately capture for all models

<img src="https://github.com/mgamankwah/ChloroFish/blob/main/images/Shannon_entropy_time_series.png" width="780">

Model improvements could be made in:
- Using a graph network NN model for locality & adjacency
- Augment certain data quantities with richer data (i.e. satellite surface temperature data)
- Assess other metrics of species diversity that better capture ecosystem health