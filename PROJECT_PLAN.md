# Project Plan

## Phase 1: Data Acquisition and Validation

Identify suitable public datasets, document their sources, and validate timestamps, units, missing values, and coverage before analysis.

## Phase 2: Exploratory Analysis

Explore demand patterns, seasonality, calendar effects, missing data, outliers, and any data quality concerns.

## Phase 3: Baseline Forecasting

Build simple baseline forecasts first so later models can be compared against a clear starting point.

## Phase 4: Leakage-Safe Feature Engineering

Create lag, rolling, calendar, and weather-related features while preventing future information from entering the training data.

## Phase 5: Model Comparison

Compare selected models using chronological train, validation, and test splits with appropriate forecasting metrics.

## Phase 6: Renewable and Net-Load Analysis

Analyze renewable generation alongside demand to understand net-load patterns and planning implications.

## Phase 7: Streamlit Application

Build an interactive app only after real analysis, models, and results are available.

## Phase 8: Documentation and Deployment

Document data sources, methods, limitations, and results, then prepare the project for sharing or deployment.
