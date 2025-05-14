# E-Commerce Recommendation System

This project implements a collaborative filtering-based recommendation system using real-world Amazon product ratings data. It is designed to suggest beauty products to users based on historical rating behavior.

## Overview

- **Dataset**: [Amazon Beauty Ratings (Kaggle)](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings)
- **Model**: Singular Value Decomposition (SVD) via the [Surprise](http://surpriselib.com/) library
- **Goal**: Predict user ratings for products they haven’t rated yet and evaluate the model’s accuracy.

## Project Structure

- `E_Commerce_Recommendation_System.ipynb`: Main notebook containing data preprocessing, model training, evaluation, and prediction.

## How to Run

1. Clone the repo or open the notebook in Google Colab.
2. Install dependencies:
   ```bash
   pip install surprise
   ```
3. Run all cells in the notebook.

## Features

- Label encoding for user and product IDs
- SVD model for collaborative filtering
- Cross-validation using RMSE and MAE
- Grid search for hyperparameter tuning
- Example prediction visualization

## Model Performance

- Final RMSE: ~1.26 on test set (depending on split)
- Predicts ratings on unseen user-product pairs

## Future Improvements

- Include content-based features (e.g., product metadata)
- Try deep learning-based recommenders (e.g., autoencoders)
- Deploy model as an API for real-time recommendations

---
