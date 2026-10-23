# E-Commerce Recommendation System

Collaborative filtering on real Amazon product ratings: given a user and a beauty product they haven't rated, predict the rating they'd give it. Built with SVD (matrix factorization) in the [Surprise](https://surpriselib.com/) library.

## Data

[Amazon Beauty ratings (Kaggle)](https://www.kaggle.com/datasets/skillsmuggler/amazon-ratings): one row per rating, with a user ID, product ID, a 1–5 star rating and a timestamp. The CSV (`ratings_Beauty.csv`) isn't in this repo; download it from Kaggle and place it next to the notebook.

## Approach

1. Drop missing rows and encode user and product IDs as integers.
2. Count ratings per product to see the most-rated products.
3. Split 80/20 into train and test (`random_state=42`).
4. Train Surprise's `SVD` and cross-validate it (3 folds) on the training set.
5. Score the held-out test set.
6. Grid search over epochs, learning rate and regularization.

## Results

All numbers below are printed by the notebook.

| Evaluation | RMSE | MAE |
|---|---|---|
| 3-fold cross-validation on the training set (mean) | 1.2633 | 0.9885 |
| Held-out test set (20%) | 1.2623 | not computed |
| Best grid-search setting (`n_epochs=10, lr_all=0.005, reg_all=0.4`), CV | 1.2660 | |

Ratings are on a 1–5 scale, so predictions are off by about one star on average. The tuned setting scored slightly worse than the default SVD, so the default model is the one evaluated on the test set.

## Run it

```bash
pip install pandas numpy scikit-learn scikit-surprise
jupyter notebook E_Commerce_Recommendation_System.ipynb
```

Or open the notebook in Google Colab and upload `ratings_Beauty.csv`.

## Limitations

- **No baseline.** The notebook doesn't compare SVD against simple predictors such as the global mean or per-product mean rating. Amazon ratings lean heavily toward 5 stars, so a baseline is needed to tell how much SVD actually adds.
- **The test-set model wasn't refit.** The model scored on the test set is the one left over from the last cross-validation fold, trained on about two-thirds of the training data rather than all of it.
- **Rating prediction only.** It predicts ratings but doesn't produce or evaluate top-N recommendation lists (precision@k, recall@k), which is what a store would show a user.
- **Cold start.** Users and products with no ratings in training get no meaningful prediction.
- The notebook's "Scalability and Performance" section lists general serving strategies (caching, load balancing); none are implemented here.

## Possible next steps

- Add mean-rating baselines and refit the final model on the full training set.
- Evaluate top-N recommendations with precision@k and recall@k.
- Add content-based features from product metadata.
- Serve the model behind an API.
