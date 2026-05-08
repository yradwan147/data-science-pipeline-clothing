# Data Science Pipeline — Women's Clothing Reviews

Final project for Udacity's *Data Scientist Nanodegree* pipelines section
(nd025). Builds a single scikit-learn pipeline that mixes **numerical**,
**categorical**, and **two text columns** (review title + body) to predict
whether a customer recommended a piece of clothing.

## Notebook flow

1. **Data exploration** — class balance, age distribution, recommend-rate
   per department.
2. **Pipeline construction** — `ColumnTransformer` with four branches:
   - numeric: `SimpleImputer(median) → StandardScaler`,
   - categorical: `SimpleImputer(constant) → OneHotEncoder(min_frequency=20)`,
   - title text: `TfidfVectorizer(1-2-grams, max_features=400)`,
   - review body text: `TfidfVectorizer(1-2-grams, max_features=2000)`.
3. **Training** — `GradientBoostingClassifier` baseline; reports
   accuracy + ROC-AUC + per-class precision/recall.
4. **Fine-tuning** — `GridSearchCV` over `n_estimators`, `max_depth`,
   `learning_rate` with 3-fold CV, scoring on AUC.

## Standing-out work

* The two text columns are vectorised independently — the title TF-IDF
  is a small, focused token set; the review body TF-IDF is much wider
  and stop-word filtered.
* `OneHotEncoder(min_frequency=20)` collapses tail categories into an
  "infrequent" bucket, so the model doesn't memoise rare clothing IDs.
* The whole preprocessor + classifier is wrapped in a single
  `Pipeline`, so cross-validation and grid search apply to the full
  feature stack — no leakage.

## Running

```bash
pip install pandas scikit-learn matplotlib seaborn jupyter
jupyter notebook starter/starter.ipynb
# Restart & Run All
```

## License

Educational submission for Udacity nd025. Starter scaffold + dataset
© Udacity / Kaggle Women's E-Commerce Clothing Reviews.
