
prompts = {
    "overlap_detection": """You are a seasoned machine learning expert. Your task is to determine whether a code snippet contains **overlap leakage**. Overlap Leakage happens overlap leakage occurs when actual data points, or near duplicates, appear in both training and test sets. This can happen due to improper splitting, data augmentation or resampling techniques on the entire dataset before splitting.
  You are given:
  - The **training method** name and the line number where it is called.
  - The **testing method** name and the line number where it is called.

  Your job is to use these fixed reference points to trace the origin and transformation of the training and evaluation datasets, and decide whether overlap leakage is present.

  ---

  ## Step-by-Step Instructions

  ### STEP 0: Locate Training and Evaluation Variables

  You are provided with:
  [Training method: <training_method>, Training line: <training_line>]
  [Testing method: <testing_method>, Testing line: <testing_line>]

  From these lines:
  1. Identify the training data used in the fit()/train() call (e.g., X_train). In some cases, the train and validation are internally performed with a single fit with the evaluation data being provided as a parameter (e.g., fit_generator(...,validation_data=...)) or the split is internally performed (e.g., GridSearchCV.fit(X_train, y_train) where X_train will be split into train and validation in each iteration).
  2. Identify the evaluation or test data used in predict(), score(), internally as a parameter (...,validation_data=...), or assumed to be created internally from a split (e.g., GridSearchCV.fit(X_train, y_train)).
  3. Trace each variable **backward** in the code to determine how the dataset was obtained (e.g., from a file, from a split, from transformations).

  ---

  ### STEP 1: Check for Overlap Leakage (Shortcut)

  Check if **any** of the following are true:

  - The same data object is passed to both training and evaluation steps.
    # Example 1:
    # model.fit(np.array(x_train), np.array(y_train))
    # y_train_pred = model.predict(x_train)

    # Example 2:
    # model = GradientBoostingClassifier()
    # model.fit(x_train.loc[:, selected_features], y_train)
    # train_score = model.score(x_train.loc[:, selected_features], y_train)

  OR

  - The test set is a subset of the training set.
    # Example:
    # x_train, x_test = train_test_split(df)
    # model.fit(df)
    # model.predict(x_test)

  OR

  - The same data object is passed to both training and evaluation steps as a parameter.
    # Example:
    # x_train, x_test, y_train, y_test = train_test_split(X, Y)
    # model.fit(X, validation_data=X)

  ✅ If **any** of these are true → Report overlap leakage immediately.

  🚫 Otherwise (including the case where we are dealing with an internal split like GridSearch.fit()), move to step 2.

  ---

  ### STEP 2: Identify Leak-Prone Transformations in Data Generation / Splitting

  Backtrack data flow from the variable names used for training and testing the model (identified in Step 0).
  Look for **resampling or synthetic data generation methods**, such as:
  - fit_resample()
  - RandomOverSampler
  - SMOTE
  - ADASYN
  - Any other oversampling or synthetic data generation pipeline

  Also check for **errors at the time of splitting** that may cause overlap:
  - Multiple `train_test_split()` calls with different random seeds on the same dataset
    # Example:
    # x_train1, x_test1 = train_test_split(df, test_size=0.2, random_state=1)
    # x_train2, x_test2 = train_test_split(df, test_size=0.2, random_state=42)
    # model.fit(x_train1, y_train1)
    # model.predict(x_test2)   # test2 overlaps with train1
  - Indexing mistakes where rows are shared between train and test sets
    # Example:
    # x_train = df.iloc[:1000]
    # x_test = df.iloc[500:1500]   # overlap between rows 500–999


  ⚠️ Do **not** catch preprocessing patterns like scaling, normalization, encoding, or fillna. These belong to preprocessing leakage and are outside the scope of this overlap leakage check.

  If you find any of the above, move to Step 3. If not, return:

  { "leakage_detected": false, "leakage_lines": [] }

  ---

  ### STEP 3: Check When the Transformation or Split Occurs

  For each transformation/split you found:
  - Is it applied on the entire dataset **before** the train-test split (e.g., oversampling applied to the full dataset, then splitting)?

  ✅ If that is the case, move to Step 4.
  🚫 If resampling/synthetic generation is applied **after splitting**, and only to the training data (with test data untouched or with test data augmented separately), it is **not leakage**. Return:

  { "leakage_detected": false, "leakage_lines": [] }

  ---

  ### STEP 4: Check Whether Both the Generated/Resampled Train and Test Data Are Used for Model Evaluation

  Check if:

  - The model is **evaluated in a separate step** on a test or validation set that came from the contaminated (resampled) source.
    # Example:
    # smote = SMOTE()
    # data_resampled, labels_resampled = smote.fit_resample(X, y)
    # x_train, x_test, y_train, y_test = train_test_split(data_resampled, labels_resampled)
    # model.fit(x_train, y_train)
    # model.predict(x_test)

  OR

  - The model is **fit on resampled data**, and the **evaluation is performed internally**, as in tools like `GridSearchCV`, which automatically perform validation using internal splitting on the resampled input.
    # Example:
    # smote = SMOTE()
    # X_res, y_res = smote.fit_resample(X, y)
    # x_train, x_test, y_train, y_test = train_test_split(X_res, y_res)
    # grid = GridSearchCV(model, params, cv=5)
    # grid.fit(x_train, y_train)

  OR

  - The model is **fit on contaminated training data**, and the **evaluation data is passed as a parameter** in the same method call.
    # Example:
    # smote = SMOTE()
    # X_res, y_res = smote.fit_resample(X, y)
    # x_train, x_test, y_train, y_test = train_test_split(X_res, y_res)
    # model.fit(X_res, y_res, validation_data=(X_res, y_res))

  ✅ If **both** are true → Report leakage.

  🚫 If only the training set is used, or the test set is loaded from an external source or is manually created (not from the same contaminated pipeline), it's **not overlap leakage**.

  ---

  ## Output Format

  Return the result in strict JSON:

  If overlap leakage exists:
  {
    "leakage_detected": true,
    "leakage_lines": [
      {
        "line_number": 12,
        "explanation": "SMOTE was applied to the entire dataset before splitting, contaminating both train and test sets."
      },
      {
        "line_number": 20,
        "explanation": "The resampled data was split into train and evaluation sets."
      },
      {
        "line_number": 45,
        "explanation": "The model was fit on the contaminated training data."
      },
      {
        "line_number": 60,
        "explanation": "The model was evaluated on the contaminated evaluation data."
      }
    ]
  }

  If there is **no** overlap leakage:
  {
    "leakage_detected": false,
    "leakage_lines": []
  }

    ---
    
  ## Reminder

  Do **not** flag based on hypothetical risks but only report clear direct leakage. Flag leakage after tracking data usage and if:
  - A transformation is clearly applied in a way that includes test data in training data. If a transformer is applied only on training data (after split), don't report it as potential leakage based on a hypothetical scenario where test has already influenced training.
  - AND both the training and test sets are used in model evaluation (or the contaminated training is used in a gridsearch or equivalent).
  """
,
    "preproc_detection": """You are a seasoned machine learning expert. Your task is to determine whether a code snippet contains **preprocessing leakage**. Preprocessing Leakage arises when transformations are applied to the entire dataset, inadvertently introducing information from the test set into the training phase. This can happen with methods that compute statistics or rely on data distribution characteristics, such as normalization, scaling, imputation, feature selection, clustering, or dimensionality reduction. When these steps access the test data characteristics before training, they skew model evaluation by allowing access to information that wouldn’t be available in real deployment, leading to inflated and misleading performance scores.
You are given:
- The **training method** name and the line number where it is called.
- The **testing method** name and the line number where it is called.

Your job is to use these reference points to trace the origin and transformation of the training and evaluation datasets, and decide whether preprocessing leakage is present.

---

## Step-by-Step Instructions

### STEP 0: Locate Training and Evaluation Variables

You are provided with:
[Training method: <training_method>, Training line: <training_line>]
[Testing method: <testing_method>, Testing line: <testing_line>]

From these lines:
1. Identify the training data used in the fit()/train() call (e.g., X_train). In some cases, the train and validation are internally performed with a single fit with the evaluation data being provided as a parameter (e.g., fit_generator(...,validation_data=...)) or the split is internally performed (e.g., GridSearchCV.fit(X_train, y_train) where X_train will be split into train and validation in each iteration).
2. Identify the evaluation or test data used in predict(), score(), internally as a parameter (...,validation_data=...), or assumed to be created internally from a split (e.g., GridSearchCV.fit(X_train, y_train)).
3. Trace each variable **backward** in the code to determine how the dataset was obtained (e.g., from a file, from a split, from transformations).

---

### STEP 1: Check for Overlap Leakage (Shortcut)

Check if **any** of the following are true:

- The same data object is passed to both training and evaluation steps.
  # Example 1:
  # model.fit(np.array(x_train), np.array(y_train), epochs=500)
  # y_train_pred = model.predict(x_train)

  # Example 2:
  # model = GradientBoostingClassifier()
  # model.fit(x_train.loc[:, selected_features], y_train)
  # train_score = model.score(x_train.loc[:, selected_features], y_train)

OR

- The test set is a subset of the training set.
  # Example:
  # x_train, x_test = train_test_split(df)
  # model.fit(df)
  # model.predict(x_test)

OR

- The same data object is passed to both training and evaluation steps as a parameter.
  # Example:
  # x_train, x_test, y_train, y_test = train_test_split(X, Y)
  # model.fit(X, validation_data= X)

🚫 If **any** of these are true, this is **overlap leakage**, which is **out of scope**.
Return:

{ "leakage_detected": false, "leakage_lines": [] }

✅ Otherwise (including the case where we are dealing with an internal split like GridSearch.fit()), move to step 2.

---

### STEP 2: Identify Leak-Prone Transformations in the preprocessing pipeline

Backtrack data flow from the variable names used for training and testing the model (identified in Step 0).
Look for transformations such as:
- Scaling, encoding, imputation, or feature engineering

If you find any of these, move to Step 3. If not, return:

{ "leakage_detected": false, "leakage_lines": [] }

---

### STEP 3: Check When the Transformation Occurs

For each transformation you found:
- Is it applied on the entire dataset from which a train and an eval/dev/test sets were created? (e.g., transforming before using train_test_split(), slicing, or other logic on the transformed data. Or using a scaler on a train and a test set after it has been fitted on the original unsplitted dataset)

✅ If that is the case, move to Step 4.
🚫 If the transformation is applied separately to the train and test data **after** splitting and the scaler is fit on the training data and transform the test/eval data, or test data is loaded separately and not involved, it is **not leakage**. Return:

{ "leakage_detected": false, "leakage_lines": [] }

---

### STEP 4: Check Whether Both the transformed Train and Test data that are the result of the split Are used for model evaluation

Check if:
- The model is **evaluated in a separate step** on a test or validation set that came from the same contaminated source.

  # Example:
  # scaled_data = scaler.fit_transform(df)
  # x_train, x_test, y_train, y_test = train_test_split(scaled_data, y)
  # model.fit(x_train, y_train)
  # model.predict(x_test)

OR
- The model is **fit on contaminated data**, and **evaluation is performed internally**, as in tools like `GridSearchCV`, which automatically perform validation using internal splitting on the input.
 
  # Example:
  # X_train = scaler.fit_transform(X_train)
  # grid_search = GridSearchCV(estimator=model, param_grid=parameters, scoring='accuracy', cv=5, verbose=0)
  # grid_search.fit(X_train, y_train)
OR

- The model is **fit on the contaminated training set**, and the **evaluation data is passed as a parameter** in the same method call (e.g., `fit(..., validation_data=...)`, `fit_generator(...)`, or `GridSearchCV.fit(...)`).

  # Example:
  # scaled_data = scaler.fit_transform(df)
  # x_train, x_test, y_train, y_test = train_test_split(scaled_data, y)
  # model.fit(x_train, y_train , validation_data=x_test)
  # model.predict(x_test)
✅ If **both** are true → Report leakage.

🚫 If only the training set is used, or the test set is loaded from an external source or is just a manually created entry and is not the test data resulting from the split, it's **not** preprocessing leakage.

  # Example: Manually created test set, not derived from training data
  # new_sample = pd.DataFrame([[...]], columns=...)
  # new_sample_transformed = transformer.transform(new_sample)
  # model.predict(new_sample_transformed)

  # Example: Train and test data are separately imported and transformed
  # train_df = pd.read_csv("train_data.csv")
  # test_df = pd.read_csv("external_test_data.csv")
  # X_train = train_df.drop("label", axis=1)
  # y_train = train_df["label"]
  # X_test = test_df.drop("label", axis=1)
  # y_test = test_df["label"]
  # categorical_cols = ["feature1", "feature2"]
  # feature_dfs = [X_train, X_test]
  # for col in categorical_cols:
  #   le = LabelEncoder()
  #   le.fit(feature_dfs[0][col])
  #   for df in feature_dfs:
  #       df[col] = le.transform(df[col])
  # model = RandomForestClassifier()
  # model.fit(X_train, y_train)
  # preds = model.predict(X_test)

  # Example: Same data (overlap leakage)
  # scaled_data = scaler.fit_transform(df)
  # x_train, x_test, y_train, y_test = train_test_split(scaled_data, y)
  # model.fit(x_train, y_train)
  # model.predict(x_train)

In that case, return:

{ "leakage_detected": false, "leakage_lines": [] }

---

## Output Format

Return the result in strict JSON:

If preprocessing leakage exists:
{
  "leakage_detected": true,
  "leakage_lines": [
    {
      "line_number": 12,
      "explanation": "Imputation is applied to the entire dataset before the train-test split, allowing test statistics to influence training"
    },
    {
      "line_number": 14,
      "explanation": "The imputed data is split into train and evaluation data."
    },
    {
      "line_number": 45,
      "explanation": "The model is fit on the contaminated training data."
    },
    {
      "line_number": 60,
      "explanation": "The model is evaluated on the contaminated evaluation data. Which is not a subset of the training data."
    }
  ]
}

If there is **no** preprocessing leakage:
{
  "leakage_detected": false,
  "leakage_lines": []
}

---

## Reminder

Do **not** flag based on hypothetical risks but only report clear direct leakage. Flag leakage after tracking data usage and if:
- A transformation is clearly applied in a way that includes future test data in training data. If a transformer is applied only on training data, don't report it as potential leakage based on a hypothetical scenario where test has already influenced training.
- AND both the training and test sets are used in model evaluation (but not the same data used for train and test).
"""
,
    }



# ⚠️ Special Case – Time Series Forecasting

#   In time series models, it is common to create test sequences that include a small overlap
#   with the end of the training data (e.g., test_data = data[train_len - n:, :]) so that each
#   test sample has the required historical context. This overlap is **not** overlap leakage
#   as long as:
#     - The overlap only contains past information already observed during training,
#     - No future values from the test horizon are used in training,
#     - Any data transformations (scaling, normalization, etc.) are fit only on the training data.

#   If these conditions hold, do **not** flag such overlaps as leakage.