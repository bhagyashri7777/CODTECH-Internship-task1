# Import libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# -------------------------
# 1. EXTRACT
# -------------------------
#def extract_data(C:\Users\bhagy\OneDrive\Desktop\Intern task\dataset.csv):
  #  df = pd.read_csv(r"C:\Users\bhagy\OneDrive\Desktop\Intern task\dataset.csv")
  #  return df
def extract_data(file_path):
    df = pd.read_csv(file_path)
    return df

data = extract_data(r"C:\Users\bhagy\OneDrive\Desktop\Intern task\dataset.csv")
print(data.head())

# -------------------------
# 2. TRANSFORM
# -------------------------
def transform_data(df):
    # Separate features
    X = df.drop("purchased", axis=1)
    y = df["purchased"]

    # Numerical and categorical columns
    numerical_features = ["age", "income"]
    categorical_features = ["city"]

    # Numerical pipeline
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ]
    )

    X_transformed = preprocessor.fit_transform(X)
    return X_transformed, y

# -------------------------
# 3. LOAD
# -------------------------
def load_data(X, y, output_file):
    processed_df = pd.DataFrame(X.toarray() if hasattr(X, "toarray") else X)
    processed_df["target"] = y.values
    processed_df.to_csv(output_file, index=False)

# -------------------------
# MAIN PIPELINE
# -------------------------
if __name__ == "__main__":
    data = extract_data("dataset.csv")
    X_processed, y = transform_data(data)
    load_data(X_processed, y, "processed_dataset.csv")

print("ETL Pipeline executed successfully!")
