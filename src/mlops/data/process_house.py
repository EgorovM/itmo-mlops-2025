"""Process House Price dataset."""
from pathlib import Path

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def process_house_data(
    input_path: Path,
    train_output_path: Path,
    val_output_path: Path,
    test_output_path: Path,
) -> None:
    """Process House Price dataset with feature engineering.

    Args:
        input_path: Path to raw data file
        train_output_path: Path to save training data
        val_output_path: Path to save validation data
        test_output_path: Path to save test data
    """
    # Read data
    df = pd.read_csv(input_path)

    # Drop unnecessary features first
    features_to_drop = [
        "Id",
        "Utilities",
        "Street",
        "PoolQC",
        "MiscFeature",
        "Alley",
        "Fence",
        "FireplaceQu",
    ]
    df = df.drop(features_to_drop, axis=1, errors="ignore")

    # Handle missing values
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df.select_dtypes(include=["object"]).columns

    # Fill numerical missing values with median
    num_imputer = SimpleImputer(strategy="median")
    df[numerical_features] = num_imputer.fit_transform(df[numerical_features])

    # Fill categorical missing values with mode
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_features] = cat_imputer.fit_transform(df[categorical_features])

    # Feature engineering
    # Total square footage
    df["TotalSF"] = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]

    # Total bathrooms
    df["TotalBathrooms"] = (
        df["FullBath"]
        + df["HalfBath"] * 0.5
        + df["BsmtFullBath"]
        + df["BsmtHalfBath"] * 0.5
    )

    # House age and remodel age
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df["RemodAge"] = df["YrSold"] - df["YearRemodAdd"]

    # Total porch area
    df["TotalPorchSF"] = (
        df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]
    )

    # Update numerical features after feature engineering
    numerical_features = df.select_dtypes(include=["int64", "float64"]).columns

    # Encode categorical variables
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])

    # Scale numerical features
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])

    # Split into train and validation sets if target exists
    if "SalePrice" in df.columns:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

        # Save processed data
        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(val_output_path, index=False)
    else:
        # If no target (test set), save as is
        df.to_csv(test_output_path, index=False)


if __name__ == "__main__":
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Process train data and split into train/val
    process_house_data(
        raw_dir / "house_train.csv",
        processed_dir / "house_train.csv",
        processed_dir / "house_val.csv",
        processed_dir / "house_test.csv",
    )

    # Process test data
    process_house_data(
        raw_dir / "house_test.csv",
        processed_dir / "house_train.csv",
        processed_dir / "house_val.csv",
        processed_dir / "house_test.csv",
    )
