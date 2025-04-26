"""Process Titanic dataset."""
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def process_titanic_data(input_path: Path, train_output_path: Path, val_output_path: Path) -> None:
    """Process Titanic dataset with feature engineering.
    
    Args:
        input_path: Path to raw data file
        train_output_path: Path to save training data
        val_output_path: Path to save validation data
    """
    # Read data
    df = pd.read_csv(input_path)
    
    # Feature engineering
    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    
    # Create title feature from Name
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    title_mapping = {
        'Mr': 'Mr',
        'Miss': 'Miss',
        'Mrs': 'Mrs',
        'Master': 'Master',
        'Dr': 'Other',
        'Rev': 'Other',
        'Col': 'Other',
        'Major': 'Other',
        'Mlle': 'Miss',
        'Countess': 'Other',
        'Ms': 'Miss',
        'Lady': 'Other',
        'Jonkheer': 'Other',
        'Don': 'Other',
        'Dona': 'Other',
        'Mme': 'Mrs',
        'Capt': 'Other',
        'Sir': 'Other'
    }
    df['Title'] = df['Title'].map(title_mapping)
    
    # Create family size feature
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    
    # Create is_alone feature
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    
    # Encode categorical variables
    categorical_features = ['Sex', 'Embarked', 'Title']
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature])
    
    # Select features for the model
    features = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 
        'Embarked', 'Title', 'FamilySize', 'IsAlone'
    ]
    
    if 'Survived' in df.columns:
        features.append('Survived')
    
    # Scale numerical features
    scaler = StandardScaler()
    numerical_features = ['Age', 'Fare', 'FamilySize']
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Select final features
    df = df[features]
    
    # Split into train and validation sets if target exists
    if 'Survived' in df.columns:
        train_df, val_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df['Survived']
        )
        
        # Save processed data
        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(val_output_path, index=False)
    else:
        # If no target (test set), save as is
        df.to_csv(train_output_path, index=False)


if __name__ == "__main__":
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process train data and split into train/val
    process_titanic_data(
        raw_dir / "titanic_train.csv",
        processed_dir / "titanic_train.csv",
        processed_dir / "titanic_val.csv"
    )
    
    # Process test data
    process_titanic_data(
        raw_dir / "titanic_test.csv",
        processed_dir / "titanic_test_processed.csv",
        processed_dir / "titanic_test_val.csv"
    ) 