
import pandas as pd
import numpy
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def preprocessing(file_path, output_path):

    dataframe = pd.read_csv(file_path)
    processed_data = dataframe.copy()

    label_encoder = LabelEncoder()

    processed_data["class"] = label_encoder.fit_transform(
    processed_data["class"]
    )

    # Pisahkan fitur dan data
    y = processed_data["class"]
    X = processed_data.drop(columns=["class"])

    # Encode kategori dengan One Hot Encoder karena nilai berupa kategori bukan rentang
    encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
    )

    X_encoded = encoder.fit_transform(X)

    # Ubah ke dataframe
    feature_names = encoder.get_feature_names_out(X.columns)

    processed_data = pd.DataFrame(
    X_encoded,
    columns=feature_names
    )

    # Add encoded target
    processed_data["class"] = y.to_numpy()
    # 6. Save processed dataset
    processed_data.to_csv(output_path, index=False)

    print(f"Processed data saved to {output_path}")

    return processed_data


if __name__ == "__main__":

    print(Path(__file__).resolve().parent.parent)
    output_file_path = Path(__file__).resolve().parent / "processed_data.csv"
    input_file_path = Path(__file__).resolve().parent.parent / "mushrooms.csv"

    preprocessing(input_file_path, output_file_path)
