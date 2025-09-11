# make_dataset.py
import pathlib
import sys

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_data(data_path):
    # Load your dataset from a given path
    df = pd.read_csv(data_path)
    return df


def split_data(df, test_split, seed):
    # Split the dataset into train and test sets
    train, test = train_test_split(df, test_size=test_split, random_state=seed)
    return train, test


def save_data(train, test, output_path):
    # Save the split datasets to the specified output path
    output_dir = pathlib.Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv((output_dir / "train.csv").as_posix(), index=False)
    test.to_csv((output_dir / "test.csv").as_posix(), index=False)


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir / "params.yaml"
    params = yaml.safe_load(open(params_file.as_posix()))["make_dataset"]

    # Accept input path from CLI; default to data/raw/creditcard.csv
    input_arg = sys.argv[1] if len(sys.argv) > 1 else "data/raw/creditcard.csv"
    data_path = (home_dir / input_arg)
    output_path = (home_dir / "data" / "processed")

    data = load_data(data_path.as_posix())
    train_data, test_data = split_data(
        data, params["test_split"], params["seed"]
    )
    save_data(train_data, test_data, output_path.as_posix())


if __name__ == "__main__":
    main()
