# train_model.py
import pathlib
import sys

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier


def train_model(train_features, target, n_estimators, max_depth, seed):
    # Train your machine learning model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed
    )
    model.fit(train_features, target)
    return model


def save_model(model, output_path):
    # Save the trained model to the specified output path
    joblib.dump(model, output_path + "/model.joblib")


def main():

    curr_dir = pathlib.Path(__file__)
    home_dir = curr_dir.parent.parent.parent
    params_file = home_dir / "params.yaml"
    params = yaml.safe_load(open(params_file.as_posix()))["train_model"]
    # Accept input dir from CLI; default to data/processed
    input_arg = sys.argv[1] if len(sys.argv) > 1 else "data/processed"
    data_dir = home_dir / input_arg
    output_path = home_dir / "models"
    output_path.mkdir(parents=True, exist_ok=True)

    TARGET = "Class"
    train_features = pd.read_csv((data_dir / "train.csv").as_posix())
    X = train_features.drop(TARGET, axis=1)
    y = train_features[TARGET]

    trained_model = train_model(
        X, y, params["n_estimators"], params["max_depth"], params["seed"]
    )
    save_model(trained_model, output_path.as_posix())


if __name__ == "__main__":
    main()
