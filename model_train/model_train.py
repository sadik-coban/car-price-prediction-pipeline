# import sys
# import yaml
# from pipelines.ingest import ingest
# from pipelines.standardize import standardize
# from pipelines.quality import quality_check
# from pipelines.features import make_features
# from pipelines.train import train_model
# from pipelines.monitor import monitor_model

# def run_pipeline(config_path: str):
#     with open(config_path) as f:
#         config = yaml.safe_load(f)

#     raw = ingest(config["data"])
#     clean = standardize(raw)
#     checked = quality_check(clean)
#     features = make_features(checked)

#     train_model(features, config["training"])
#     monitor_model()

# if __name__ == "__main__":
#     run_pipeline(sys.argv[1])

from preprocess import preprocess
from train import train
import pandas as pd

df = pd.read_json("data/bmw/train/raw/arabam_details_bmw.jsonl", lines=True)
df.head()

preprocessed = preprocess(df)
model = train(preprocessed)
