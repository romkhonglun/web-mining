import pandas as pd

article_df = pd.read_parquet("data/kfujikawa/v0xxx_preprocess/v0101_article_inviews_in_split/train/dataset.parquet")
# hist_df = pd.read_parquet("data/ebnerd_large/train/history.parquet")
# behaviors_df = pd.read_parquet("data/ebnerd_large/train/behaviors.parquet")
print("article_df columns:", list(article_df.columns))
# print("hist_df columns:", list(hist_df.columns))
# print("behaviors_df columns:", list(behaviors_df.columns))