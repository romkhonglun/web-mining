#!/bin/bash -ex

# shellcheck disable=SC2046
FILE_DIR=$(cd $(dirname "$0"); pwd)

# Base datasets
uv run python "$FILE_DIR"/NewsEncoding.py --model_name "all-MiniLM-L6-v2" \
  --batch_size 16 \
  --input "data/ebnerd_small/articles.parquet" \
  --output "data/small_embeddings_all-MiniLM-L6-v2.parquet"
uv run python "$FILE_DIR"/GenerateCategoryDescription.py \
  --input "data/ebnerd_small/articles.parquet" \
  --output "data/small_articles_with_cat_desc.parquet" \
  --id-col "article_id" \
  --cat-col "category_str" \
  --model "Qwen/Qwen3-4B-Instruct-2507-FP8"
PYTHON_PATH=src uv run python src/newsRecSys/model/naml/GenerateCategoryDescription.py \
  --input "data/ebnerd_small/articles.parquet" \
  --output "data/small_articles_with_cat_desc.parquet" \
  --id-col "article_id" \
  --cat-col "category_str" \
  --model "Qwen/Qwen3-4B-Instruct-2507"