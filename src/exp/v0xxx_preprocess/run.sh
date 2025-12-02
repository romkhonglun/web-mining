#!/bin/bash -ex

FILE_DIR=$(cd $(dirname $0); pwd)
OPTIONS="--skip"
#OPTIONS="--overwrite"

# Base datasets
uv run python $FILE_DIR/v0001_download_rawdata.py download
uv run python $FILE_DIR/v0001_download_rawdata.py unzip

# Base datasets
uv run python $FILE_DIR/v0100_articles.py $OPTIONS
uv run python $FILE_DIR/v0200_users.py $OPTIONS
uv run python $FILE_DIR/v0300_impressions.py $OPTIONS

# Additional features
uv run python $FILE_DIR/v0101_article_inviews_in_split.py $OPTIONS
uv run python $FILE_DIR/v0101_article_inviews_in_split_v2.py $OPTIONS
uv run python $FILE_DIR/v0102_article_metadata_id_v2.py $OPTIONS
uv run python $FILE_DIR/v0103_article_history_counts.py $OPTIONS
uv run python $FILE_DIR/v0201_user_inviews_in_split.py $OPTIONS
uv run python $FILE_DIR/v0301_imp_counts_per_user.py $OPTIONS