# v0300_impressions.py (Đã sửa đổi)

import shutil
import sys
from pathlib import Path
from typing import Optional

import polars as pl
import typer
from loguru import logger
from typing_extensions import Annotated

from newsRecSys.utils._behaviors import create_binary_labels_column
from exputils.const import RAWDATA_DIRS, PREPROCESS_DIR
from exputils.utils import timer

# --- Hằng số mới ---
NUM_CHUNKS = 10  # Số lượng chunk áp dụng cho tất cả các splits
# --------------------

APP = typer.Typer(pretty_exceptions_enable=False)
FILE_NAME = Path(__file__).stem
OUTPUT_DIR = PREPROCESS_DIR / FILE_NAME
ARTICLES_DIR = PREPROCESS_DIR / "v0100_articles"
USERS_DIR = PREPROCESS_DIR / "v0200_users"


def prepare_output_dir(overwrite: bool | None):
    # (Giữ nguyên)
    if OUTPUT_DIR.exists():
        if overwrite or (overwrite is None and typer.confirm(f"Delete {OUTPUT_DIR}?")):
            logger.debug(f"Delete {OUTPUT_DIR}")
            shutil.rmtree(OUTPUT_DIR)
        else:
            logger.info(f"Skip to overwrite {OUTPUT_DIR}")
            sys.exit(0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return


def compute_impressions(
        lf_impressions: pl.LazyFrame,
        lf_history: pl.LazyFrame,
        lf_articles: pl.LazyFrame,
) -> pl.LazyFrame:
    """Tính toán các trường cần thiết cho impressions LazyFrame."""

    # 🚨 SỬA ĐỔI QUAN TRỌNG: Đã loại bỏ .with_row_index() vì nó được tạo ở main()
    lf_impressions = lf_impressions.with_columns(
        pl.col("user_id").cast(pl.Int32),
        # 🚨 Đảm bảo 'impression_index' đã tồn tại và đúng
        pl.col("impression_index").cast(pl.Int32),
    )
    # ... (Các logic tiếp theo giữ nguyên) ...
    if "article_ids_clicked" not in lf_impressions.collect_schema().names():
        # ... (giữ nguyên)
        lf_impressions = lf_impressions.with_columns(
            pl.lit([]).cast(pl.List(pl.Int32)).alias("article_ids_clicked"),
            pl.lit(0).cast(pl.UInt16).alias("next_read_time"),
            pl.lit(0).cast(pl.UInt8).alias("next_scroll_percentage"),
        )

    # ... (Các logic Join, Xử lý Inview, Xử lý Clicked, Kết hợp giữ nguyên) ...
    lf_impressions = lf_impressions.join(
        lf_history.select("user_id", "user_index", "in_small"),
        on="user_id",
        validate="m:1",
    )
    lf_impressions_inview = (
        lf_impressions.select("impression_index", "article_ids_inview")
        .explode("article_ids_inview")
        .cast(pl.Int32)
        .join(
            lf_articles.select("article_id", "article_index"),
            left_on="article_ids_inview",
            right_on="article_id",
        )
        .group_by("impression_index", maintain_order=True)
        .agg(pl.col("article_index").alias("article_indices_inview"))
    )
    lf_impressions_click = (
        lf_impressions.select("impression_index", "article_ids_clicked")
        .explode("article_ids_clicked")
        .cast(pl.Int32)
        .join(
            lf_articles.select("article_id", "article_index"),
            left_on="article_ids_clicked",
            right_on="article_id",
            validate="m:1",
        )
        .group_by("impression_index", maintain_order=True)
        .agg(pl.col("article_index").alias("article_indices_clicked"))
    )
    lf_impressions = (
        lf_impressions.join(
            lf_impressions_inview,
            on="impression_index",
            how="left",
            validate="1:1",
        )
        .join(
            lf_impressions_click,
            on="impression_index",
            how="left",
            validate="1:1",
        )
        .pipe(
            create_binary_labels_column,  # type: ignore
            clicked_col="article_indices_clicked",
            inview_col="article_indices_inview",
            shuffle=False,
            seed=123,
        )
        .with_columns(
            (pl.col("impression_time").dt.timestamp() // 10 ** 6)
            .cast(pl.Int32)
            .alias("impression_ts"),
        )
        # Bỏ sort("impression_index") để giữ tính lazy, chỉ sort khi cần thiết
        .select(
            pl.col("impression_index"),
            pl.col("impression_id"),
            pl.col("impression_ts"),
            pl.col("impression_time"),
            pl.col("user_index"),
            pl.col("session_id"),
            pl.col("read_time").fill_null(0).cast(pl.UInt16),
            pl.col("scroll_percentage").fill_null(0).cast(pl.UInt8),
            pl.col("device_type").cast(pl.Int8),
            pl.col("is_sso_user").cast(bool),
            pl.col("gender").fill_null(-1).cast(pl.Int8),
            pl.col("postcode").fill_null(-1).cast(pl.Int8),
            pl.col("age").fill_null(-1).cast(pl.Int8),
            pl.col("is_subscriber").cast(bool),
            pl.col("next_read_time").fill_null(0).cast(pl.UInt16),
            pl.col("next_scroll_percentage").fill_null(0).cast(pl.UInt8),
            pl.col("article_indices_inview").fill_null(pl.lit([])),
            pl.col("article_indices_clicked").fill_null(pl.lit([])),
            pl.col("in_small").cast(bool),
            pl.col("labels"),
        )
    )
    return lf_impressions


@APP.command()
def main(
        overwrite: Annotated[Optional[bool], typer.Option("--overwrite/--skip")] = None,
):
    prepare_output_dir(overwrite=overwrite)

    lf_articles = pl.scan_parquet(ARTICLES_DIR / "dataset.parquet")

    for split in ["train", "validation", "test"]:

        impressions_file = RAWDATA_DIRS[split] / "behaviors.parquet"
        lf_history = pl.scan_parquet(USERS_DIR / split / "dataset.parquet")
        output_path_final = OUTPUT_DIR / split / "dataset.parquet"
        output_path_final.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"*** Bắt đầu xử lý {split} với {NUM_CHUNKS} chunks ***")

        # 1. Quét LazyFrame GỐC và TẠO CHỈ MỤC TOÀN CỤC MỘT LẦN
        lf_full = pl.scan_parquet(impressions_file).with_row_index(
            name="impression_index"
        )

        with timer(f"Get total rows for chunking ({split})"):
            n_rows_input = lf_full.select(pl.len()).collect().item()

        chunk_size = (n_rows_input + NUM_CHUNKS - 1) // NUM_CHUNKS

        logger.info(f"Tổng số hàng: {n_rows_input}. Kích thước mỗi chunk: {chunk_size}")

        processed_chunks = []

        for i in range(NUM_CHUNKS):
            offset = i * chunk_size
            limit = chunk_size

            if offset >= n_rows_input:
                break

            logger.info(f"Processing chunk {i + 1}/{NUM_CHUNKS} (offset={offset}, limit={limit})...")

            # 2. Slicing LazyFrame (đã có chỉ mục)
            lf_chunk = lf_full.slice(offset, limit)

            # 3. Thực hiện tính toán
            lf_output_chunk = compute_impressions(
                lf_impressions=lf_chunk,  # Lf_chunk đã có 'impression_index' đúng
                lf_history=lf_history,
                lf_articles=lf_articles,
            )

            with timer(f"Collect chunk {i + 1}"):
                df_output_chunk = lf_output_chunk.collect(engine="streaming")

            processed_chunks.append(df_output_chunk)
            logger.info(f"Chunk {i + 1} shape: {df_output_chunk.shape}")

        # 4. Ghép nối và Ghi kết quả cuối cùng
        if not processed_chunks:
            logger.warning(f"No data processed for split {split}. Output file will not be created.")
            continue

        with timer(f"Concatenate and write final output ({split})"):
            # Ghép nối các DataFrames đã xử lý
            df_output_final = pl.concat(processed_chunks).sort("impression_index")  # Sort lại để đảm bảo thứ tự

            df_output_final.write_parquet(output_path_final, compression="zstd", use_pyarrow=True)

            logger.info(f"Finished. Final output shape: {df_output_final.shape}. Saved to {output_path_final}")
            logger.info(df_output_final.head(5))

        # --- Kiểm tra tính nhất quán (Test consistency) ---
        with timer(f"Test consistency ({split})"):
            if not output_path_final.exists():
                logger.warning(f"Skipping consistency check for {split}: Output file not found.")
                continue

            lf_output_check = pl.scan_parquet(output_path_final)
            n_rows_output_check = lf_output_check.select(pl.len()).collect().item()

            assert n_rows_input == n_rows_output_check, (
                f"Dữ liệu {split} không khớp số dòng: Input={n_rows_input}, Output={n_rows_output_check}"
            )
            logger.info(f"Consistency check passed for {split}: Input and output row counts match.")


if __name__ == "__main__":
    APP()