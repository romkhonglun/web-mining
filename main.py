import polars as pl
from argparse import ArgumentParser
def main():
    argparse = ArgumentParser(description="Process some integers.")
    argparse.add_argument("--input", type=str, required=True, help="Input Parquet file path")
    args = argparse.parse_args()
    # Read the Parquet file into a Polars DataFrame
    df = pl.read_parquet(args.input)
    print(df.columns)
    import os
    def _hr(n):
        for unit in ["B", "KB", "MB", "GB", "TB"]:
            if n < 1024:
                return f"{n:.2f} {unit}"
            n /= 1024
        return f"{n:.2f} PB"
    size = os.path.getsize(args.input)
    print("File size:", _hr(size))

if __name__ == "__main__":
    main()