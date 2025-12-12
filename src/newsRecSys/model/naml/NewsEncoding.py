import gc

from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
import polars as pl
import torch

def main():
    # Argument parser for command line usage
    parser = ArgumentParser(description="Encode news articles using SentenceTransformer.")
    parser.add_argument('--model_name', type=str, default='all-MiniLM-L6-v2', help='SentenceTransformer model name')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for encoding')
    parser.add_argument('--input', type=str, help='Path to input dataset')
    parser.add_argument('--category', type=str, help='Path to category dataset')
    parser.add_argument('--output', type=str, help='Path to output dataset')
    args = parser.parse_args()
    # Load the SentenceTransformer model
    model = SentenceTransformer(args.model_name)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    # Read the input dataset using Polars
    df = pl.read_parquet(args.input)
    title = df['title'].to_list()
    subtitle = df['subtitle'].to_list()
    body = df['body'].to_list()
    # Clean missing values
    title = [t if t is not None else "" for t in title]
    subtitle = [s if s is not None else "" for s in subtitle]
    body = [b if b is not None else "" for b in body]

    with torch.no_grad():
        title_vecs = model.encode(title, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True)
        subtitle_vecs = model.encode(subtitle, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True)
        body_vecs = model.encode(body, batch_size=args.batch_size, show_progress_bar=True, convert_to_tensor=True)

    article_id = df['article_id'].to_list()

    out_df = pl.DataFrame({
        'article_id': article_id,
        'title_vector': [v.tolist() for v in title_vecs],
        'subtitle_vector': [v.tolist() for v in subtitle_vecs],
        'body_vector': [v.tolist() for v in body_vecs],
    })

    out_df.write_parquet(args.output)

if __name__  == "__main__":
    main()