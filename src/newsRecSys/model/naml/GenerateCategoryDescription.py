import argparse
import torch
import polars as pl
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import sys


def setup_args():
    parser = argparse.ArgumentParser(description="Generate Category Description & Save by Article ID")

    # Input/Output
    parser.add_argument("--input", type=str, required=True, help="Path to input parquet file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output parquet file")

    # Column mapping
    parser.add_argument("--id-col", type=str, default="article_id",
                        help="Column name for Article ID (default: article_id)")
    parser.add_argument("--cat-col", type=str, default="category_str",
                        help="Column name for Category (default: category_str)")

    # Model config
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Model ID")

    return parser.parse_args()


def load_model(model_id):
    print(f"Loading model: {model_id}...")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto"
        )
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)


def generate_category_desc(model, tokenizer, device, category):
    if category is None or str(category).strip() == "":
        return ""

    # Prompt tiếng Đan Mạch (Danish)
    system_prompt = "Du er en AI-assistent for en dansk nyhedsplatform."

    # Yêu cầu mô tả category chung
    user_prompt = (
        f"Skriv en kort beskrivelse (1-2 sætninger) af nyhedskategorien '{category}'. "
        "Forklar hvad denne sektion typisk indeholder. "
        "Svar kun på dansk."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=60,
            temperature=0.6,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
    response = tokenizer.batch_decode(response_ids, skip_special_tokens=True)[0]
    return response.strip()


def main():
    args = setup_args()

    # 1. Đọc dữ liệu
    print(f"Reading {args.input}...")
    try:
        # Chỉ đọc các cột cần thiết để tiết kiệm RAM, sau đó join lại nếu cần
        # Hoặc đọc hết nếu file không quá lớn
        df = pl.read_parquet(args.input)
    except Exception as e:
        print(f"Error reading input file: {e}")
        return

    # Kiểm tra cột tồn tại
    if args.id_col not in df.columns:
        raise ValueError(f"Column '{args.id_col}' not found.")
    if args.cat_col not in df.columns:
        raise ValueError(f"Column '{args.cat_col}' not found.")

    # 2. Lấy danh sách Category Unique (Để tránh generate lặp lại cho hàng triệu article)
    print(f"Extracting unique categories from '{args.cat_col}'...")
    unique_cats_df = df.select(args.cat_col).unique().drop_nulls()
    cat_list = unique_cats_df[args.cat_col].to_list()

    print(f"Total articles: {len(df)}")
    print(f"Total unique categories to generate: {len(cat_list)}")

    # 3. Load Model
    model, tokenizer, device = load_model(args.model)

    # 4. Generate Description Loop
    desc_map = {}
    print("Generating descriptions...")
    for cat in tqdm(cat_list):
        try:
            desc = generate_category_desc(model, tokenizer, device, cat)
            desc_map[cat] = desc
        except Exception as e:
            print(f"Error at '{cat}': {e}")
            desc_map[cat] = ""

    # 5. Tạo DataFrame chứa mapping và Join lại vào DataFrame gốc
    print("Mapping descriptions back to Article IDs...")

    desc_df = pl.DataFrame({
        args.cat_col: list(desc_map.keys()),
        "category_description": list(desc_map.values())
    })

    # Join left để đảm bảo giữ nguyên số lượng article ban đầu
    final_df = df.join(desc_df, on=args.cat_col, how="left")

    # 6. Sắp xếp theo article_id (để đúng yêu cầu "lưu theo article_id")
    print(f"Sorting by {args.id_col}...")
    final_df = final_df.sort(args.id_col)

    # 7. Lưu file
    # Bạn có thể chọn chỉ lưu article_id và description nếu file quá nặng
    # output_df = final_df.select([args.id_col, args.cat_col, "category_description"])

    print(f"Saving {len(final_df)} articles to {args.output}...")
    final_df.write_parquet(args.output)
    print("Done!")


if __name__ == "__main__":
    main()