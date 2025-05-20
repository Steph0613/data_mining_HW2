import os
import json
import pandas as pd
from pyarrow.parquet import ParquetFile
from tqdm import tqdm
import ast

INPUT_DIR = './outputs/cleaned_chunks'  
OUTPUT_DIR = './outputs/expanded_items_chunks'  
PRODUCT_CATALOG_PATH = './product_catalog.json'
CHUNKSIZE = 5_000_000
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(PRODUCT_CATALOG_PATH, 'r', encoding='utf-8') as f:
    raw_catalog = json.load(f)
    product_catalog = {
        str(p['id']): {
            'category': p['category'],
            'price': p['price']
        } for p in raw_catalog['products']
    }

def extract_items(row):
    try:
        items = row['purchase_item_ids']
        if isinstance(items, str):
            items = ast.literal_eval(items)
        if not isinstance(items, list):
            if hasattr(items, '__iter__'):
                items = list(items)
            else:
                return []
        if not items:
            return []

        result = []
        for item in items:
            item_id = str(item.get('id'))
            info = product_catalog.get(item_id, {})
            result.append({
                'user_id': row['id'],
                'purchase_date': row.get('purchase_date'),
                'item_id': item_id,
                'item_category': info.get('category', '未知'),
                'item_price': info.get('price', -1),
                'payment_method': row.get('payment_method'),
                'payment_status': row.get('payment_status'),
                'is_high_value': info.get('price', 0) > 5000
            })
        return result
    except Exception:
        return []

preview_checked = False
batch_counter = 0

parquet_files = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith('.parquet')])

for filename in tqdm(parquet_files, desc='展开 items 文件级处理'):
    file_path = os.path.join(INPUT_DIR, filename)
    pf = ParquetFile(file_path)
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=CHUNKSIZE)):
        print(f"正在处理文件: {filename}, 分块批次: {batch_idx}")
        df = batch.to_pandas()
        required_cols = ['id', 'purchase_item_ids', 'purchase_date', 'payment_method', 'payment_status']
        df = df[[col for col in required_cols if col in df.columns]]
        exploded = df.apply(extract_items, axis=1).explode().dropna()

        if not preview_checked:
            preview_checked = True
            if exploded.empty:
                raise ValueError(f"数据展开失败：首批数据无有效商品，请检查清洗后的字段格式！\n示例行：\n{df.head(3)}")
            else:
                print("首批数据通过，继续处理...")

        if not exploded.empty:
            normalized = pd.json_normalize(exploded)
            output_path = os.path.join(OUTPUT_DIR, f"expanded_items_batch_{batch_counter}.parquet")
            normalized.to_parquet(output_path, index=False)
            print(f"已保存批次 {batch_counter}，记录数: {len(normalized)} → {output_path}")
            batch_counter += 1
        else:
            print(f"跳过空批次: {filename}, 分块 {batch_idx}")

print(f"全部处理完成，共生成 {batch_counter} 个 expanded_items 批次文件，存储于: {OUTPUT_DIR}")