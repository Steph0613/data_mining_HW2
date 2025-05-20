import os
import pandas as pd
from pyarrow.parquet import ParquetFile
from tqdm import tqdm
from collections import defaultdict
from mlxtend.frequent_patterns import fpgrowth, association_rules
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt

EXPANDED_DIR = './outputs/expanded_items_chunks'
OUTPUT_CSV = './outputs/task1_category_transactions.csv'
RULES_OUTPUT = './outputs/task1_category_rules.csv'
CHUNKSIZE = 250_000
MIN_SUPPORT = 0.02
MIN_CONFIDENCE = 0.5


print("第一步：从 expanded_items 构建事务数据……")
from collections import defaultdict
user_order_to_categories = defaultdict(set)

files = sorted([f for f in os.listdir(EXPANDED_DIR) if f.endswith('.parquet')])
for fname in tqdm(files, desc="提取类别事务"):
    fpath = os.path.join(EXPANDED_DIR, fname)
    pf = ParquetFile(fpath) 
    for batch in pf.iter_batches(batch_size=CHUNKSIZE):
        df = batch.to_pandas()
        df = df.dropna(subset=['item_category', 'user_id', 'purchase_date'])
        for row in df.itertuples():
            key = (row.user_id, row.purchase_date)
            user_order_to_categories[key].add(row.item_category)

transactions = [list(cats) for cats in user_order_to_categories.values() if len(cats) > 1]
print(f"共构建事务数: {len(transactions)}")

pd.Series(transactions).to_csv(OUTPUT_CSV, index=False, header=False)
print(f"事务 CSV 已保存: {OUTPUT_CSV}")

print("第二步：运行 FP-Growth 挖掘……")
mlb = MultiLabelBinarizer()
encoded = pd.DataFrame(mlb.fit_transform(transactions), columns=mlb.classes_)
frequent_itemsets = fpgrowth(encoded, min_support=MIN_SUPPORT, use_colnames=True)
frequent_itemsets.sort_values(by='support', ascending=False, inplace=True)
print(f"找到 {len(frequent_itemsets)} 个频繁项集")

print("第三步：生成关联规则……")
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
print(f"找到 {len(rules)} 条关联规则")

focus_rules = rules[rules['antecedents'].apply(lambda x: '电子产品' in x or any('电子产品' in s for s in x)) |
                    rules['consequents'].apply(lambda x: '电子产品' in x or any('电子产品' in s for s in x))]
print(f"与 '电子产品' 有关的规则数: {len(focus_rules)}")

rules.sort_values(by="lift", ascending=False).to_csv(RULES_OUTPUT, index=False)
print(f"所有规则已保存至: {RULES_OUTPUT}")

top10 = rules.sort_values(by="lift", ascending=False).head(10)
plt.figure(figsize=(10,6))
plt.barh(range(len(top10)), top10['lift'], color='skyblue')
plt.yticks(range(len(top10)), [f"{', '.join(list(a))} → {', '.join(list(c))}" for a, c in zip(top10['antecedents'], top10['consequents'])])
plt.xlabel("Lift")
plt.title("Top 10 Lift 最高的关联规则")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("./outputs/task1_lift_top10.png")
plt.show()
