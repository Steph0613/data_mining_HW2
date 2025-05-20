import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

quarterly_df = pd.read_csv("task3_quarterly_category_counts.csv")
weekday_df = pd.read_csv("task3_weekday_category_counts.csv")
sequential_df = pd.read_csv("task3_sequential_category_pairs.csv")

# 年度/季度趋势图
quarter_map = {"1": "Q1", "2": "Q2", "3": "Q3", "4": "Q4"}
quarterly_df = quarterly_df.rename(columns={"Unnamed: 0": "category"})
quarterly_df.columns = ["category"] + [f"2020-{quarter_map[str(c)]}" for c in quarterly_df.columns[1:]]

plt.figure(figsize=(10, 6))
for _, row in quarterly_df.iterrows():
    plt.plot(row.index[1:], row[1:], marker='o', label=row['category'])
plt.title("季度消费趋势（按类别）")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

# 每周消费分布柱状图
weekday_df = weekday_df.rename(columns={"Unnamed: 0": "category"})
weekday_df.columns = ["category", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday_sum = weekday_df.sum()
weekday_sum = weekday_sum.drop("category")

plt.figure(figsize=(8, 5))
sns.barplot(x=weekday_sum.index, y=weekday_sum.values, color='skyblue')
plt.title("周内购买分布")
plt.ylabel("购买量")
plt.tight_layout()
plt.show()

# 季度热力图 
melted = pd.melt(quarterly_df, id_vars="category", var_name="quarter", value_name="count")
pivot = melted.pivot(index="category", columns="quarter", values="count")

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5)
plt.title("各商品类别季度购买热力图")
plt.tight_layout()
plt.show()


# Top5 季度增长类别条形图
q1 = quarterly_df.columns[1]
q4 = quarterly_df.columns[-1]
quarterly_df["growth"] = (quarterly_df[q4] - quarterly_df[q1]) / quarterly_df[q1]
top5 = quarterly_df[["category", "growth"]].sort_values("growth", ascending=False).head(5)

plt.figure(figsize=(8, 5))
sns.barplot(data=top5, y="category", x="growth", palette="Blues_d")
plt.title("Top5 季度增长品类")
plt.xlabel("平均季度增长率")
plt.tight_layout()
plt.show()

# 顺序购物模式
sequential_df_sorted = sequential_df.sort_values(by="count", ascending=False).head(8)
sequential_df_sorted["pair"] = sequential_df_sorted["from_category"] + " → " + sequential_df_sorted["to_category"]

plt.figure(figsize=(10, 6))
sns.barplot(data=sequential_df_sorted, y="pair", x="count", palette="PuBuGn_d")
plt.title("Top 8 顺序购物模式")
plt.xlabel("出现次数")
plt.tight_layout()
plt.show()
