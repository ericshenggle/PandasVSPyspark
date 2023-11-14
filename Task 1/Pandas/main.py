import pandas as pd
import datetime


def log_time(task_name):
    print(f"{task_name} completed at {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


# Step 1: 读取数据
log_time("Start")
input_file_path = '../Datasets/Patio_Lawn_and_Garden.json'
metadata_file_path = '../Datasets/meta_Patio_Lawn_and_Garden.json'

reviews_df = pd.read_json(input_file_path, lines=True)
log_time("Reviews data loaded")

metadata_df = pd.read_json(metadata_file_path, lines=True)
log_time("Metadata data loaded")

# Step 2: 数据处理
reviews_grouped = reviews_df.groupby(['asin', 'reviewTime']).agg(
    num_reviews=pd.NamedAgg(column='overall', aggfunc='count'),
    avg_rating=pd.NamedAgg(column='overall', aggfunc='mean')
).reset_index()
log_time("Reviews data grouped and aggregated")

metadata_df = metadata_df[['asin', 'brand']]
log_time("Metadata data transformed")

joined_df = pd.merge(reviews_grouped, metadata_df, on='asin')
log_time("DataFrames joined")

# Step 3: 选择和排序数据
top_15_products = joined_df.sort_values(by='num_reviews', ascending=False).head(15)
log_time("Top 15 products determined")

# Step 4: 输出结果
output_file_path = 'output_pandas.txt'
with open(output_file_path, "w") as output_file:
    # 写入表头
    field_widths = [15, 20, 15, 20, 20]
    headers = ["<product ID>", "<num of reviews>", "<review time>", "<avg ratings>", "<product brand name>"]
    headers = "\t".join(header.rjust(width) for header, width in zip(headers, field_widths))
    output_file.write(headers + "\n")

    for index, row in top_15_products.iterrows():
        _data = [row['asin'], row['num_reviews'], row['reviewTime'], round(row['avg_rating'], 4), row['brand']]
        output_line = "\t".join(str(field).rjust(width)
                                if isinstance(field, (int, float)) else field.rjust(width)
                                for field, width in zip(_data, field_widths))
        output_file.write(output_line + "\n")
    log_time("Output file written")

log_time("Process completed")
