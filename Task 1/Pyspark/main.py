from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder \
    .appName("Product Review Analysis") \
    .config('spark.driver.host', '127.0.0.1') \
    .master("local") \
    .getOrCreate()

sc = spark.sparkContext

# Step 1: Read input arguments
input_file_path = '../Datasets/Patio_Lawn_and_Garden.json'          # Patio_Lawn_and_Garden.json
metadata_file_path = '../Datasets/meta_Patio_Lawn_and_Garden.json'       # meta_Patio_Lawn_and_Garden.json

reviews_df = spark.read.json(sc.textFile(input_file_path), allowBackslashEscapingAnyCharacter=True)

metadata_df = spark.read.json(sc.textFile(metadata_file_path), allowBackslashEscapingAnyCharacter=True)

reviews_grouped = reviews_df.groupBy("asin", "reviewTime") \
    .agg(F.count("overall").alias("num_reviews"),
         F.avg("overall").alias("avg_rating"))

# 转换metadata_df
metadata_df = metadata_df.select("asin", "brand")

# 连接reviews_df和metadata_df
joined_df = reviews_grouped.join(metadata_df, "asin")

# 选择和排序数据
top_15_products = joined_df.orderBy(F.desc("num_reviews")).limit(15)

# 输出结果
output_file_path = 'output_pyspark.txt'
with open(output_file_path, "w") as output_file:
    # 写入表头
    field_widths = [15, 20, 15, 20, 20]
    headers = ["<product ID>", "<num of reviews>", "<review time>", "<avg ratings>", "<product brand name>"]
    headers = "\t".join(header.rjust(width) for header, width in zip(headers, field_widths))
    output_file.write(headers + "\n")

    # 写入数据
    for row in top_15_products.collect():
        _data = [row["asin"], row["num_reviews"], row["reviewTime"], round(row["avg_rating"], 4), row["brand"]]
        output_line = "\t".join(str(field).rjust(width)
                                if isinstance(field, (int, float)) else field.rjust(width)
                                for field, width in zip(_data, field_widths))
        output_file.write(output_line + "\n")

# 停止Spark
spark.stop()