from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType
import os
import logging

logger = logging.getLogger(__name__)

def load_partitioned_csv(
    spark: SparkSession,
    data_path: str,
    schema: StructType,
    recursive: bool = True,
) -> DataFrame:

    logger.info(f"Завантаження даних з: {data_path}")

    if "*" not in data_path and not os.path.exists(data_path):
        raise ValueError(f"Шлях не існує: {data_path}")


    reader = (
        spark.read
        .option("header", "true")
        .option("recursiveFileLookup", str(recursive).lower())
        .option("pathGlobFilter", "*.csv")
        .schema(schema)
    )

    df = reader.csv(data_path)

    row_count = df.count()
    logger.info(f"DataFrame успішно створено. Рядків: {row_count}, Колонок: {len(df.columns)}")

    return df

