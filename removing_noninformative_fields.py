import logging
from pyspark.sql import DataFrame
from pyspark.sql.functions import col

logger = logging.getLogger(__name__)


def clean_data(df: DataFrame) -> DataFrame:
    logger.info("=== Етап попередньої обробки: Очищення та оптимізація типів ===")

    # Видалення технічної колонки
    df = df.drop("Unnamed: 27")
    logger.info("Видалено технічну колонку Unnamed: 27.")

    # Приведення типів

    cols_to_cast = [
        "OP_CARRIER_FL_NUM", "DEP_TIME", "DEP_DELAY", "TAXI_OUT",
        "WHEELS_OFF", "WHEELS_ON", "TAXI_IN", "ARR_TIME",
        "ARR_DELAY", "CANCELLED", "DIVERTED", "ACTUAL_ELAPSED_TIME",
        "AIR_TIME", "DISTANCE"
    ]

    for col_name in cols_to_cast:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("integer"))

    logger.info(f"Оптимізовано типи для {len(cols_to_cast)} колонок (Double -> Integer).")

    return df