import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from dataframe_loader import load_partitioned_csv
from data_schema import flight_schema
from data_validation import validate_dataframe

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    logger.info("Створення SparkSession...")

    spark = SparkSession.builder \
        .appName("Airline_Project") \
        .master("local[*]") \
        .getOrCreate()

    logger.info("SparkSession успішно створено.")

    data_path = r"/data"

    try:
        flight_df = load_partitioned_csv(
            spark=spark,
            data_path=data_path,
            schema=flight_schema,
            recursive=True
        )

        logger.info("Перегляд перших 5 рядків DataFrame:")
        flight_df.show(5)

        is_valid = validate_dataframe(flight_df, flight_schema)
        if not is_valid:
            raise ValueError("Дані не пройшли валідацію. Зупинка пайплайну.")

        logger.info("Дані готові до попередньої обробки (Preprocessing).")

    except Exception as e:
        logger.error(f"Сталася помилка під час виконання: {e}")

    finally:
        logger.info("Завершення роботи. Зупинка SparkSession.")
        spark.stop()


if __name__ == "__main__":
    main()
