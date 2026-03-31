import logging
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from dataframe_loader import load_partitioned_csv
from data_schema import flight_schema
from data_validation import validate_dataframe
from statistic_analysis import analyze_numerical_features
from removing_noninformative_fields import clean_data

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
        logger.info("Схема завантаженого DataFrame:")
        flight_df.printSchema()

        logger.info("Перегляд перших 5 рядків DataFrame:")
        flight_df.show(5)

        # Коментар Лізи: я взяла тільки 1% даних, щоб побачити чи все працює
        flight_df = flight_df.sample(fraction=0.01, seed=42)
        logger.info(f"Для тестування взято {flight_df.count()} рядків.")

        is_valid = validate_dataframe(flight_df, flight_schema)
        if not is_valid:
            raise ValueError("Дані не пройшли валідацію. Зупинка пайплайну.")

        logger.info("Дані готові до попередньої обробки (Preprocessing).")

        # ЕТАП ПОПЕРЕДНЬОЇ ОБРОБКИ
        logger.info("--- Початок попередньої обробки ---")

        flight_df = clean_data(flight_df)
        logger.info("Схема DataFrame після оптимізації типів:")
        flight_df.printSchema()

        # Статистика числових ознак
        stats_df = analyze_numerical_features(flight_df)

    except Exception as e:
        logger.error(f"Сталася помилка під час виконання: {e}")

    finally:
        logger.info("Завершення роботи. Зупинка SparkSession.")
        spark.stop()


if __name__ == "__main__":
    main()
