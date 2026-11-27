import logging
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import *

from dataframe_loader import load_partitioned_csv
from data_schema import flight_schema
from data_validation import validate_dataframe
from statistic_analysis import analyze_numerical_features
from removing_noninformative_fields import clean_data
from data_preprocessing import preprocess
from data_modeling import create_all_tables
from business_logic_1 import run_business_questions, save_results
from business_logic_2 import run_business_questions_2, save_results_2
from business_logic_3 import run_business_questions_3, save_results_3
from data_analysis import run_data_analysis

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

    # Шлях до даних можна змінити змінною середовища DATA_PATH (за замовчуванням /data у Docker)
    data_path = os.environ.get("DATA_PATH", "/data")

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

        flight_df = preprocess(flight_df)
        logger.info("Схема DataFrame після препроцесингу:")
        flight_df.printSchema()
        flight_df.show(5, truncate=False)

        # Статистика числових ознак
        stats_df = analyze_numerical_features(flight_df)

        # ЕТАП ТРАНСФОРМАЦІЇ
        logger.info("--- Початок етапу трансформації ---")
        flights_fact, carriers_dim, airports_dim, date_dim = create_all_tables(flight_df)
        logger.info("Таблиці створено. Запускаємо бізнес-питання...")

        bq1, bq2, bq3, bq4, bq5, bq6 = run_business_questions(
            flights_fact, carriers_dim, airports_dim, date_dim
        )

        logger.info("Зберігаємо результати...")
        save_results(bq1, bq2, bq3, bq4, bq5, bq6, output_path=r"output/business_logic_1")

        # business_logic_2
        r1, r2, r3, r4, r5, r6 = run_business_questions_2(
            flights_fact, carriers_dim, airports_dim, date_dim
        )
        save_results_2(r1, r2, r3, r4, r5, r6, output_path="output/business_logic_2")

        logger.info("Всі результати збережено.")
        #business_logic_3
        r1, r2, r3, r4, r5, r6 = run_business_questions_3(
            flights_fact, carriers_dim, airports_dim, date_dim
        )

        save_results_3(r1, r2, r3, r4, r5, r6, output_path="output/business_logic_3")

        # ЕТАП АНАЛІЗУ ДАНИХ (регресія та класифікація)
        logger.info("--- Початок етапу аналізу даних ---")
        run_data_analysis(spark, flight_df, output_path="output/data_analysis")

    except Exception as e:
        logger.error(f"Сталася помилка під час виконання: {e}")

    finally:
        logger.info("Завершення роботи. Зупинка SparkSession.")
        spark.stop()


if __name__ == "__main__":
    main()
