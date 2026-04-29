import logging
import os
import shutil
import glob

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/business_logic_3"


# =========================================================
# SAVE helper -> ОДИН CSV файл на результат
# =========================================================
def save_single_csv(df: DataFrame, path: str):
    tmp_path = path + "_tmp"

    df.coalesce(1).write.mode("overwrite") \
        .option("header", "true") \
        .csv(tmp_path)

    part_file = glob.glob(f"{tmp_path}/part-*")[0]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    shutil.move(part_file, path)

    shutil.rmtree(tmp_path)


# =========================================================
# BQ7
# =========================================================
def bq7_airport_season_reliability(flights: DataFrame, date_dim: DataFrame):

    date_clean = date_dim.select("FL_DATE", "season")

    df = (
        flights
        .join(date_clean, "FL_DATE", "inner")
        .groupBy("ORIGIN", "season")
        .agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("DEP_DELAY") <= 0, 1).otherwise(0)).alias("on_time")
        )
        .withColumn("on_time_rate", F.col("on_time") / F.col("total"))
        .filter(F.col("total") >= 200)
    )

    w = Window.partitionBy("season").orderBy(F.desc("on_time_rate"))

    result = df.withColumn("rank", F.dense_rank().over(w))

    result.show(5, truncate=False)
    return result


# =========================================================
# BQ8 (FIXED YEAR AMBIGUITY)
# =========================================================
def bq8_carrier_trend(flights: DataFrame, date_dim: DataFrame):

    date_clean = date_dim.select(
        "FL_DATE",
        F.col("YEAR").alias("d_year"),
        F.col("MONTH").alias("d_month")
    )

    monthly = (
        flights
        .join(date_clean, "FL_DATE", "inner")
        .groupBy("OP_CARRIER", "d_year", "d_month")
        .agg(F.avg("ARR_DELAY").alias("avg_delay"))
    )

    w = Window.partitionBy("OP_CARRIER").orderBy("d_year", "d_month")

    result = (
        monthly
        .withColumn("prev", F.lag("avg_delay").over(w))
        .filter(F.col("prev").isNotNull())
        .withColumn("change", F.col("avg_delay") - F.col("prev"))
    )

    result.show(5, truncate=False)
    return result


# =========================================================
# BQ9
# =========================================================
def bq9_airport_variability(flights: DataFrame):

    result = (
        flights
        .filter(F.col("ARR_DELAY").isNotNull())
        .groupBy("ORIGIN")
        .agg(
            F.stddev("ARR_DELAY").alias("std_delay"),
            F.count("*").alias("flights")
        )
        .filter(F.col("flights") >= 100)
        .orderBy(F.desc("std_delay"))
        .limit(15)
    )

    result.show(15, truncate=False)
    return result


# =========================================================
# BQ10
# =========================================================
def bq10_route_stability(flights: DataFrame, airports_dim: DataFrame):

    df = (
        flights
        .join(airports_dim, flights.ORIGIN == airports_dim.AIRPORT, "inner")
        .filter(F.col("ARR_DELAY").isNotNull())
        .groupBy("OP_CARRIER", "ORIGIN", "DEST")
        .agg(
            F.avg("ARR_DELAY").alias("avg_delay"),
            F.stddev("ARR_DELAY").alias("std_delay"),
            F.count("*").alias("cnt")
        )
        .filter(F.col("cnt") >= 80)
    )

    w = Window.partitionBy("OP_CARRIER").orderBy("std_delay")

    result = df.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= 5)

    result.show(10, truncate=False)
    return result


# =========================================================
# BQ11
# =========================================================
def bq11_early_arrivals(flights: DataFrame):

    df = (
        flights
        .filter(F.col("ARR_DELAY").isNotNull())
        .groupBy("DEST")
        .agg(
            F.count("*").alias("total"),
            F.sum(F.when(F.col("ARR_DELAY") < 0, 1).otherwise(0)).alias("early")
        )
        .filter(F.col("total") >= 150)
        .withColumn("early_rate", F.col("early") / F.col("total"))
    )

    w = Window.orderBy(F.desc("early_rate"))

    result = df.withColumn("percent_rank", F.percent_rank().over(w))

    result.show(10, truncate=False)
    return result


# =========================================================
# BQ12
# =========================================================
def bq12_distance_effect(flights: DataFrame):

    df = (
        flights
        .filter(F.col("ARR_DELAY").isNotNull())
        .withColumn(
            "distance_bin",
            F.when(F.col("DISTANCE") < 500, "short")
             .when(F.col("DISTANCE") < 1500, "medium")
             .otherwise("long")
        )
        .groupBy("distance_bin")
        .agg(
            F.avg("ARR_DELAY").alias("avg_delay"),
            F.count("*").alias("flights")
        )
    )

    w = Window.orderBy("distance_bin")

    result = df.withColumn("trend_rank", F.dense_rank().over(w))

    result.show(truncate=False)
    return result


# =========================================================
# RUN ALL
# =========================================================
def run_business_questions_3(flights_fact, carriers_dim, airports_dim, date_dim):

    r1 = bq7_airport_season_reliability(flights_fact, date_dim)
    r2 = bq8_carrier_trend(flights_fact, date_dim)
    r3 = bq9_airport_variability(flights_fact)
    r4 = bq10_route_stability(flights_fact, airports_dim)
    r5 = bq11_early_arrivals(flights_fact)
    r6 = bq12_distance_effect(flights_fact)

    return r1, r2, r3, r4, r5, r6


# =========================================================
# SAVE ALL (FINAL FIX)
# =========================================================
def save_results_3(r1, r2, r3, r4, r5, r6, output_path="output/business_logic_3"):

    os.makedirs(output_path, exist_ok=True)

    results = [r1, r2, r3, r4, r5, r6]

    for i, df in enumerate(results, start=1):
        print(f"\n=== ТЕХНІЧНИЙ ПЛАН ВИКОНАННЯ (EXTERNAL) ДЛЯ BQ{i + 6} ===")
        df.explain(mode="extended")
        final_path = f"{output_path}/bq{i}.csv"
        save_single_csv(df, final_path)

    print(f"All results saved as SINGLE CSV files in {output_path}")

