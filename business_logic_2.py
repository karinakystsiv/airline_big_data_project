import logging
import os
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/business_logic_2"


def _save(df: DataFrame, name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(path)
    logger.info(f"Збережено: {path}")


def bq1_longest_taxi_out(flights: DataFrame) -> DataFrame:
    title = "BQ1: Топ-15 аеропортів за середнім TAXI_OUT (далекі рейси, DISTANCE > 1000)"
    logger.info(title)

    result = (
        flights
        .filter(
            (F.col("CANCELLED") == 0)                              # filter
            & (F.col("DISTANCE") > 1000)                           # filter
            & (F.col("TAXI_OUT").isNotNull())
        )
        .groupBy("ORIGIN")                                         # group by
        .agg(
            F.round(F.avg("TAXI_OUT"), 2).alias("AVG_TAXI_OUT"),
            F.round(F.max("TAXI_OUT"), 2).alias("MAX_TAXI_OUT"),
            F.count("*").alias("FLIGHT_COUNT"),
        )
        .filter(F.col("FLIGHT_COUNT") >= 50)                       # filter
        .orderBy(F.desc("AVG_TAXI_OUT"))
        .limit(15)
    )

    print(f"\n{title}")
    result.show(truncate=False)
    result.explain(mode="extended")
    _save(result, "bq1_longest_taxi_out")
    return result


def bq2_diversion_rate_delayed(flights: DataFrame) -> DataFrame:
    title = "BQ2: Частка перенаправлених рейсів серед затриманих > 30 хв"
    logger.info(title)

    result = (
        flights
        .filter(F.col("DEP_DELAY") > 30)                           # filter
        .groupBy("OP_CARRIER")                                      # group by
        .agg(
            F.count("*").alias("DELAYED_FLIGHTS"),
            F.sum(F.col("DIVERTED").cast("int")).alias("DIVERTED_COUNT"),
        )
        .withColumn(
            "DIVERSION_RATE_PCT",
            F.round(F.col("DIVERTED_COUNT") / F.col("DELAYED_FLIGHTS") * 100, 2),
        )
        .filter(F.col("DELAYED_FLIGHTS") >= 100)                    # filter
        .orderBy(F.desc("DIVERSION_RATE_PCT"))
    )

    print(f"\n{title}")
    result.show(truncate=False)
    result.explain(mode="extended")
    _save(result, "bq2_diversion_rate")
    return result


def bq3_actual_vs_scheduled_time(flights: DataFrame,
                                  airports_dim: DataFrame) -> DataFrame:
    title = "BQ3: Аеропорти де фактичний час > запланованого (join airports_dim)"
    logger.info(title)

    result = (
        flights
        .filter(
            (F.col("CANCELLED") == 0)                               # filter
            & F.col("ACTUAL_ELAPSED_TIME").isNotNull()
            & F.col("CRS_ELAPSED_TIME").isNotNull()
        )
        .withColumn(
            "TIME_DIFF",
            F.col("ACTUAL_ELAPSED_TIME") - F.col("CRS_ELAPSED_TIME"),
        )
        .join(                                                       # join
            airports_dim,
            flights["ORIGIN"] == airports_dim["AIRPORT"],
            "inner",
        )
        .filter(F.col("total_flights_from") > 5000)                  # filter
        .groupBy("ORIGIN", "total_flights_from", "avg_departure_delay")  # group by
        .agg(
            F.round(F.avg("TIME_DIFF"), 2).alias("AVG_TIME_DIFF"),
            F.count("*").alias("FLIGHTS"),
        )
        .filter(F.col("AVG_TIME_DIFF") > 10)                        # filter
        .orderBy(F.desc("AVG_TIME_DIFF"))
    )

    print(f"\n{title}")
    result.show(20, truncate=False)
    result.explain(mode="extended")
    _save(result, "bq3_actual_vs_scheduled")
    return result


def bq4_delay_by_part_of_day(flights: DataFrame,
                              carriers_dim: DataFrame,
                              date_dim: DataFrame) -> DataFrame:
    title = "BQ4: Затримка прибуття за частиною доби для великих авіакомпаній (join date_dim + carriers_dim)"
    logger.info(title)

    result = (
        flights
        .join(                                                       # join 1
            date_dim.select("FL_DATE", "DEP_HOUR", "part_of_day").distinct(),
            on=["FL_DATE", "DEP_HOUR"],
            how="inner",
        )
        .join(                                                       # join 2
            carriers_dim.select("OP_CARRIER", "total_flights"),
            on="OP_CARRIER",
            how="inner",
        )
        .filter(
            (F.col("total_flights") > 50000)                         # filter
            & (F.col("ARR_DELAY").isNotNull())
        )
        .groupBy("OP_CARRIER", "part_of_day")                       # group by
        .agg(
            F.round(F.avg("ARR_DELAY"), 2).alias("AVG_ARR_DELAY"),
            F.round(F.stddev("ARR_DELAY"), 2).alias("STD_ARR_DELAY"),
            F.count("*").alias("FLIGHTS"),
        )
        .orderBy("OP_CARRIER", "part_of_day")
    )

    print(f"\n{title}")
    result.show(30, truncate=False)
    result.explain(mode="extended")
    _save(result, "bq4_delay_by_part_of_day")
    return result


def bq5_mom_flight_volume(flights: DataFrame) -> DataFrame:
    title = "BQ5: Зміна кількості рейсів місяць-до-місяця за аеропортами (window lag)"
    logger.info(title)

    monthly = (
        flights
        .filter(F.col("CANCELLED") == 0)                            # filter
        .groupBy("ORIGIN", "YEAR", "MONTH")
        .agg(F.count("*").alias("FLIGHTS"))
    )

    w = Window.partitionBy("ORIGIN").orderBy("YEAR", "MONTH")

    result = (
        monthly
        .withColumn("PREV_FLIGHTS", F.lag("FLIGHTS", 1).over(w))    # window (lag)
        .filter(F.col("PREV_FLIGHTS").isNotNull())                   # filter
        .withColumn(
            "MOM_CHANGE",
            F.col("FLIGHTS") - F.col("PREV_FLIGHTS"),
        )
        .withColumn(
            "MOM_CHANGE_PCT",
            F.round(F.col("MOM_CHANGE") / F.col("PREV_FLIGHTS") * 100, 2),
        )
        .orderBy("ORIGIN", "YEAR", "MONTH")
    )

    print(f"\n{title}")
    result.show(30, truncate=False)
    result.explain(mode="extended")
    _save(result, "bq5_mom_flight_volume")
    return result

def bq6_route_delay_percentile(flights: DataFrame) -> DataFrame:
    title = "BQ6: Процентильний рейтинг маршрутів за затримкою (window percent_rank)"
    logger.info(title)

    route_stats = (
        flights
        .filter(
            (F.col("CANCELLED") == 0)                                # filter
            & (F.col("ARR_DELAY").isNotNull())
        )
        .groupBy("ORIGIN", "DEST")                                  # group by
        .agg(
            F.round(F.avg("ARR_DELAY"), 2).alias("AVG_ARR_DELAY"),
            F.count("*").alias("FLIGHT_COUNT"),
        )
        .filter(F.col("FLIGHT_COUNT") >= 50)                        # filter
    )

    w = Window.orderBy(F.desc("AVG_ARR_DELAY"))

    result = (
        route_stats
        .withColumn(
            "DELAY_PERCENTILE",                                      # window (percent_rank)
            F.round(F.percent_rank().over(w) * 100, 2),
        )
        .filter(F.col("DELAY_PERCENTILE") <= 5)                    
        .orderBy("DELAY_PERCENTILE")
    )

    print(f"\n{title}")
    result.show(30, truncate=False)
    result.explain(mode="extended")
    _save(result, "bq6_route_percentile")
    return result


def run_business_questions_2(flights_fact, carriers_dim, airports_dim, date_dim):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    r1 = bq1_longest_taxi_out(flights_fact)
    r2 = bq2_diversion_rate_delayed(flights_fact)
    r3 = bq3_actual_vs_scheduled_time(flights_fact, airports_dim)
    r4 = bq4_delay_by_part_of_day(flights_fact, carriers_dim, date_dim)
    r5 = bq5_mom_flight_volume(flights_fact)
    r6 = bq6_route_delay_percentile(flights_fact)

    logger.info("=== business_logic_2: усі 6 питань оброблено ===")
    return r1, r2, r3, r4, r5, r6


def save_results_2(r1, r2, r3, r4, r5, r6, output_path="output/business_logic_2"):
    for i, df in enumerate([r1, r2, r3, r4, r5, r6], start=1):
        df.coalesce(1).write.mode("overwrite").option("header", "true") \
            .csv(f"{output_path}/bq{i}")
    print(f"Results saved to {output_path}/")