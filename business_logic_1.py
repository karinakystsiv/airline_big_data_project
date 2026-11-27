from pyspark.sql import functions as F
from pyspark.sql.window import Window

def run_business_questions(flights_fact, carriers_dim, airports_dim, date_dim):
    # BQ1: Які 10 авіакомпаній мають найбільшу частку скасованих рейсів
    #      серед тих, що виконали понад 10 000 рейсів?
    bq1 = (
        flights_fact
        .groupBy("OP_CARRIER")
        .agg(
            F.count("*").alias("total_flights"),
            F.sum(F.col("CANCELLED").cast("int")).alias("cancelled_flights")
        )
        .filter(F.col("total_flights") > 10_000)
        .withColumn(
            "cancellation_rate",
            F.round(F.col("cancelled_flights") / F.col("total_flights") * 100, 2)
        )
        .orderBy(F.col("cancellation_rate").desc())
        .limit(10)
    )

    print("BQ1: Top-10 авіакомпаній за часткою скасувань (мін. 10 000 рейсів)")
    bq1.show()
    bq1.explain()

    # BQ2: Який середній час затримки вильоту для кожної пари (авіакомпанія, пора року)
    bq2 = (
        flights_fact
        .join(
            date_dim.select("FL_DATE", "season").distinct(),
            on="FL_DATE",
            how="inner"
        )
        .filter(F.col("DEP_DELAY") > 0)
        .groupBy("OP_CARRIER", "season")
        .agg(
            F.avg("DEP_DELAY").alias("avg_delay_min"),
            F.count("*").alias("delayed_flights")
        )
        .orderBy("OP_CARRIER", "season")
    )

    print("BQ2: Середня затримка вильоту за авіакомпанією та порою року")
    bq2.show()
    bq2.explain()

    # BQ3: Для кожного аеропорту вильоту — яка частина рейсів вилітає вчасно?
    bq3 = (
        flights_fact
        .groupBy("ORIGIN")
        .agg(
            F.count("*").alias("total"),
            F.sum(
                F.when(F.col("DEP_DELAY") <= 0, 1).otherwise(0)
            ).alias("on_time_count")
        )
        .withColumn(
            "on_time_rate_pct",
            F.round(F.col("on_time_count") / F.col("total") * 100, 2)
        )
        .join(airports_dim, on="ORIGIN" == airports_dim["AIRPORT"], how="left")
        .select(
            "ORIGIN",
            "total",
            "on_time_count",
            "on_time_rate_pct",
            "avg_departure_delay"
        )
        .orderBy(F.col("on_time_rate_pct").desc())
    )

    print("BQ3: Частка рейсів, що вилітають вчасно, по аеропортах")
    bq3.show()
    bq3.explain()

    # BQ4: Як змінюється середня затримка прибуття місяць за місяцем для кожної авіакомпанії — ковзне середнє за 3 місяці?
    monthly_avg = (
        flights_fact
        .join(
            date_dim.select("FL_DATE", "YEAR", "MONTH").distinct()
            .withColumnRenamed("YEAR", "d_YEAR")
            .withColumnRenamed("MONTH", "d_MONTH"),
            on="FL_DATE",
            how="inner"
        )
        .filter(F.col("ARR_DELAY").isNotNull())
        .groupBy("OP_CARRIER", "d_YEAR", "d_MONTH")
        .agg(F.avg("ARR_DELAY").alias("avg_arr_delay"))
        .withColumnRenamed("d_YEAR", "YEAR")
        .withColumnRenamed("d_MONTH", "MONTH")
    )

    window_3m = (
        Window
        .partitionBy("OP_CARRIER")
        .orderBy("YEAR", "MONTH")
        .rowsBetween(-2, 0)
    )

    bq4 = (
        monthly_avg
        .withColumn(
            "rolling_avg_3m",
            F.round(F.avg("avg_arr_delay").over(window_3m), 2)
        )
        .orderBy("OP_CARRIER", "YEAR", "MONTH")
    )

    print("BQ4: Ковзне середнє (3 міс.) затримки прибуття по авіакомпаніях")
    bq4.show()
    bq4.explain()

    # BQ5: Які 5 маршрутів у кожної авіакомпанії мають найбільшу середню затримку?

    route_delay = (
        flights_fact
        .filter(F.col("DEP_DELAY").isNotNull())
        .groupBy("OP_CARRIER", "ORIGIN", "DEST")
        .agg(
            F.avg("DEP_DELAY").alias("avg_dep_delay"),
            F.count("*").alias("flight_count")
        )
        .filter(F.col("flight_count") >= 100)
    )

    window_rank = (
        Window
        .partitionBy("OP_CARRIER")
        .orderBy(F.col("avg_dep_delay").desc())
    )

    bq5 = (
        route_delay
        .withColumn("rank", F.rank().over(window_rank))
        .filter(F.col("rank") <= 5)
        .orderBy("OP_CARRIER", "rank")
    )

    print("BQ5: Топ-5 найзатриманіших маршрутів для кожної авіакомпанії")
    bq5.show()
    bq5.explain()

    # BQ6: Яка частка рейсів у вихідні vs будні за кожною авіакомпанією, і чи відрізняється середня затримка між ними?
    bq6 = (
        flights_fact
        .join(
            date_dim.select("FL_DATE", "is_weekend").distinct(),
            on="FL_DATE",
            how="inner"
        )
        .join(
            carriers_dim.select("OP_CARRIER", "total_flights"),
            on="OP_CARRIER",
            how="inner"
        )
        .filter(F.col("DEP_DELAY").isNotNull())
        .groupBy("OP_CARRIER", "is_weekend")
        .agg(
            F.count("*").alias("flights_count"),
            F.avg("DEP_DELAY").alias("avg_dep_delay")
        )
        .withColumn(
            "day_type",
            F.when(F.col("is_weekend") == 1, "Weekend").otherwise("Weekday")
        )
        .select("OP_CARRIER", "day_type", "flights_count", "avg_dep_delay")
        .orderBy("OP_CARRIER", "day_type")
    )

    print("BQ6: Затримки у вихідні vs будні по авіакомпаніях")
    bq6.show()
    bq6.explain()

    return bq1, bq2, bq3, bq4, bq5, bq6


def save_results(bq1, bq2, bq3, bq4, bq5, bq6, output_path="output"):
    bq1.write.mode("overwrite").csv(f"{output_path}/bq1_cancellation_rate", header=True)
    bq2.write.mode("overwrite").csv(f"{output_path}/bq2_delay_by_season", header=True)
    bq3.write.mode("overwrite").csv(f"{output_path}/bq3_on_time_rate", header=True)
    bq4.write.mode("overwrite").csv(f"{output_path}/bq4_rolling_avg_delay", header=True)
    bq5.write.mode("overwrite").csv(f"{output_path}/bq5_worst_routes", header=True)
    bq6.write.mode("overwrite").csv(f"{output_path}/bq6_weekend_vs_weekday", header=True)
    print(f"Results saved to {output_path}/")
