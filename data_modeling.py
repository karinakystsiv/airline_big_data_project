from pyspark.sql import functions as F

def create_all_tables(df):

    flights_fact = df


    carriers_dim = df.groupBy("OP_CARRIER").agg(
        F.count("*").alias("total_flights"),
        F.avg("DEP_DELAY").alias("avg_dep_delay")
    )


    airport_stats = df.groupBy("ORIGIN").agg(
        F.count("*").alias("total_flights_from"),
        F.avg("DEP_DELAY").alias("avg_departure_delay")
    )


    airports_list = df.select(F.col("ORIGIN").alias("AIRPORT")).union(
        df.select(F.col("DEST").alias("AIRPORT"))
    ).distinct()


    airports_dim = airports_list.join(
        airport_stats, airports_list.AIRPORT == airport_stats.ORIGIN, "left"
    ).select(
        F.col("AIRPORT"),
        F.coalesce(F.col("total_flights_from"), F.lit(0)).alias("total_flights_from"),
        F.coalesce(F.col("avg_departure_delay"), F.lit(0.0)).alias("avg_departure_delay")
    )


    date_dim = df.select(
        "FL_DATE", "YEAR", "MONTH", "DAY_OF_WEEK", "DEP_HOUR"
    ).distinct().withColumn(
        "is_weekend",
        F.when(F.col("DAY_OF_WEEK").isin([1, 7]), 1).otherwise(0)
    ).withColumn(
        "part_of_day",
        F.when(F.col("DEP_HOUR") < 6, "Night")
        .when(F.col("DEP_HOUR") < 12, "Morning")
        .when(F.col("DEP_HOUR") < 18, "Afternoon")
        .otherwise("Evening")
    ).withColumn(
        "season",
        F.when(F.col("MONTH").isin([12, 1, 2]), "Winter")
        .when(F.col("MONTH").isin([3, 4, 5]), "Spring")
        .when(F.col("MONTH").isin([6, 7, 8]), "Summer")
        .otherwise("Autumn")
    )

    return flights_fact, carriers_dim, airports_dim, date_dim