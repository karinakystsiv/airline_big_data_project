
import logging
import os
import shutil
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output/business_logic_3"


def save_single_csv(df: DataFrame, path: str):
    tmp_path = path + "_tmp"
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(tmp_path)
    part_files = glob.glob(f"{tmp_path}/part-*")
    if part_files:
        part_file = part_files[0]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        shutil.move(part_file, path)
    shutil.rmtree(tmp_path)


def create_visualizations(df: DataFrame, bq_name: str, title: str):
    """Створює та зберігає графік у синіх тонах для кожного BQ"""
    try:
        # Конвертуємо в Pandas для візуалізації (тільки невеликі результати!)
        pdf = df.limit(15).toPandas()
        if pdf.empty:
            return

        plt.figure(figsize=(12, 6))
        sns.set_theme(style="whitegrid")

        # Налаштування синьої палітри
        blue_palette = sns.color_palette("Blues_r", len(pdf))

        if bq_name == "bq7":
            sns.barplot(data=pdf, x="distance_category", y="cancellation_rate_pct", palette="Blues_d")
        elif bq_name == "bq8":
            sns.lineplot(data=pdf, x="d_month", y="recovery_diff", hue="OP_CARRIER", marker="o", palette="Blues")
        elif bq_name == "bq9":
            sns.barplot(data=pdf, x="ORIGIN", y="sec_delay_share_pct", palette="Blues_d")
        elif bq_name == "bq10":
            sns.barplot(data=pdf, x="d_year", y="yoy_growth_pct", hue="ORIGIN", palette="Blues")
        elif bq_name == "bq11":
            sns.barplot(data=pdf, x="ORIGIN", y="avg_late_delay", palette="Blues_d")
        elif bq_name == "bq12":
            sns.barplot(data=pdf, x="ORIGIN", y="avg_total_taxi", palette="Blues_d")

        plt.title(title, fontsize=15, pad=20, color='#1f4e79')
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Зберігаємо графік
        chart_path = f"{OUTPUT_DIR}/{bq_name}_chart.png"
        plt.savefig(chart_path)
        plt.close()
        logger.info(f"Chart saved: {chart_path}")
    except Exception as e:
        logger.warning(f"Could not create chart for {bq_name}: {e}")


# Функції bq7 - bq12 залишаються такими ж, як у попередньому повідомленні...
# (Тут вони просто повертають DataFrame)

def bq7_distance_cancellation_impact(flights):
    return flights.withColumn("distance_category",
                              F.when(F.col("DISTANCE") < 500, "Short").when(F.col("DISTANCE") < 1500,
                                                                            "Medium").otherwise("Long")) \
        .groupBy("distance_category").agg(F.count("*").alias("total"),
                                          F.sum(F.col("CANCELLED").cast("int")).alias("cancelled")) \
        .withColumn("cancellation_rate_pct", F.round(F.col("cancelled") / F.col("total") * 100, 2)).orderBy(
        F.desc("cancellation_rate_pct"))


def bq8_carrier_recovery_speed(flights, date_dim):
    d_clean = date_dim.select("FL_DATE", F.col("YEAR").alias("d_year"), F.col("MONTH").alias("d_month")).distinct()
    res = flights.filter(F.col("DEP_DELAY") > 15).join(d_clean, "FL_DATE") \
        .groupBy("OP_CARRIER", "d_year", "d_month").agg(F.avg("DEP_DELAY").alias("avg_delay"),
                                                        F.avg("ARR_DELAY").alias("avg_arr_delay")) \
        .withColumn("recovery_diff", F.col("avg_arr_delay") - F.col("avg_delay"))
    w = Window.partitionBy("OP_CARRIER").orderBy("d_year", "d_month")
    return res.withColumn("prev_recovery", F.lag("recovery_diff").over(w)).limit(20)


def bq9_security_delay_airports(flights, airports_dim):
    air_c = airports_dim.select(F.col("AIRPORT").alias("AIR_ID"), "total_flights_from")
    return flights.filter(F.col("SECURITY_DELAY") >= 0).join(air_c, flights.ORIGIN == air_c.AIR_ID) \
        .groupBy("ORIGIN", "total_flights_from").agg(F.count("*").alias("sec_delayed_flights")) \
        .withColumn("sec_delay_share_pct",
                    F.round(F.col("sec_delayed_flights") / F.col("total_flights_from") * 100, 4)).orderBy(
        F.desc("sec_delay_share_pct")).limit(10)


def bq10_airport_growth_yoy(flights, date_dim):
    d_c = date_dim.select("FL_DATE", F.col("YEAR").alias("d_year")).distinct()
    yearly = flights.join(d_c, "FL_DATE").groupBy("ORIGIN", "d_year").agg(F.count("*").alias("flights"))
    w = Window.partitionBy("ORIGIN").orderBy("d_year")
    return yearly.withColumn("prev_year_flights", F.lag("flights").over(w)).filter("prev_year_flights is not null") \
        .withColumn("yoy_growth_pct",
                    F.round((F.col("flights") - F.col("prev_year_flights")) / F.col("prev_year_flights") * 100,
                            2)).limit(15)


def bq11_late_aircraft_propagation(flights):
    return flights.filter("LATE_AIRCRAFT_DELAY >= 0").groupBy("ORIGIN", "DEST").agg(
        F.avg("LATE_AIRCRAFT_DELAY").alias("avg_late_delay")) \
        .filter("avg_late_delay > 0").orderBy(F.desc("avg_late_delay")).limit(10)


def bq12_taxi_efficiency(flights, airports_dim):
    air_c = airports_dim.select(F.col("AIRPORT").alias("A_ID"), "total_flights_from")
    return flights.filter("TAXI_IN is not null and TAXI_OUT is not null").withColumn("total_taxi",
                                                                                     F.col("TAXI_IN") + F.col(
                                                                                         "TAXI_OUT")) \
        .join(air_c, F.col("ORIGIN") == F.col("A_ID")).groupBy("ORIGIN").agg(
        F.avg("total_taxi").alias("avg_total_taxi")).orderBy("avg_total_taxi").limit(10)


def run_business_questions_3(flights, carriers, airports, date):
    return bq7_distance_cancellation_impact(flights), bq8_carrier_recovery_speed(flights, date), \
        bq9_security_delay_airports(flights, airports), bq10_airport_growth_yoy(flights, date), \
        bq11_late_aircraft_propagation(flights), bq12_taxi_efficiency(flights, airports)


def save_results_3(r1, r2, r3, r4, r5, r6, output_path="output/business_logic_3"):
    os.makedirs(output_path, exist_ok=True)
    results = [r1, r2, r3, r4, r5, r6]
    titles = [
        "Cancellation Probability by Distance Category",
        "Carrier Recovery Speed Trend",
        "Security Delay Share by Airport",
        "Year-over-Year Flight Volume Growth",
        "Late Aircraft Delay Propagation by Route",
        "Airport Operational Efficiency (Total Taxi Time)"
    ]

    for i, df in enumerate(results, start=7):
        bq_id = f"bq{i}"
        print(f"\n--- Processing {bq_id} ---")
        df.show(10)

        # Створення графіка
        create_visualizations(df, bq_id, titles[i - 7])

        # Збереження CSV
        save_single_csv(df, f"{output_path}/{bq_id}.csv")