from pyspark.sql import SparkSession


def main():
    print("Створення SparkSession.")

    spark = SparkSession.builder \
        .appName("Airline_Project") \
        .master("local[*]") \
        .getOrCreate()

    print("SparkSession успішно створено")


    print("Завершення роботи. Зупинка SparkSession.")
    spark.stop()

if __name__ == "__main__":
    main()
