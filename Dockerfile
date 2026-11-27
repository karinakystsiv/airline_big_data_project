ARG IMAGE_VARIANT=slim-bookworm
ARG OPENJDK_VERSION=8
ARG PYTHON_VERSION=3.11.9

FROM python:${PYTHON_VERSION}-${IMAGE_VARIANT} AS py3
FROM eclipse-temurin:${OPENJDK_VERSION}-jdk

COPY --from=py3 / /

ARG PYSPARK_VERSION=3.5.0
RUN pip --no-cache-dir install pyspark==${PYSPARK_VERSION} numpy pandas matplotlib seaborn matplotlib seaborn

COPY . .

CMD python main.py