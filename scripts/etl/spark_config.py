"""
Spark Configuration Module
==========================
This module creates optimized SparkSession configurations for the H&M project.


"""

from pyspark.sql import SparkSession
import os
import multiprocessing

def get_spark_session(app_name="HM_Ecommerce_ETL"):
    """
    Creates and returns an optimized SparkSession for local execution.
    
    Parameters:
    -----------
    app_name : str
        Name of the Spark application (shows in Spark UI)
    
    Returns:
    --------
    SparkSession : Configured Spark session ready for use
    
    Configuration Details:
    ----------------------
    - Driver Memory: 5GB (for collecting results)
    - Executor Memory: 5GB (for processing partitions)
    - Cores: Auto-detect available CPUs
    - Partitions: 2x CPU cores (optimal for I/O + compute)
    """
    
    # Auto-detect available CPU cores
    num_cores = multiprocessing.cpu_count()
    print(f" Detected {num_cores} CPU cores")
    
    # Calculate optimal partition count (2x cores is good for mixed workloads)
    default_parallelism = num_cores * 2
    print(f" Setting default parallelism to {default_parallelism}")
    
    # Build Spark session with optimized configuration
    spark = SparkSession.builder \
        .appName(app_name) \
        .master(f"local[{num_cores}]") \
        .config("spark.driver.memory", "6g") \
        .config("spark.executor.memory", "6g") \
        .config("spark.sql.shuffle.partitions", str(default_parallelism)) \
        .config("spark.default.parallelism", str(default_parallelism)) \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "false") \
        .config("spark.driver.maxResultSize", "4g") \
        .getOrCreate()
    
    # Set log level to reduce noise (change to INFO for debugging)
    spark.sparkContext.setLogLevel("WARN")
    
    # Print confirmation
    print(f"SparkSession created: {app_name}")
    print(f"Spark UI available at: http://localhost:4040")
    print(f"Spark Version: {spark.version}")
    
    return spark


def stop_spark_session(spark):
    """
    Gracefully stops SparkSession and releases resources.
    
    WHY this matters:
    - Prevents memory leaks
    - Releases file handles
    - Cleans up temporary files
    """
    if spark:
        spark.stop()
        print(" SparkSession stopped successfully")


# Configuration dictionary for paths (we'll use this in all scripts)
PATHS = {
    "raw_data": "/media/dk/Data/HM_Ecommerce_Project/raw_data",
    "processed_data": "/media/dk/Data/HM_Ecommerce_Project/processed_data",
    "models": "/media/dk/Data/HM_Ecommerce_Project/models",
    "outputs": "/media/dk/Data/HM_Ecommerce_Project/outputs",
    "logs": "/media/dk/Data/HM_Ecommerce_Project/logs"
}


if __name__ == "__main__":
    # Test the configuration
    print("Testing Spark Configuration...")
    spark = get_spark_session("Config_Test")
    
    # Create a simple test DataFrame
    test_data = [(1, "test"), (2, "spark"), (3, "config")]
    test_df = spark.createDataFrame(test_data, ["id", "value"])
    
    print("\n Test DataFrame:")
    test_df.show()
    
    print(f"\n Configuration test passed!")
    print(f" Partitions: {test_df.rdd.getNumPartitions()}")
    
    # Stop session
    stop_spark_session(spark)