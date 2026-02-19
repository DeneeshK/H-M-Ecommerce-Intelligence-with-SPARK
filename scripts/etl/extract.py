"""
Data Extraction Module
======================
Loads raw CSV files into Spark DataFrames with proper schema inference,
validation, and error handling.

Business Context:
- Articles: Product catalog (105K products, 25 attributes)
- Customers: Customer profiles (1.37M customers, 7 attributes)
- Transactions: Purchase history (31.8M transactions, 5 attributes)

WHY this module:
- Centralized data loading logic
- Schema validation and type casting
- Memory-efficient reading with Spark
- Detailed logging and error handling
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import os
import time
from datetime import datetime


class DataExtractor:
    """
    Handles extraction of raw CSV data into Spark DataFrames.
    
  
    """
    
    def __init__(self, spark, raw_data_path):
        """
        Initialize the DataExtractor.
        
        Parameters:
        -----------
        spark : SparkSession
            Active Spark session
        raw_data_path : str
            Path to directory containing raw CSV files
        """
        self.spark = spark
        self.raw_data_path = raw_data_path
        self.articles_df = None
        self.customers_df = None
        self.transactions_df = None
        
        print(f" Data Extractor initialized")
        print(f" Raw data path: {raw_data_path}")
    
    
    def load_articles(self):
        """
        Load articles (product catalog) CSV file.
        
        
        Returns:
        --------
        DataFrame : Spark DataFrame containing articles data
        """
        
        print(" LOADING ARTICLES (Product Catalog)")
        
        
        file_path = os.path.join(self.raw_data_path, "articles.csv")
        
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f" File not found: {file_path}")
        
        start_time = time.time()
        
        # Load CSV with Spark
        # WHY these options?
        # - header=True: First row contains column names
        # - inferSchema=True: Auto-detect data types (int, string, etc.)
        # - escape='"': Handle quotes in product descriptions
        self.articles_df = self.spark.read.csv(
            file_path,
            header=True,
            inferSchema=True,
            escape='"'
        )
        
        # Cache in memory (we'll use this multiple times)
        self.articles_df.cache()
        
        elapsed_time = time.time() - start_time
        
        # Print summary statistics
        row_count = self.articles_df.count()
        col_count = len(self.articles_df.columns)
        partitions = self.articles_df.rdd.getNumPartitions()
        
        print(f" Loaded successfully in {elapsed_time:.2f} seconds")
        print(f" Rows: {row_count:,}")
        print(f" Columns: {col_count}")
        print(f" Partitions: {partitions}")
        print(f" Cached in memory: Yes")
        
        return self.articles_df
    
    
    def load_customers(self):
        """
        Load customers (customer profiles) CSV file.
        
        Returns:
        --------
        DataFrame : Spark DataFrame containing customers data
        """
        
        
        file_path = os.path.join(self.raw_data_path, "customers.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f" File not found: {file_path}")
        
        start_time = time.time()
        
        self.customers_df = self.spark.read.csv(
            file_path,
            header=True,
            inferSchema=True
        )
        
        # Cache in memory
        self.customers_df.cache()
        
        elapsed_time = time.time() - start_time
        
        row_count = self.customers_df.count()
        col_count = len(self.customers_df.columns)
        partitions = self.customers_df.rdd.getNumPartitions()
        
        print(f" Loaded successfully in {elapsed_time:.2f} seconds")
        print(f" Rows: {row_count:,}")
        print(f" Columns: {col_count}")
        print(f" Partitions: {partitions}")
        print(f" Cached in memory: Yes")
        
        return self.customers_df
    
    
    def load_transactions(self):
        """
        Load transactions (purchase history) CSV file.
        
        WHY special attention to transactions?
        - Largest file (31.8M rows, 3.3GB)
        - Most critical for ML models
        - Needs proper partitioning for performance
        
        Returns:
        --------
        DataFrame : Spark DataFrame containing transactions data
        """
     
        
        file_path = os.path.join(self.raw_data_path, "transactions_train.csv")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f" File not found: {file_path}")
        
        start_time = time.time()
        
        # Load transactions with optimized settings
        self.transactions_df = self.spark.read.csv(
            file_path,
            header=True,
            inferSchema=True
        )
        
        # DON'T cache transactions yet (too large for memory)
        # We'll partition and cache specific subsets later
        
        elapsed_time = time.time() - start_time
        
        row_count = self.transactions_df.count()
        col_count = len(self.transactions_df.columns)
        partitions = self.transactions_df.rdd.getNumPartitions()
        
        print(f" Loaded successfully in {elapsed_time:.2f} seconds")
        print(f" Rows: {row_count:,}")
        print(f" Columns: {col_count}")
        print(f" Partitions: {partitions}")
        print(f" Cached in memory: No (too large)")
        
        return self.transactions_df
    
    
    def load_all(self):
        """
        Load all three datasets in sequence.
        
        
        
        
        
        Returns:
        --------
        tuple : (articles_df, customers_df, transactions_df)
        """
        
        print("STARTING DATA EXTRACTION PIPELINE")
        
        
        start_time = time.time()
        
        # Load each dataset
        self.load_articles()
        self.load_customers()
        self.load_transactions()
        
        total_time = time.time() - start_time
        
        
        print(" ALL DATA LOADED SUCCESSFULLY")
        
        print(f"  Total loading time: {total_time:.2f} seconds")
        print(f" Total rows: {self.get_total_row_count():,}")
        
        return self.articles_df, self.customers_df, self.transactions_df
    
    
    def get_total_row_count(self):
        """Calculate total rows across all datasets."""
        total = 0
        if self.articles_df:
            total += self.articles_df.count()
        if self.customers_df:
            total += self.customers_df.count()
        if self.transactions_df:
            total += self.transactions_df.count()
        return total
    
    
    def print_schemas(self):
        """
        Print schema (column names and data types) for all datasets.
        
        WHY print schemas?
        - Verify data types were inferred correctly
        - Understand data structure before transformations
        - Catch issues early (wrong types, missing columns)
        """
        
        print("📋 DATASET SCHEMAS")
       
        
        if self.articles_df:
            print("\n🔸 ARTICLES SCHEMA:")
            self.articles_df.printSchema()
        
        if self.customers_df:
            print("\n🔸 CUSTOMERS SCHEMA:")
            self.customers_df.printSchema()
        
        if self.transactions_df:
            print("\n🔸 TRANSACTIONS SCHEMA:")
            self.transactions_df.printSchema()


def main():
    """
    Main function to test the extraction module.
    """
    # Import configuration
    from spark_config import get_spark_session, stop_spark_session, PATHS
    
    # Create Spark session
    spark = get_spark_session("Data_Extraction_Test")
    
    try:
        # Initialize extractor
        extractor = DataExtractor(spark, PATHS["raw_data"])
        
        # Load all data
        articles, customers, transactions = extractor.load_all()
        
        # Print schemas
        extractor.print_schemas()
        
        # Show sample data from each dataset
        print("\n" + "="*60)
        print(" SAMPLE DATA PREVIEW")
        print("="*60)
        
        print("\n Articles (first 3 rows):")
        articles.show(3, truncate=50)
        
        print("\n Customers (first 3 rows):")
        customers.show(3, truncate=50)
        
        print("\n Transactions (first 3 rows):")
        transactions.show(3, truncate=50)
        
        print("\n Extraction test completed successfully!")
        
    except Exception as e:
        print(f"\n Error during extraction: {str(e)}")
        raise
    
    finally:
        # Always stop Spark session to release resources
        stop_spark_session(spark)


if __name__ == "__main__":
    main()