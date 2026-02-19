"""
Data Validation Module
=====================
Performs initial data quality checks on loaded datasets.

WHY data validation?
- Catch issues early (missing values, duplicates, outliers)
- Verify data integrity before expensive transformations
- Document data quality for stakeholders
- Build trust in the pipeline
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, when, isnan, isnull, countDistinct
from pyspark.sql.functions import min as spark_min, max as spark_max, avg, stddev


class DataValidator:
    """
    Validates data quality across all datasets.
    """
    
    def __init__(self, articles_df, customers_df, transactions_df):
        self.articles = articles_df
        self.customers = customers_df
        self.transactions = transactions_df
        print("🔍 Data Validator initialized")
    
    
    def check_nulls(self, df, dataset_name):
        """
        Count NULL values in each column.
        
        WHY check nulls?
        - Nulls can break ML models
        - Indicates data quality issues
        - Helps decide imputation strategy
        """
        print(f"\n{'='*60}")
        print(f"🔍 NULL CHECK: {dataset_name}")
        print(f"{'='*60}")
        
        total_rows = df.count()
        
        # Get column data types
        # WHY? isnan() only works on numeric types, not dates/strings
        numeric_types = ['int', 'bigint', 'float', 'double', 'decimal']
        
        # Calculate null count for each column
        # Use isnan() only for numeric columns
        null_counts = {}
        for c in df.columns:
            col_type = dict(df.dtypes)[c]
            
            if any(t in col_type for t in numeric_types):
                # Numeric column: check both isnull and isnan
                null_counts[c] = df.select(
                    count(when(isnull(c) | isnan(c), c)).alias(c)
                ).collect()[0][c]
            else:
                # Non-numeric column: check only isnull
                null_counts[c] = df.select(
                    count(when(isnull(c), c)).alias(c)
                ).collect()[0][c]
        
        # Print results
        has_nulls = False
        for col_name, null_count in null_counts.items():
            if null_count > 0:
                pct = (null_count / total_rows) * 100
                print(f"    {col_name}: {null_count:,} nulls ({pct:.2f}%)")
                has_nulls = True
        
        if not has_nulls:
            print(f"   No NULL values found!")
        
        return null_counts
    
    
    def check_duplicates(self, df, dataset_name, key_column):
        """
        Check for duplicate records based on key column.
        """
        print(f"\n{'='*60}")
        print(f" DUPLICATE CHECK: {dataset_name}")
        print(f"{'='*60}")
        
        total_rows = df.count()
        unique_keys = df.select(key_column).distinct().count()
        duplicates = total_rows - unique_keys
        
        if duplicates > 0:
            print(f"    Found {duplicates:,} duplicate {key_column}s")
        else:
            print(f"   No duplicates found in {key_column}")
        
        print(f"   Total rows: {total_rows:,}")
        print(f"   Unique {key_column}s: {unique_keys:,}")
        
        return duplicates
    
    
    def check_date_range(self, df, date_column):
        """
        Check date range in transactions.
        """
        print(f"\n{'='*60}")
        print(f" DATE RANGE CHECK")
        print(f"{'='*60}")
        
        date_stats = df.select(
            spark_min(date_column).alias("min_date"),
            spark_max(date_column).alias("max_date")
        ).collect()[0]
        
        print(f"   First transaction: {date_stats['min_date']}")
        print(f"   Last transaction: {date_stats['max_date']}")
        
        # Calculate date span
        from datetime import datetime
        start = datetime.strptime(str(date_stats['min_date']), '%Y-%m-%d')
        end = datetime.strptime(str(date_stats['max_date']), '%Y-%m-%d')
        days_span = (end - start).days
        
        print(f"   Total days span: {days_span} days ({days_span/365:.1f} years)")
        
        return date_stats
    
    
    def check_value_distributions(self, df, column, dataset_name):
        """
        Check value distribution for categorical columns.
        """
        print(f"\n{'='*60}")
        print(f" VALUE DISTRIBUTION: {dataset_name}.{column}")
        print(f"{'='*60}")
        
        value_counts = df.groupBy(column).count() \
                         .orderBy(col("count").desc()) \
                         .limit(10)
        
        print(f"  Top 10 values:")
        value_counts.show(10, truncate=False)
    
    
    def validate_all(self):
        """
        Run all validation checks.
        """
        print("\n" + ""*30)
        print("STARTING DATA QUALITY VALIDATION")
        print("🔍"*30)
        
        # Articles validation
        print("\n" + "="*20)
        print("ARTICLES VALIDATION")
        print("="*20)
        self.check_nulls(self.articles, "Articles")
        self.check_duplicates(self.articles, "Articles", "article_id")
        self.check_value_distributions(self.articles, "product_group_name", "Articles")
        
        # Customers validation
        print("\n" + "="*20)
        print("CUSTOMERS VALIDATION")
        print("="*20)
        self.check_nulls(self.customers, "Customers")
        self.check_duplicates(self.customers, "Customers", "customer_id")
        self.check_value_distributions(self.customers, "club_member_status", "Customers")
        
        # Transactions validation
        print("\n" + "="*20)
        print("TRANSACTIONS VALIDATION")
        print("="*20)
        self.check_nulls(self.transactions, "Transactions")
        self.check_date_range(self.transactions, "t_dat")
        self.check_value_distributions(self.transactions, "sales_channel_id", "Transactions")
        
        # Check referential integrity
        self.check_referential_integrity()
        
        print("\n" + "="*60)
        print(" DATA QUALITY VALIDATION COMPLETE")
        print("="*60)
    
    
    def check_referential_integrity(self):
        """
        Check if transactions reference valid customers and articles.
        
        WHY this matters?
        - Ensures data consistency
        - Prevents JOIN errors later
        - Identifies orphaned records
        """
        print(f"\n{'='*60}")
        print(f" REFERENTIAL INTEGRITY CHECK")
        print(f"{'='*60}")
        
        # Check customer_id references
        trans_customers = self.transactions.select("customer_id").distinct()
        valid_customers = self.customers.select("customer_id")
        
        orphaned_customers = trans_customers.join(
            valid_customers, 
            on="customer_id", 
            how="left_anti"
        ).count()
        
        if orphaned_customers > 0:
            print(f"    {orphaned_customers:,} transactions reference non-existent customers")
        else:
            print(f"   All transactions reference valid customers")
        
        # Check article_id references
        trans_articles = self.transactions.select("article_id").distinct()
        valid_articles = self.articles.select("article_id")
        
        orphaned_articles = trans_articles.join(
            valid_articles,
            on="article_id",
            how="left_anti"
        ).count()
        
        if orphaned_articles > 0:
            print(f"    {orphaned_articles:,} transactions reference non-existent articles")
        else:
            print(f"   All transactions reference valid articles")


def main():
    """Test validation module."""
    from spark_config import get_spark_session, stop_spark_session, PATHS
    from extract import DataExtractor
    
    spark = get_spark_session("Data_Validation_Test")
    
    try:
        # Load data
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
        # Validate data
        validator = DataValidator(articles, customers, transactions)
        validator.validate_all()
        
    except Exception as e:
        print(f"\n Error during validation: {str(e)}")
        raise
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()