"""
Load Module - Feature Store
===========================
Saves engineered features to optimized Parquet format.

WHY Parquet?
- 10x faster read than CSV
- Columnar storage (efficient for ML)
- Schema preserved
- Supports partitioning
- Industry standard

Author: Data Engineering Portfolio Project
Date: February 2026
"""

from pyspark.sql import SparkSession
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStoreLoader:
    """
    Saves engineered features to Parquet-based feature store.
    """
    
    def __init__(self, spark, output_base_path):
        """
        Initialize feature store loader.
        
        Parameters:
        -----------
        spark : SparkSession
        output_base_path : str
            Base directory for feature store
        """
        self.spark = spark
        self.output_base_path = output_base_path
        logger.info(f"FeatureStoreLoader initialized")
        logger.info(f"Output path: {output_base_path}")
    
    
    def save_customer_features(self, customer_features_df, version="v1"):
        """
        Save customer features to Parquet.
        
        Parameters:
        -----------
        customer_features_df : DataFrame
            Engineered customer features
        version : str
            Version tag for feature store
        """
        logger.info("="*80)
        logger.info("SAVING CUSTOMER FEATURES TO FEATURE STORE")
        logger.info("="*80)
        
        output_path = f"{self.output_base_path}/customer_features_{version}"
        
        # Write to Parquet
        logger.info(f"Writing to: {output_path}")
        customer_features_df.write \
            .mode("overwrite") \
            .parquet(output_path)
        
        # Verify
        saved_count = self.spark.read.parquet(output_path).count()
        logger.info(f" Saved {saved_count:,} customer feature records")
        logger.info(f" Features: {len(customer_features_df.columns)}")
        logger.info(f" Path: {output_path}")
        
        return output_path
    
    
    def save_transactions_clean(self, transactions_df, version="v1"):
        """
        Save cleaned transactions (partitioned by year-month).
        
        WHY partition?
        - Faster queries (read only needed partitions)
        - Parallel writes
        - Easier data management
        
        Parameters:
        -----------
        transactions_df : DataFrame
            Clean transactions
        version : str
            Version tag
        """
        logger.info("="*80)
        logger.info("SAVING CLEAN TRANSACTIONS (PARTITIONED)")
        logger.info("="*80)
        
        from pyspark.sql.functions import year, month
        
        # Add partition columns
        trans_partitioned = transactions_df.withColumn(
            "year", year("t_dat")
        ).withColumn(
            "month", month("t_dat")
        )
        
        output_path = f"{self.output_base_path}/transactions_clean_{version}"
        
        logger.info(f"Writing partitioned data to: {output_path}")
        trans_partitioned.write \
            .partitionBy("year", "month") \
            .mode("overwrite") \
            .parquet(output_path)
        
        # Verify
        saved_count = self.spark.read.parquet(output_path).count()
        logger.info(f" Saved {saved_count:,} transaction records")
        logger.info(f" Partitioned by: year, month")
        logger.info(f" Path: {output_path}")
        
        return output_path
    
    
    def create_feature_metadata(self, customer_features_df, save_path):
        """
        Create metadata file documenting features.
        
        Parameters:
        -----------
        customer_features_df : DataFrame
            Feature DataFrame
        save_path : str
            Where to save metadata
        """
        logger.info("Creating feature metadata...")
        
        metadata = {
            "creation_date": datetime.now().isoformat(),
            "total_customers": customer_features_df.count(),
            "total_features": len(customer_features_df.columns),
            "feature_names": customer_features_df.columns,
            "schema": str(customer_features_df.schema)
        }
        
        # Save as text file
        metadata_path = f"{save_path}/METADATA.txt"
        with open(metadata_path.replace("file://", ""), 'w') as f:
            f.write("="*80 + "\n")
            f.write("FEATURE STORE METADATA\n")
            f.write("="*80 + "\n\n")
            f.write(f"Creation Date: {metadata['creation_date']}\n")
            f.write(f"Total Customers: {metadata['total_customers']:,}\n")
            f.write(f"Total Features: {metadata['total_features']}\n\n")
            f.write("Features:\n")
            f.write("-"*80 + "\n")
            for i, feat in enumerate(metadata['feature_names'], 1):
                f.write(f"{i:3d}. {feat}\n")
        
        logger.info(f" Metadata saved: {metadata_path}")
        
        return metadata


def main():
    """Test feature store save."""
    from spark_config import get_spark_session, stop_spark_session, PATHS
    from extract import DataExtractor
    from feature_engineering import CustomerFeatureEngineer
    
    spark = get_spark_session("Feature_Store_Save")
    
    try:
        # Load data
        print("\n Loading data...")
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
        # Engineer features
        print("\n  Engineering features...")
        engineer = CustomerFeatureEngineer(spark, reference_date="2020-09-15")
        customer_features = engineer.create_all_customer_features(
            transactions, articles, customers
        )
        
        # Save to feature store
        print("\n Saving to feature store...")
        loader = FeatureStoreLoader(spark, PATHS["processed_data"])
        
        # Save customer features
        features_path = loader.save_customer_features(customer_features, version="v1")
        
        # Save clean transactions
        trans_path = loader.save_transactions_clean(transactions, version="v1")
        
        # Create metadata
        loader.create_feature_metadata(customer_features, PATHS["processed_data"])
        
        print("\n" + "="*80)
        print(" FEATURE STORE CREATED SUCCESSFULLY!")
        print("="*80)
        print(f"Customer Features: {features_path}")
        print(f"Transactions: {trans_path}")
        print("="*80)
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        raise
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()