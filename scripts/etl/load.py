"""
Load Module - Save Features for Churn Model
===========================================
Saves clean features to Parquet for fast model training.

Author: Data Engineering Portfolio Project  
Date: February 2026 (Fixed)
"""

from pyspark.sql import SparkSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureStoreLoader:
    """
    Saves features to Parquet feature store.
    """
    
    def __init__(self, spark, output_path):
        self.spark = spark
        self.output_path = output_path
        logger.info(f"FeatureStoreLoader initialized: {output_path}")
    
    
    def save_churn_features(self, train_df, val_df, test_df, version="v1"):
        """
        Save train/val/test splits for churn model.
        
        Parameters:
        -----------
        train_df, val_df, test_df : DataFrames
            Feature splits
        version : str
            Version tag
        """
        logger.info("="*80)
        logger.info("SAVING CHURN FEATURES TO PARQUET")
        logger.info("="*80)
        
        # Save train
        train_path = f"{self.output_path}/churn_train_{version}"
        logger.info(f"\n Saving training set...")
        train_df.write.mode("overwrite").parquet(train_path)
        logger.info(f" {train_df.count():,} rows → {train_path}")
        
        # Save validation
        val_path = f"{self.output_path}/churn_val_{version}"
        logger.info(f"\n Saving validation set...")
        val_df.write.mode("overwrite").parquet(val_path)
        logger.info(f" {val_df.count():,} rows → {val_path}")
        
        # Save test
        test_path = f"{self.output_path}/churn_test_{version}"
        logger.info(f"\n Saving test set...")
        test_df.write.mode("overwrite").parquet(test_path)
        logger.info(f"    {test_df.count():,} rows → {test_path}")
        
        logger.info("\n" + "="*80)
        logger.info(" ALL FEATURE SETS SAVED")
        logger.info("="*80)
        
        return {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path
        }


def main():
    """Test feature saving."""
    import sys
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, project_root)
    
    from scripts.etl.spark_config import get_spark_session, stop_spark_session, PATHS
    from scripts.etl.extract import DataExtractor
    from scripts.etl.feature_engineering import ChurnFeatureEngineer
    from scripts.etl.clean_data import DataCleaner
    
    spark = get_spark_session("Load_Test")
    
    try:
        # Load, engineer, clean
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
        engineer = ChurnFeatureEngineer(spark, train_end_date="2020-06-24")
        features = engineer.create_churn_features(transactions, customers)
        
        cleaner = DataCleaner(spark)
        clean_features = cleaner.clean_churn_features(features)
        train, val, test = cleaner.create_train_val_test_split(clean_features)
        
        # Save
        loader = FeatureStoreLoader(spark, PATHS["processed_data"])
        paths = loader.save_churn_features(train, val, test, version="v1")
        
        print("\n Load test complete!")
        
    except Exception as e:
        print(f"\n Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    main()