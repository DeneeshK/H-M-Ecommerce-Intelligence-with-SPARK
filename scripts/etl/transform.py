"""
Complete Transform Pipeline
===========================
Master pipeline that orchestrates feature engineering, cleaning, and saving.

Pipeline Flow:
1. Extract raw data (CSV)
2. Engineer features (40+ features)
3. Clean data (handle nulls, outliers)
4. Split train/test
5. Save to feature store (Parquet)

Author: Data Engineering Portfolio Project
Date: February 2026
"""

from pyspark.sql import SparkSession
from spark_config import get_spark_session, stop_spark_session, PATHS
from extract import DataExtractor
from feature_engineering import CustomerFeatureEngineer
from clean_data import DataCleaner
from load import FeatureStoreLoader
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_complete_transform_pipeline(
    reference_date="2020-09-15",
    test_split_date="2020-09-08",
    version="v1"
):
    """
    Execute complete ETL transform pipeline.
    
    Parameters:
    -----------
    reference_date : str
        Cutoff date for feature calculation
    test_split_date : str
        Date to split train/test
    version : str
        Feature store version tag
    """
    logger.info("="*80)
    logger.info(" COMPLETE TRANSFORM PIPELINE STARTING")
    logger.info("="*80)
    logger.info(f"Reference Date: {reference_date}")
    logger.info(f"Test Split Date: {test_split_date}")
    logger.info(f"Version: {version}")
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("="*80)
    
    # Initialize Spark
    spark = get_spark_session("Complete_Transform_Pipeline")
    
    try:
        
        # STEP 1: EXTRACT
        
        logger.info("\n" + "="*80)
        logger.info("STEP 1/5: EXTRACT RAW DATA")
        logger.info("="*80)
        
        extractor = DataExtractor(spark, PATHS["raw_data"])
        articles, customers, transactions = extractor.load_all()
        
       
        # STEP 2: FEATURE ENGINEERING
        
        logger.info("\n" + "="*80)
        logger.info("STEP 2/5: ENGINEER FEATURES")
        logger.info("="*80)
        
        engineer = CustomerFeatureEngineer(spark, reference_date=reference_date)
        customer_features = engineer.create_all_customer_features(
            transactions_df=transactions,
            articles_df=articles,
            customers_df=customers
        )
        
       
        # STEP 3: DATA CLEANING
        
        logger.info("\n" + "="*80)
        logger.info("STEP 3/5: CLEAN DATA")
        logger.info("="*80)
        
        cleaner = DataCleaner(spark)
        clean_features = cleaner.clean_customer_features(customer_features)
        
       
        # STEP 4: TRAIN/TEST SPLIT
        
        logger.info("\n" + "="*80)
        logger.info("STEP 4/5: CREATE TRAIN/TEST SPLIT")
        logger.info("="*80)
        
        train_df, test_df = cleaner.create_train_test_split(
            clean_features, 
            test_date=test_split_date
        )
        
        
        # STEP 5: SAVE TO FEATURE STORE
        
        logger.info("\n" + "="*80)
        logger.info("STEP 5/5: SAVE TO FEATURE STORE")
        logger.info("="*80)
        
        loader = FeatureStoreLoader(spark, PATHS["processed_data"])
        
        # Save complete dataset
        features_path = loader.save_customer_features(clean_features, version=f"{version}_complete")
        
        # Save train split
        train_path = loader.save_customer_features(train_df, version=f"{version}_train")
        
        # Save test split
        test_path = loader.save_customer_features(test_df, version=f"{version}_test")
        
        # Save clean transactions
        trans_path = loader.save_transactions_clean(transactions, version=version)
        
        # Create metadata
        loader.create_feature_metadata(clean_features, PATHS["processed_data"])
        
        
        # PIPELINE COMPLETE
        
        logger.info("\n" + "="*80)
        logger.info(" TRANSFORM PIPELINE COMPLETE!")
        logger.info("="*80)
        logger.info(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("\n OUTPUT PATHS:")
        logger.info(f"   Complete Features: {features_path}")
        logger.info(f"   Train Set: {train_path}")
        logger.info(f"   Test Set: {test_path}")
        logger.info(f"   Transactions: {trans_path}")
        logger.info("\n SUMMARY:")
        logger.info(f"   Total Customers: {clean_features.count():,}")
        logger.info(f"   Train Customers: {train_df.count():,}")
        logger.info(f"   Test Customers: {test_df.count():,}")
        logger.info(f"   Features: {len(clean_features.columns)}")
        logger.info("="*80)
        
        return {
            "features_path": features_path,
            "train_path": train_path,
            "test_path": test_path,
            "transactions_path": trans_path,
            "train_count": train_df.count(),
            "test_count": test_df.count(),
            "feature_count": len(clean_features.columns)
        }
        
    except Exception as e:
        logger.error(f"\n Pipeline failed: {str(e)}")
        raise
    
    finally:
        stop_spark_session(spark)


if __name__ == "__main__":
    # Run complete pipeline
    results = run_complete_transform_pipeline(
        reference_date="2020-09-15",
        test_split_date="2020-09-08",
        version="v1"
    )
    
    print("\n" + "="*80)
    print(" WEEK 2-3: ETL TRANSFORM COMPLETE")
    print("="*80)
    print(f"All feature sets saved and ready for ML modeling (Week 4+)")
    print("="*80)