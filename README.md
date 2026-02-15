# 🛍️ H&M E-commerce Intelligence Platform

> End-to-end Big Data & Machine Learning pipeline for customer analytics, personalized recommendations, and churn prediction using Apache Spark

[![Apache Spark](https://img.shields.io/badge/Apache%20Spark-3.5.0-E25A1C?style=flat&logo=apachespark)](https://spark.apache.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python)](https://www.python.org/)
[![PySpark](https://img.shields.io/badge/PySpark-MLlib-orange)](https://spark.apache.org/mllib/)

---

##  Project Overview

A production-grade data engineering and machine learning system built to solve real e-commerce challenges using **31.8 million transactions** from H&M's fashion retail data.

### Business Problems Solved
1. **Customer Churn Prediction** - Identify at-risk customers before they leave
2. **Personalized Recommendations** - Increase conversion with AI-powered product suggestions  
3. **Customer Segmentation** - Enable targeted marketing campaigns

### Technical Highlights
-  Distributed ETL pipeline processing **3.5GB** of data using Apache Spark
-  Scalable feature engineering with **1.37M customers** and **105K products**
-  3 production-ready ML models (Churn, Recommendations, Segmentation)
-  End-to-end pipeline: Raw Data → ETL → ML → Business Insights

---

## Architecture
```
┌─────────────────┐
│   Raw Data      │  31.8M transactions, 1.37M customers, 105K products
│   (CSV files)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ETL Pipeline  │  Spark-based data cleaning, validation, feature engineering
│   (PySpark)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature Store  │  Customer features (RFM, behavioral), Product features
│   (Parquet)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Models     │  Churn (GBT), Recommendations (ALS), Segmentation (K-Means)
│  (Spark MLlib)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Predictions   │  Business insights, dashboards, API endpoints
│   & Insights    │
└─────────────────┘
```

---

##  Project Structure
```
HM_Ecommerce_Project/
├── raw_data/              # Original datasets (gitignored)
├── processed_data/        # Cleaned & transformed data (gitignored)
├── scripts/               # Production ETL & ML scripts
│   ├── etl/
│   │   ├── extract.py
│   │   ├── transform.py
│   │   └── load.py
│   ├── models/
│   │   ├── churn_prediction.py
│   │   ├── recommendation.py
│   │   └── segmentation.py
│   └── pipeline/
│       └── main_pipeline.py
├── notebooks/             # Jupyter notebooks for EDA & experiments
│   ├── 01_EDA.ipynb
│   ├── 02_Feature_Engineering.ipynb
│   ├── 03_Churn_Model.ipynb
│   ├── 04_Recommendations.ipynb
│   └── 05_Segmentation.ipynb
├── models/                # Saved ML models (gitignored)
├── outputs/               # Results, reports, predictions (gitignored)
├── logs/                  # Pipeline execution logs (gitignored)
├── config/                # Configuration files
├── DATA_DICTIONARY.md     # Dataset documentation
└── README.md              # This file
```

---

##  Quick Start

### Prerequisites
- Python 3.8+
- Apache Spark 3.5.0
- 8GB+ RAM recommended

### Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/HM_Ecommerce_Project.git
cd HM_Ecommerce_Project

# Create virtual environment
conda create -n spark-env python=3.8
conda activate spark-env

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline
```bash
# Execute end-to-end pipeline
python scripts/pipeline/main_pipeline.py --config config/config.yaml
```

---

##  Results & Business Impact

### Model Performance
| Model | Metric | Score |
|-------|--------|-------|
| **Churn Prediction** | AUC-ROC | TBD |
| **Recommendations** | Precision@10 | TBD |
| **Segmentation** | Silhouette Score | TBD |

### Business Impact (Projected)
-  **Churn Reduction:** X% decrease in customer attrition
-  **Revenue Lift:** Y% increase from personalized recommendations  
-  **Marketing Efficiency:** Z% improvement in campaign targeting

---

##  Tech Stack

- **Big Data Processing:** Apache Spark 3.5.0, PySpark
- **Machine Learning:** Spark MLlib, scikit-learn
- **Data Storage:** Parquet, CSV
- **Orchestration:** Python scripts
- **Notebooks:** Jupyter
- **Version Control:** Git

---

##  Dataset

**Source:** [Kaggle - H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations)

- **Transactions:** 31,788,324 records (2018-2020)
- **Customers:** 1,371,980 profiles
- **Products:** 105,542 articles with 25 attributes

---

##  Development Timeline

| Week | Phase | Deliverables |
|------|-------|--------------|
| 1-3 | ETL Pipeline | Data extraction, cleaning, feature engineering |
| 4-7 | ML Models | Churn, recommendations, segmentation models |
| 8 | Integration | Batch predictions, dashboard, API |
| 9 | Documentation | Technical docs, business report, presentation |

---

##  Author

**Deneesh Kumar**
- LinkedIn: 
- Email: dksteam2004@gmail.com



##  Acknowledgments

- H&M Group for providing the dataset
- Apache Spark community for excellent documentation
- Kaggle for hosting the competition
