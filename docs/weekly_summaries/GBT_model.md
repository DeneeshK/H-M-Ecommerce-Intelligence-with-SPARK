NFO:__main__:                                                                  
====================================================================================================
INFO:__main__: MODEL PERFORMANCE METRICS
INFO:__main__:====================================================================================================
INFO:__main__:   AUC-ROC:   0.8441
INFO:__main__:   Accuracy:  0.7842
INFO:__main__:   Precision: 0.7799
INFO:__main__:   Recall:    0.7842
INFO:__main__:====================================================================================================
26/02/19 22:28:44 WARN SparkStringUtils: Truncated the string representation of a plan since it was too large. This behavior can be adjusted by setting 'spark.sql.debug.maxToStringFields'.
26/02/19 22:28:44 WARN DAGScheduler: Broadcasting large task binary with size 1336.4 KiB
26/02/19 22:28:47 WARN DAGScheduler: Broadcasting large task binary with size 1310.7 KiB
INFO:__main__:                                                                  
 Prediction Summary:
INFO:__main__:   Predicted churn:  134,917 (69.8%)
INFO:__main__:   Predicted active: 58,417 (30.2%)
26/02/19 22:28:47 WARN DAGScheduler: Broadcasting large task binary with size 1525.0 KiB
INFO:__main__:                                                                  
 Predictions saved at: /media/dk/Data/HM_Ecommerce_Project/processed_data/churn_predictions
INFO:__main__:
====================================================================================================
INFO:__main__: 🎯 ELITE CHURN MODEL COMPLETE
INFO:__main__:====================================================================================================
INFO:__main__:   AUC-ROC:   0.8441
INFO:__main__:   Accuracy:  0.7842
INFO:__main__:   Precision: 0.7799
INFO:__main__:   Recall:    0.7842
INFO:__main__:====================================================================================================
 SparkSession stopped successfully
INFO:py4j.clientserver:Closing down clientserver connection
(spark-env) dk@pc-365M:/media/dk/Data/HM_Ecommerce_Project$ python scripts/models/churn_model_simple.py
🔧 Detected 6 CPU cores
 Setting default parallelism to 12
26/02/19 22:34:17 WARN Utils: Your hostname, pc-365M resolves to a loopback address: 127.0.1.1; using 192.168.220.3 instead (on interface enp7s0)
26/02/19 22:34:17 WARN Utils: Set SPARK_LOCAL_IP if you need to bind to another address
Setting default log level to "WARN".
To adjust logging level use sc.setLogLevel(newLevel). For SparkR, use setLogLevel(newLevel).
26/02/19 22:34:17 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
26/02/19 22:34:18 WARN Utils: Service 'SparkUI' could not bind on port 4040. Attempting port 4041.
SparkSession created: Elite_Churn_Model_Tuning
Spark UI available at: http://localhost:4040
Spark Version: 3.5.0
INFO:__main__:====================================================================================================
INFO:__main__: ELITE CUSTOMER CHURN MODEL — WITH HYPERPARAMETER TUNING
INFO:__main__:====================================================================================================
INFO:__main__: Train customers: 902,837                                         
INFO:__main__: Test customers:  193,334
INFO:__main__:
 Training model with hyperparameter tuning (this will take time)...
26/02/19 22:48:15 WARN BlockManager: Asked to remove block broadcast_27579, which does not exist
INFO:py4j.clientserver:Closing down clientserver connection                     
INFO:py4j.clientserver:Closing down clientserver connection
INFO:py4j.clientserver:Closing down clientserver connection
INFO:py4j.clientserver:Closing down clientserver connection
INFO:py4j.clientserver:Closing down clientserver connection
INFO:__main__: Hyperparameter tuning completed!
INFO:__main__:                                                                  
====================================================================================================
INFO:__main__: FINAL TUNED MODEL PERFORMANCE
INFO:__main__:====================================================================================================
INFO:__main__:   AUC-ROC:   0.8447
INFO:__main__:   Accuracy:  0.7850
INFO:__main__:   Precision: 0.7807
INFO:__main__:   Recall:    0.7850
INFO:__main__:====================================================================================================
INFO:__main__:
 BEST MODEL PARAMETERS:
INFO:__main__:   maxDepth: 6
INFO:__main__:   maxIter: 50
INFO:__main__:   stepSize: 0.1
INFO:__main__:   subsamplingRate: 0.8
 SparkSession stopped successfully
INFO:py4j.clientserver:Closing down clientserver connection
(spark-env) dk@pc-365M:/media/dk/Data/HM_Ecommerce_Project$ 