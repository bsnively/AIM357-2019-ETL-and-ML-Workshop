##################################################################
TEST PAGE2
##################################################################

Machine learning involves more than just training models; you need to source and prepare data, engineer features, select algorithms, train and tune models, and then deploy those models and monitor their performance in production. Learn how to set up an ETL pipeline to analyze customer data using Amazon SageMaker, AWS Glue, and AWS Step Functions.

This workshop will be around the ETL and full pipeline to perofrm Time-series forecasting 
using NYC Taxi Dataset.  It includes the following steps:

- Crawl, Discover, and Explore the new datasets in a Data lake
- Perform Extract, Transform, Load (ETL) jobs to clean the data
- Train a Machine Learning model and run inference
- Assess the response
- Send an alert if value is outside specified range

The workshop uses the following architecture:

.. image:: images/WorkshopArchitecture.png

*********************************************************************
Crawl, Discover, and Explore the new datasets in a Data lake
*********************************************************************

In this section, we'll use AWS Glue to crawl a new dataset in a data lake,  We'll then use Glue and Athena to explore the dataset

Crawling the datasets
=========================

Glue let's us crawl

*********************************************************************
Perform Extract, Transform, Load (ETL) jobs to clean the data
*********************************************************************

Info on Step 2

Sub step under Step 2
=========================

Your training script must be a Python 2.7 or 3.6 compatible source file.

*********************************************************************
Train a Machine Learning model and run inference
*********************************************************************

We will now train a model and test the inference

*********************************************************************
Assess the response
*********************************************************************

We'll assess how well our model performs

*********************************************************************
Send an alert if value is outside specified range
*********************************************************************

We'll send alerts when new incomign values are outside the range
