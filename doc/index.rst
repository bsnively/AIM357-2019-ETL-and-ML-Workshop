##################################################################
AIM357 - Build an ETL pipeline to analyze customer data
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

.. toctree::
    :maxdepth: 2
    
****************
Notebooks
****************

.. toctree::
    :maxdepth: 2

    DataDiscovery/DataDiscoveryAndConversation

*********************************************************************
Crawl, Discover, and Explore the new datasets in a Data lake
*********************************************************************

A table in the AWS Glue Data Catalog is the metadata definition that represents the data in a data store. You create tables when you run a crawler, or you can create a table manually in the AWS Glue console. The Tables list in the AWS Glue console displays values of your table's metadata. You use table definitions to specify sources and targets when you create ETL (extract, transform, and load) jobs. 

You can use a crawler to populate the AWS Glue Data Catalog with tables. This is the primary method used by most AWS Glue users. A crawler can crawl multiple data stores in a single run. Upon completion, the crawler creates or updates one or more tables in your Data Catalog. Extract, transform, and load (ETL) jobs that you define in AWS Glue use these Data Catalog tables as sources and targets. The ETL job reads from and writes to the data stores that are specified in the source and target Data Catalog tables. 

After starting the crawler, you can go to the glue console if you'd like to see it running.

https://console.aws.amazon.com/glue/home?region=us-east-1#catalog:tab=crawlers

Quering the data

We'll use Athena to query the data. Athena allows us to perform SQL queries against datasets on S3, without having to transform them, load them into a traditional sql datastore, and allows rapid ad-hoc investigation.

Later we'll use Spark to do ETL and feature engineering.

Next we'll create an Athena connection we can use, much like a standard JDBC/ODBC connection

We see some bad data here...

We are expecting only 2018 and 2019 datasets here, but can see there are records far into the future and in the past. This represents bad data that we want to eliminate before we build our model.

All the bad data, at least the bad data in the future, is coming from the yellow taxi license type.
Note, we are querying the transformed data.

We should check the raw dataset to see if it's also bad or something happened in the ETL process

Let's find the 2 2088 records to make sure they are in the source data

Let's bring in the other fhvhv data since the new law went into affect

Some details of what caused this drop:
On August 14, 2018, Mayor de Blasio signed Local Law 149 of 2018, creating a new license category for TLC-licensed FHV businesses that currently dispatch or plan to dispatch more than 10,000 FHV trips in New York City per day under a single brand, trade, or operating name, referred to as High-Volume For-Hire Services (HVFHS). This law went into effect on Feb 1, 2019

Let's bring the other license type and see how it affects the time series charts:

.. image:: images/joinedtaxidata.png

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