import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import lit
from awsglue.dynamicframe import DynamicFrame

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'output_location'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

datasource0 = glueContext.create_dynamic_frame.from_options(connection_type="s3", connection_options = {"paths": ["s3://serverless-analytics/reinvent-2019/taxi_data/green/"]},  format="csv", transformation_ctx = "datasource0")

## @type: ApplyMapping
## @args: [mapping = [("vendorid", "long", "vendorid", "long"), ("lpep_pickup_datetime", "string", "pickup_datetime", "timestamp"), ("lpep_dropoff_datetime", "string", "dropoff_datetime", "timestamp"),("pulocationid", "long", "pulocationid", "long"), ("dolocationid", "long", "dolocationid", "long")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("vendorid", "long", "vendorid", "string"), ("lpep_pickup_datetime", "string", "pickup_datetime", "timestamp"), ("lpep_dropoff_datetime", "string", "dropoff_datetime", "timestamp"), ("pulocationid", "long", "pulocationid", "long"), ("dolocationid", "long", "dolocationid", "long")], transformation_ctx = "applymapping1")
## @type: ResolveChoice
## @args: [choice = "make_struct", transformation_ctx = "resolvechoice2"]
## @return: resolvechoice2
## @inputs: [frame = applymapping1]
resolvechoice2 = ResolveChoice.apply(frame = applymapping1, choice = "make_struct", transformation_ctx = "resolvechoice2")
## @type: DropNullFields
## @args: [transformation_ctx = "dropnullfields3"]
## @return: dropnullfields3
## @inputs: [frame = resolvechoice2]
dropnullfields3 = DropNullFields.apply(frame = resolvechoice2, transformation_ctx = "dropnullfields3")

##----------------------------------
#convert to a Spark DataFrame...
customDF = dropnullfields3.toDF()
customDF = customDF.withColumn("type", lit('green'))
customDF = customDF.coalesce(5)
customDynamicFrame = DynamicFrame.fromDF(customDF, glueContext, "customDF_df")
##----------------------------------

## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://serverless-analytics/reinvent-2019/canonical/"}, format = "parquet", transformation_ctx = "datasink4"]
## @return: datasink4
## @inputs: [frame = dropnullfields3]
datasink4 = glueContext.write_dynamic_frame.from_options(frame = customDynamicFrame, connection_type = "s3", connection_options = {"path": args['output_location']}, format = "parquet", transformation_ctx = "datasink4")
job.commit()
