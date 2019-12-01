
Running the Step Functions inference workflow
=============================================

*We're going to use AWS Step Functions to automate a workflow that will invoke our model endpoint and send and text us with the results. This is to exemplify that we can get notified about how our inferences are performing.*

Activity 1: Update our Lambda function with the model endpoint name
-------------------------------------------------------------------

*A Lambda function has been created before the workshop, which will invoke our model endpoint*

**Steps:**


#. In the AWS Console, click **Services** in the top, left-hand corner of the screen
#. Type **SageMaker** into the search field and hit Enter
#. Select **Endpoints** from the menu on the left-hand side of the screen (see screenshot below)


.. image:: /images/endpoint.png
   :target: /images/endpoint.png
   :alt: Endpoint



#. Copy the name of the endpoint to a text editor
#. Next, click **Services** in the top, left-hand corner of the screen
#. Type **Lambda** into the search field and hit Enter
#. Click on the lambda function name ("\ **reinvent-etl-inference-initialize-workflow**\ ")
#. Scroll down to the **Function code** area, and replace the value of the *\ *ENDPOINT_NAME* variable with the name of the endpoint that you copied from the SageMaker console (see screenshot below)


.. image:: /images/lambda.png
   :target: /images/lambda.png
   :alt: lambda_function



#. Click **Save** in the top, right-hand corner of the screen

Activity 2: Subscribe to the SNS topic
--------------------------------------

*An SNS topic for our workflow has also been created before the workshop, and you can subscribe your cell-phone number or email address to this topic to receive a notification with the model inference results. (We recommend using the SMS option and your cell-phone number, because the email method requires verifying your email address, which takes longer).*

**Steps:**


#. In the AWS Console, click **Services** in the top, left-hand corner of the screen
#. Type **SNS** into the search field and hit Enter
#. Select **Subscriptions** from the menu on the left-hand side of the screen
#. Click **Create subscription** 
#. In the **Topic ARN** field, select the ETLSNS topic ARN (see screenshot below)


.. image:: /images/subscription.png
   :target: /images/subscription.png
   :alt: subscription



#. In the **Protocol** field, select SMS
#. In the **Endpoint** field, enter your cell-phone number (note that these accounts and all data stored within them will be deleted directly after this workshop, so all of these details will be deleted)
#. Click **Create subscription** 

Activity 3: Let's test our workflow!
------------------------------------

**Steps:**


#. In the AWS Console, click **Services** in the top, left-hand corner of the screen
#. Type **Step Functions** into the search field and hit Enter
#. Click **State machines** in the top, right-hand corner of the screen.
#. Click on the **reinvent-etl-inference-workflow** state machine 
#. Click **Start execution**
#. Paste the following json text into the input field (see screenshot below), and click **Start execution**. Note the version of the dataset that we are using for training, as denoted by the "DataDate" parameter:
   .. code-block::

      {
      "instances": [
       {
         "start": "2019-07-03",
         "target": [
           120
         ]
       }
      ],
      "configuration": {
       "num_samples": 5,
       "output_types": [
         "mean",
         "quantiles",
         "samples"
       ],
       "quantiles": [
         "0.5",
         "0.9"
       ]
      }
      }


.. image:: /images/new_execution.png
   :target: /images/new_execution.png
   :alt: New Execution



#. Now we can watch the workflow progress through each of the states. Be sure to to inspect the inputs and outputs of each state, and you should receive an SMS on your cell-phone with the inference results when it completes!
