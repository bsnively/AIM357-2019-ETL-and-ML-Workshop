# Running the Step Functions inference workflow 
*We're going to use AWS Step Functions to automate a workflow that will invoke our model endpoint and send and text us with the results. This is to exemplify that we can get notified about how our inferences are performing.*

## Activity 1: Update our Lambda function with the model endpoint name
*A Lambda function has been created before the workshop, which will invoke our model endpoint*

**Steps:**

1. In the AWS Console, click **Services** in the top, left-hand corner of the screen
2. Type **SageMaker** into the search field and hit Enter
3. Select **Endpoints** from the menu on the left-hand side of the screen (see screenshot below)

![Endpoint](/images/endpoint.png)

4. Copy the name of the endpoint to a text editor
5. Next, click **Services** in the top, left-hand corner of the screen
6. Type **Lambda** into the search field and hit Enter
7. Click on the lambda function name ("**reinvent-etl-inference-initialize-workflow**")
8. Scroll down to the **Function code** area, and replace the value of the **ENDPOINT_NAME* variable with the name of the endpoint that you copied from the SageMaker console (see screenshot below)

![lambda_function](/images/lambda.png)

9. Click **Save** in the top, right-hand corner of the screen

## Activity 2: Subscribe to the SNS topic
*An SNS topic for our workflow has also been created before the workshop, and you can subscribe your cell-phone number or email address to this topic to receive a notification with the model inference results. (We recommend using the SMS option and your cell-phone number, because the email method requires verifying your email address, which takes longer).*

**Steps:**

1. In the AWS Console, click **Services** in the top, left-hand corner of the screen
2. Type **SNS** into the search field and hit Enter
3. Select **Subscriptions** from the menu on the left-hand side of the screen
4. Click **Create subscription** 
5. In the **Topic ARN** field, select the ETLSNS topic ARN (see screenshot below)

![subscription](/images/subscription.png)

6. In the **Protocol** field, select SMS
7. In the **Endpoint** field, enter your cell-phone number (note that these accounts and all data stored within them will be deleted directly after this workshop, so all of these details will be deleted)
8. Click **Create subscription** 

## Activity 3: Let's test our workflow!

**Steps:**

1. In the AWS Console, click **Services** in the top, left-hand corner of the screen
2. Type **Step Functions** into the search field and hit Enter
3. Click **State machines** in the top, right-hand corner of the screen.
2. Click on the **reinvent-etl-inference-workflow** state machine 
3. Click **Start execution**
4. Paste the following json text into the input field (see screenshot below), and click **Start execution**. Note the version of the dataset that we are using for training, as denoted by the "DataDate" parameter:
```
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
```

![New Execution](/images/new_execution.png)

5. Now we can watch the workflow progress through each of the states. Be sure to to inspect the inputs and outputs of each state, and you should receive an SMS on your cell-phone with the inference results when it completes!
