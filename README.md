# Sagemaker Model and Pipeline Deployment

Condensed repo to develop, train, deploy, and orchestrate sagemaker pipelines. 

- Uses DS libs to create and train models. Can do this locally or via training jobs.
- Includes a model deployer to package up model files either locally or in s3 and create models to be used in pipelines.
- Includes a pipeline config section to orchestrate the deployment of processing, transform, training and monitoring jobs.
- Includes some dummy code to deploy cdk stacks for the infra needed for AWS and pipelines. 

In a production env, you may want to break these out to use in other CI/CD tools.