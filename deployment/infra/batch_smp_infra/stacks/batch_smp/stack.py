from pathlib import Path
import uuid
import tarfile

import aws_cdk
from aws_cdk import (
    # Duration,
    Stack,
    Stage,
    Duration,
    aws_ecr as ecr,
    aws_s3 as s3,
    aws_stepfunctions as sfn,
    aws_events as events,
    aws_events_targets as targets,
    aws_s3_notifications as s3n,
    aws_lambda as lambda_,
    aws_lambda_destinations as destinations,
    aws_iam as iam,
    aws_ecr_assets as ecr_assets,
    aws_s3_assets as s3_assets,
    aws_ecr as ecr,
    aws_kms as kms,
    RemovalPolicy,
    aws_ssm as ssm,
    aws_ec2 as ec2,
    aws_rds as rds,
    aws_secretsmanager as secrets,
    aws_apigateway as apigateway,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
)
import cdk_docker_image_deployment
from constructs import Construct

import settings

FILE_DIR = Path(__file__).parent
LAMBDA_DIR = Path(FILE_DIR, "lambda").resolve()
SFN_DIR = Path(FILE_DIR, "stepfunctions").resolve()

# Existing ARN Dependencies
LAMBDA_PG_LAYER_VERSION = ""
LAMBDA_PG_LAYER_ARN = f""

EMAIL_NOTIFICATIONS = []


class NflMoneylineInfra(Stack):
    """
    Stack consists of the infrastructure portion of the nfl moneyline inference
    Deploys the s3 input bucket, the triggering lambda, the extraction sfn
    Relies on the model deployer to run prior and create the SMP
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        ######################################################
        # 0. Create notifcations
        ######################################################
        topic = sns.Topic(self, "NotificationTopic", topic_name = settings.resource_name("sns-topic"))
        for email in EMAIL_NOTIFICATIONS:
            topic.add_subscription(subscriptions.EmailSubscription(email))
        
        ######################################################
        # 1. Create the input bucket
        ######################################################
        # Location where inputs get placed
        input_bucket_name = settings.resource_name("-trigger-input-bucket").lower()
        input_bucket = s3.Bucket(
            self,
            settings.resource_name("TriggerInputBucket"),
            bucket_name=input_bucket_name,
            access_control=s3.BucketAccessControl.PRIVATE,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
        )
        export_bucket_name = settings.resource_name("-db-export-bucket").lower()
        db_export_bucket = s3.Bucket(
            self,
            settings.resource_name("DBExportBucket"),
            bucket_name=export_bucket_name,
            access_control=s3.BucketAccessControl.PRIVATE,
            encryption=s3.BucketEncryption.S3_MANAGED,
            versioned=True,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            removal_policy=RemovalPolicy.DESTROY,
        )

        ######################################################
        # 2. Create the lambda funciton that kicks off sfn
        ######################################################
        # Lambda that gets triggered from input file being placed in above bucket
        # Kicks off the stepfunction
        trigger_lambda = lambda_.Function(
            self,
            settings.resource_name("TriggerLambda"),
            handler="trigger_lambda.lambda_handler",
            runtime=lambda_.Runtime.PYTHON_3_10,
            code=lambda_.Code.from_asset(str(LAMBDA_DIR)),
            on_failure=destinations.SnsDestination(topic),
            on_success=destinations.SnsDestination(topic),
        )
        snapshot_check_lambda = lambda_.Function(
            self,
            settings.resource_name("SnapshotCheckLambda"),
            handler="snapshot_check_lambda.lambda_handler",
            runtime=lambda_.Runtime.PYTHON_3_10,
            code=lambda_.Code.from_asset(str(LAMBDA_DIR)),
            on_failure=destinations.SnsDestination(topic),
            on_success=destinations.SnsDestination(topic),
        )
        db_export_bucket.grant_read(snapshot_check_lambda)

        # Allow lambda to update
        topic.grant_publish(trigger_lambda)

        # ADD SAGEMAKER PIPELINE NAME EXPORTED TO SSM IN CI
        pipeline_name = f"{settings.get_tier().lower()}-NflMoneylinePipelineName"
        trigger_lambda.add_environment(
            "SAGEMAKER_PIPELINE_NAME",
            ssm.StringParameter.value_for_string_parameter(self, pipeline_name),
        )
        trigger_lambda.add_environment(
            "SNAPSHOT_LAMBDA_ARN", snapshot_check_lambda.function_arn
        )
        trigger_lambda.add_environment("SNS_TOPIC_ARN", topic.topic_arn)

        # Get the model role that was exported from CI
        model_role_name = ssm.StringParameter.value_for_string_parameter(
            self, f"{settings.get_tier().lower()}-NflMoneylineModelRole"
        )
        model_role = iam.Role.from_role_name(
            self, settings.resource_name("ModelRole"), role_name=model_role_name
        )
        model_role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSageMakerFullAccess")
        )
        model_role.add_to_principal_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["kms:*"],
                resources=["arn:aws:kms:*"],
            )
        )
        db_export_bucket.grant_read_write(model_role)
        input_bucket.grant_read_write(model_role)

        # Allow lambda to read bucket as well as bucket to trriger lambda
        input_bucket.grant_read(trigger_lambda)
        input_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED, s3n.LambdaDestination(trigger_lambda)
        )

        ######################################################
        # 3. Create the Stepfunction that extracts db
        ######################################################
        sfn_role = iam.Role(
            self,
            "StefunctionRole",
            assumed_by=iam.CompositePrincipal(
                iam.ServicePrincipal("export.rds.amazonaws.com"),
                iam.ServicePrincipal("states.amazonaws.com"),
            ),
        )
        # Create step function to perform DB extraction and setup inference
        step_func = sfn.StateMachine(
            self,
            settings.resource_name("Stepfunc"),
            definition_body=sfn.DefinitionBody.from_file(
                str(Path(SFN_DIR, "extract_stepfunc.json").resolve())
            ),
            role=sfn_role,
            timeout=Duration.minutes(45),
        )
        step_func.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=[
                    "rds:CreateDBSnapshot",
                    "rds:StartExportTask",
                    "rds:AddTagsToResource",
                    "sagemaker:StartPipelineExecution",
                    "iam:PassRole",
                    "kms:*",
                    "s3:GetBucketLocation",
                ],
                resources=[
                    "arn:aws:rds:*",
                    "arn:aws:sagemaker:*",
                    "arn:aws:kms:*",
                    input_bucket.bucket_arn,
                    step_func.role.role_arn,
                ],
            )
        )
        input_bucket.grant_read_write(step_func)
        db_export_bucket.grant_read_write(step_func)
        step_func.role.grant_assume_role(step_func)

        ######################################################
        # 4. Link up some config values and permissions
        ######################################################
        # Add stepfunction to lambda env vars so it knows what to call
        trigger_lambda.add_environment("STEPFUNCTION_ARN", step_func.state_machine_arn)
        trigger_lambda.add_environment("RDS_EXPORT_IAM_ROLE", step_func.role.role_arn)
        trigger_lambda.add_environment("TIER", settings.get_tier().lower())
        trigger_lambda.add_environment("EXPORT_BUCKET_NAME", export_bucket_name)

        # Allow stepfunc to call lambda
        snapshot_check_lambda.grant_invoke(step_func)

        key = kms.Key(self, settings.resource_name("RdsExportKmsKey"))
        key.grant_encrypt_decrypt(step_func)
        trigger_lambda.add_environment("RDS_EXPORT_KMS_KEY", key.key_arn)

        # Add permission to start the sfn
        trigger_lambda.add_to_role_policy(
            iam.PolicyStatement(
                effect=iam.Effect.ALLOW,
                actions=["states:StartExecution"],
                resources=[step_func.state_machine_arn],
            )
        )

        ######################################################
        # 5. Create lambda to persist the predictions
        ######################################################
        lambda_role = iam.Role(
            self,
            settings.resource_name("LambdaAccessRdsVpcRole"),
            assumed_by=iam.ServicePrincipal("lambda.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name("service-role/AWSLambdaVPCAccessExecutionRole")
            ]
        )

        pg_layer = lambda_.LayerVersion.from_layer_version_arn(
            self, 
            "LambdaLayerPg", 
            layer_version_arn=LAMBDA_PG_LAYER_ARN
        )

        # Create the actual lambda
        db_lambda = lambda_.Function(
            self,
            settings.resource_name("PersistPredsLambda"),
            handler="persist_in_db_lambda.lambda_handler",
            runtime=lambda_.Runtime.PYTHON_3_10,
            code=lambda_.Code.from_asset(str(LAMBDA_DIR)),
            timeout=Duration.minutes(1),
            role=lambda_role,
            layers=[pg_layer]
        )
        db_lambda.add_environment("TIER", settings.get_tier().lower())
        db_lambda.add_environment("TIER", settings.get_tier().lower())
        db_lambda.add_environment("SNS_TOPIC_ARN", topic.topic_arn)
        
        # Give lambda permissions to get notifications from bucket and conn to db
        db_export_bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(db_lambda),
            s3.NotificationKeyFilter(suffix="predictions.csv"),
        )
        db_export_bucket.grant_read(db_lambda)

        ######################################################
        # 5. Create lambda and API for interaction
        ######################################################
        endpoint_lambda = lambda_.Function(
            self,
            settings.resource_name("DbEndpointLambda"),
            handler="db_endpoint.lambda_handler",
            runtime=lambda_.Runtime.PYTHON_3_10,
            code=lambda_.Code.from_asset(str(LAMBDA_DIR)),
            timeout=Duration.minutes(1),
            role=lambda_role,
            layers=[pg_layer]
        )
        api = apigateway.LambdaRestApi(
            self,
            "DbEndpointApi",
            rest_api_name=f"{settings.get_tier()}-DbApi",
            handler=endpoint_lambda
        )
        teams = api.root.add_resource("team_name")
        api.root.add_method("GET")
        teams.add_method("GET")

        ######################################################
        # 5. Notifications for SMP
        ######################################################
        smp_notifications_lambda = lambda_.Function(
            self,
            settings.resource_name("SageMakerPipelineExecutionFailedLambda"),
            handler="smp_failure.lambda_handler",
            runtime=lambda_.Runtime.PYTHON_3_10,
            code=lambda_.Code.from_asset(str(LAMBDA_DIR)),
            timeout=Duration.minutes(1),
            role=lambda_role,
            layers=[pg_layer]
        )
        fail_rule = events.Rule(
            self,
            "SageMakerPipelineExecutionFailedTriggerRule",
            event_pattern=events.EventPattern(
                source=["aws.sagemaker"],
                detail={"currentPipelineExecutionStatus": ["Failed"]},
                detail_type=[
                    "SageMaker Model Building Pipeline Execution Status Change"
                ],
            ),
            targets=[targets.LambdaFunction(smp_notifications_lambda)],
        )
        smp_notifications_lambda.add_environment("SNS_TOPIC_ARN", topic.topic_arn)
        smp_notifications_lambda.add_environment("TIRE", settings.get_tier())
        topic.grant_publish(smp_notifications_lambda)
        topic.grant_publish(db_lambda)