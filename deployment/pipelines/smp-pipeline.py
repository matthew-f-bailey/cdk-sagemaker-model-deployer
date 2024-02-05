""" Create the pipeline from deploying models and linking it up with processing jobs"""
# python -m pipelines.moneyline_pipeline
import sys
import argparse
from pathlib import Path
from datetime import datetime
import subprocess
import uuid
import os
import json
import boto3
from botocore.exceptions import ClientError
import sagemaker
from sagemaker.inputs import TransformInput
from sagemaker.processing import (
    Processor,
    ProcessingInput,
    ProcessingOutput,
    FrameworkProcessor,
)
from sagemaker.transformer import Transformer
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.steps import ProcessingStep, TransformStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession, LocalPipelineSession
from sagemaker.workflow.functions import Join
from sagemaker.xgboost.estimator import XGBoost
from sagemaker.model import Model
from sagemaker.image_uris import get_base_python_image_uri
from sagemaker.model_monitor import DataCaptureConfig


from model_deployer import ModelConf

NOW = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

LOCAL_MODE = False
OVERWRITE_MODEL = False
parser = argparse.ArgumentParser()
parser.add_argument("-l", "--local", action="store_true")
parser.add_argument("-o", "--overwrite_model", action="store_true")
args = parser.parse_args()
if args.local:
    LOCAL_MODE = True
if args.overwrite_model:
    OVERWRITE_MODEL = True

CONF_FILENAME = "model_config.json"

# TODO: Relies on relative path, probably better way
MODELS_DIR = Path(Path(__file__).parent.parent.parent, "models/moneyline-pipeline")

if LOCAL_MODE:
    session = LocalPipelineSession()
    INSTANCE = "local"
else:
    session = PipelineSession()
    INSTANCE = "ml.m5.xlarge"

# LINK ALL OF IT UP IN PIPELINE
region = session.boto_region_name
role_name = "AmazonSageMaker-ExecutionRole-20230928T172214"
try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName=role_name)["Role"]["Arn"]

TIER = os.environ.get("AWS_TIER", "dev")
if TIER is None:
    raise ValueError(
        "AWS_TIER not found in env. Cannot deploy without a tier."
        " Export an env var of [dev|qa|uat|prod] before running."
    )


# Ensure model monitoring bucket is created
s3 = boto3.client('s3')
monitoring_bucket_name = 'model-monitoring-pipeline'
monitoring_s3_path = f"s3://{monitoring_bucket_name}/datacapture/"
try:
    s3.create_bucket(
        Bucket=monitoring_bucket_name, 
        CreateBucketConfiguration={"LocationConstraint": "us-east-2"}
    )
except s3.exceptions.BucketAlreadyOwnedByYou:
    print("Model Monitoring Bucket already created in account.")

# DEFINE PATHS TO MODELS
point_group_path = Path(MODELS_DIR, "model-point-group", CONF_FILENAME)
point_group_conf = ModelConf(point_group_path)

preprocessing_path = Path(MODELS_DIR, "preprocessing", CONF_FILENAME)
preprocessing_conf = ModelConf(preprocessing_path)

postprocessing_path = Path(MODELS_DIR, "postprocessing", CONF_FILENAME)
postprocessing_conf = ModelConf(postprocessing_path)

# SPREAD MODELS
spread_path = Path(MODELS_DIR, "preprocessing-spread", CONF_FILENAME)
spread_preprocessing_conf = ModelConf(spread_path)

spread_group_path = Path(MODELS_DIR, "model-spread-group", CONF_FILENAME)
spread_group_conf = ModelConf(spread_group_path)


def pipeline_str(*args):
    return Join(on="", values=[x for x in args])


# Set our input params
pipeline_name = f"{TIER}-modelMoneylineSagemakerPipeline"

# Will be passed in via stepfunction after taking the snapshot
snapshot_id = ParameterString(name="snapshotId")
input_file = ParameterString(name="inputFile")
execution_bucket = ParameterString(name="inputBucket")


def pipeline_str(*args):
    return Join(on="", values=[x for x in args])


bucket_prefix = [
    "s3://",
    execution_bucket,
    "/",
    snapshot_id,
    "/",
]

# Model inputs and outputs
preprocessing_input = pipeline_str(*bucket_prefix, "football/")
preprocessing_output = pipeline_str(*bucket_prefix, "preprocessing-output/")
preprocessing_considered_output = pipeline_str(*bucket_prefix, "considered/")

inference_input = preprocessing_output
point_group_output = pipeline_str(*bucket_prefix, "inference-output/", "point-group/")

# Spread model inputs and outputs
spread_preprocessing_input = pipeline_str(*bucket_prefix, "football/")
spread_preprocessing_output = pipeline_str(
    *bucket_prefix, "spread-preprocessing-output/"
)

spread_group_input = spread_preprocessing_output
spread_group_output = pipeline_str(*bucket_prefix, "inference-output/", "spread-group/")

postprocessing_inputs = {
    "point_group": point_group_output,
    "spread_group": spread_group_output,
}
postprocessing_output = pipeline_str(*bucket_prefix, "postprocessing-output/")


def code_location(core_name, code_type):
    return f"s3://sagemakermodel-artifacts/{NOW}/{code_type}-{core_name}"


def processing_code_location(core_name):
    return code_location(core_name=core_name, code_type="processing")


def model_code_location(core_name):
    return code_location(core_name=core_name, code_type="model")


#######################################################
# Deploy processors to ECR and create step from it
#######################################################
# PREPROCESSING CONTAINER
#######################################################
preprocessor = FrameworkProcessor(
    estimator_cls=XGBoost,
    framework_version="",
    role=role,
    image_uri=get_base_python_image_uri(region=region),
    instance_count=1,
    instance_type=INSTANCE,
    sagemaker_session=session,
    code_location=processing_code_location(preprocessing_conf.core_name),
)
preprocessor_args = preprocessor.run(
    inputs=[
        ProcessingInput(
            input_name="Historical",
            source=preprocessing_input,
            destination="/opt/ml/processing/input",
        ),
        ProcessingInput(
            input_name="WeeksInput",
            source=input_file,
            destination="/opt/ml/processing/input/predictions",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="Preprocessed",
            source="/opt/ml/processing/output/preprocessed",
            destination=preprocessing_output,
        ),
        ProcessingOutput(
            output_name="Considered",
            source="/opt/ml/processing/output/considered",
            destination=preprocessing_considered_output,
        ),
    ],
    source_dir=str(preprocessing_conf.source_dir),
    code=preprocessing_conf.entrypoint,
)
preprocess_step = ProcessingStep(
    name="PreprocessMoneyline",
    step_args=preprocessor_args,
)

#######################################################
# SPREAD PREPROCESSING CONTAINER
#######################################################
preprocessor_spread = FrameworkProcessor(
    estimator_cls=XGBoost,
    framework_version="",
    role=role,
    image_uri=get_base_python_image_uri(region=region),
    instance_count=1,
    # This takes a really long time, however pandarrallel takes cores to speed up
    # This has 8 cores which should speed up costs about the same
    instance_type="ml.c5.2xlarge",
    sagemaker_session=session,
    code_location=processing_code_location(spread_preprocessing_conf.core_name),
)
preprocessor_spread_args = preprocessor_spread.run(
    inputs=[
        ProcessingInput(
            input_name="Historical",
            source=preprocessing_input,
            destination="/opt/ml/processing/input",
        ),
        ProcessingInput(
            input_name="WeeksInput",
            source=input_file,
            destination="/opt/ml/processing/input/predictions",
        ),
    ],
    outputs=[
        ProcessingOutput(
            output_name="Preprocessed",
            source="/opt/ml/processing/output/preprocessed",
            destination=spread_preprocessing_output,
        )
    ],
    source_dir=str(spread_preprocessing_conf.source_dir),
    code=spread_preprocessing_conf.entrypoint,
)
preprocess_spread_step = ProcessingStep(
    name="PreprocessSpreadMoneyline",
    step_args=preprocessor_spread_args,
)

#######################################################
# POSTPROCESSING
#######################################################
postprocessor = FrameworkProcessor(
    estimator_cls=XGBoost,
    framework_version="",
    role=role,
    image_uri=get_base_python_image_uri(region=region),
    instance_count=1,
    instance_type=INSTANCE,
    sagemaker_session=session,
    code_location=processing_code_location(postprocessing_conf.core_name),
)
postprocessor_args = postprocessor.run(
    inputs=[
        ProcessingInput(
            input_name=name,
            source=s3_path,
            destination=f"/opt/ml/processing/input/{name}",
        )
        for name, s3_path in postprocessing_inputs.items()
    ]
    + [
        # Add in the considered teams
        ProcessingInput(
            input_name="considered",
            source=preprocessing_considered_output,
            destination=f"/opt/ml/processing/input/considered",
        )
    ],
    outputs=[
        ProcessingOutput(
            output_name="output",
            source="/opt/ml/processing/output",
            destination=postprocessing_output,
        )
    ],
    source_dir=str(postprocessing_conf.source_dir),
    code=postprocessing_conf.entrypoint,
)
postprocessing_step = ProcessingStep(
    name="PostprocessMoneyline",
    step_args=postprocessor_args,
)


#######################################################
# Deploy Models
def create_model(conf, inputs, outputs):
    if OVERWRITE_MODEL:
        try:
            sm = boto3.client("sagemaker")
            sm.describe_model(ModelName=conf.model_name)
            sm.delete_model(ModelName=conf.model_name)
            print("Overwriting current model of", conf.model_name)
        except ClientError:
            print("No Model Found for", conf.model_name)

    model = Model(
        name=conf.model_name,
        image_uri=conf.base_image,
        model_data=str(Path(conf.model_dir, "model.tar.gz")),
        source_dir=str(conf.source_dir),
        entry_point=conf.entrypoint,
        # Create outside of context of pipeline
        sagemaker_session=sagemaker.Session() if not LOCAL_MODE else session,
        role=role,
        code_location=model_code_location(conf.core_name),
    )
    if LOCAL_MODE:
        # In local mode, create model locally
        model_step = ModelStep(
            name="Create-" + conf.core_name,
            step_args=model.create(instance_type=INSTANCE),
        )
        model_name = model_step.properties.ModelName
    else:
        print("Creating Model", model.name)
        model.create()
        model_step = None
        model_name = model.name

    transformer = Transformer(
        model_name=model_name,
        instance_type=INSTANCE,
        instance_count=1,
        output_path=outputs,
        sagemaker_session=session,
    )
    transform_step = TransformStep(
        name="Inference-" + conf.core_name,
        step_args=transformer.transform(
            inputs, 
            content_type="text/csv",
            batch_data_capture_config=sagemaker.inputs.BatchDataCaptureConfig(
                destination_s3_uri=f"{monitoring_s3_path}/{conf.core_name}"
            )
        ),
    )
    return model, model_step, transformer, transform_step


#######################################################
# NON-SPREAD MODELS
#######################################################
(
    spread_group_model,
    spread_group_model_step,
    spread_group_transformer,
    spread_group_transform_step,
) = create_model(spread_group_conf, spread_group_input, spread_group_output)

(
    point_group_model,
    point_group_model_step,
    point_group_transformer,
    point_group_transform_step,
) = create_model(point_group_conf, inference_input, point_group_output)

#######################################################
# Link up all deps and steps in a pipeline
#######################################################
model_create_steps = []
if LOCAL_MODE:
    preprocess_step.depends_on = [point_group_model_step]
    preprocess_step.depends_on = [spread_group_model_step]

    model_create_steps = [
        point_group_model_step,
        spread_group_model_step,
    ]

point_group_transform_step.depends_on = [preprocess_step]
spread_group_transform_step.depends_on = [preprocess_spread_step]

postprocessing_step.depends_on = [
    point_group_transform_step,
    spread_group_transform_step,
]

pipeline = Pipeline(
    name=pipeline_name,
    parameters=[snapshot_id, input_file, execution_bucket],
    steps=[
        preprocess_step,
        preprocess_spread_step,
        point_group_transform_step,
        spread_group_transform_step,
        postprocessing_step,
    ]
    + model_create_steps,
    sagemaker_session=session,
)
print(pipeline.definition())

pipeline.upsert(role_arn=role)

if LOCAL_MODE:
    pipeline.start(
        parameters={}
    )
else:
    # EXPORT VARS NEEDED FOR INFRA TO RUN
    ssm = boto3.client("ssm")
    ssm.put_parameter(
        Name=f"{TIER}-SmpPipelineName",
        Value=pipeline_name,
        Description="Pipeline name for SmpPipelineName Pipeline. Read in by Lambda and given to sfn to start execution.",
        Type="String",
        Overwrite=True,
    )
    ssm.put_parameter(
        Name=f"{TIER}-SmpPipelineNameRole",
        Value=role_name,
        Description="Model Roles used by all models in pipeline",
        Type="String",
        Overwrite=True,
    )
