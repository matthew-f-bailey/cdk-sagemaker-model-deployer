"""
Deploys a model directory to SM container

1. Read in config file from dir
"""
import argparse
import json
import os
import shutil
import subprocess
import tarfile
from typing import Union
from pathlib import Path

import boto3
from sagemaker.model import Model
from sagemaker import get_execution_role, Session

from settings import MODEL_ARTIFACT_BUCKET, MODEL_ARTIFACT_PREFIX

# Create sagemaker session
session = Session()
try:
    role = get_execution_role()
except ValueError:
    iam = boto3.client("iam")
    role = iam.get_role(RoleName="AmazonSageMaker-ExecutionRole-20230928T172214")[
        "Role"
    ]["Arn"]

AWS_ACCOUNT = boto3.client("sts").get_caller_identity().get("Account")
TIER = "dev"
if TIER is None:
    raise ValueError(
        "BTA_AWS_TIER not found in env. Cannot deploy without a tier."
        " Export an env var of [dev|qa|uat|prod] before running."
    )


class ModelConf:
    def __init__(
        self,
        config_filepath: Path,
    ):
        if not config_filepath.exists():
            raise FileNotFoundError(
                f"Config File not found @ {config_filepath.resolve()}"
            )

        # Read config file
        self.config_filepath = config_filepath
        with open(config_filepath, "r") as fh:
            self.config = json.load(fh)

        # Path Locations
        self.model_dir = Path(config_filepath.parent, "model")
        self.source_dir = Path(config_filepath.parent, "code")

        self.is_processing = False
        if self.config["type"] == "processing":
            self.is_processing = True
            # Code dir for processing jobs is everything at the config files level
            # Not just the /code directory
            self.source_dir = self.source_dir.parent

        # SM Model Params
        self.core_name = self.config["model"]["core_name"]
        self.version = self.config["model"]["version"]
        self.framework = self.config["model"]["framework"]
        self.model_filename = self.config["model"]["filename"]

        # Docker params
        self.entrypoint = self.config["docker"]["entrypoint"]
        self.base_image = self.config["docker"]["base_image"]

        self.model_name = f"{self.core_name}-{self.framework}-{self.version}-{TIER}"
        self.init()

    def init(self):
        """Takes an already trained model in the models dir where the model_config.json
        file is located and tars/zips them to be ready for upload to s3.

        Only zips up model types, if processing, nothing happens
        """
        if self.is_processing:
            return None
        # model_filepath = Path(self.model_dir, self.model_filename)
        zip_model_path = Path(self.model_dir, "model.tar.gz")
        if zip_model_path.exists():
            zip_model_path.unlink()
        with tarfile.open(zip_model_path, "w:gz") as tar:
            tar.add(self.model_dir, arcname="")
