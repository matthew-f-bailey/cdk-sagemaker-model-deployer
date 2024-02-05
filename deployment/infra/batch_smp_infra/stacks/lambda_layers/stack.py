from pathlib import Path

import aws_cdk
from aws_cdk import (
    Stack,
    aws_lambda as lambda_,
)
import cdk_docker_image_deployment
from constructs import Construct

import settings

PG_DIR = Path(Path(__file__).parent, "psycopglibs")

class LambdaLayers(Stack):
    """
    Creates various lambda layers needed for account
    """

    def __init__(self, scope: Construct, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        layer = lambda_.LayerVersion(self, "PostgresLayer",
            code=lambda_.Code.from_asset(str(PG_DIR)),
            compatible_runtimes=[lambda_.Runtime.PYTHON_3_9, lambda_.Runtime.PYTHON_3_10, lambda_.Runtime.PYTHON_3_11],
            license="Apache-2.0",
            description="Layer including psycopg deps",
        )
