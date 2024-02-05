#!/usr/bin/env python3
import os

import aws_cdk as cdk

from stacks.stack1.stack import Infra
from stacks.lambda_layers.stack import LambdaLayers

import settings

tier = settings.get_tier()
app = cdk.App()

# Tier agnostic stack to deploy layers to lambda
# Must be deployed first in new account as includes DB connection libs
LambdaLayers(
    app,
    f"LambdaLayers",
    env=cdk.Environment(account=settings.ACCOUNT, region=settings.REGION),
)

# CI/CD Pipeline
Infra(
    app,
    f"{tier}-Infra",
    env=cdk.Environment(account=settings.ACCOUNT, region=settings.REGION),
)


app.synth()
