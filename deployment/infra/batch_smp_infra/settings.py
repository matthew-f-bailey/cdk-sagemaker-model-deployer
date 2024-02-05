import os
from pathlib import Path


# Bucket must be present prior to deploying
ACCOUNT = "xxxxxxxx"
REGION = "us-east-1"
GH_CONN_ARN = "xxxxxx"

# The codestar connection to github (Must be present before deploy)
# Developer Tools > Settings > Connections
GITHUB_CONNECTION_ARN = f"arn:aws:codestar-connections:{REGION}:{ACCOUNT}:connection/{GH_CONN_ARN}"


def get_tier():
    return os.environ.get("TIER", "dev").title()


def resource_name(text):
    return f"{get_tier()}-NflMoneyline{text.title()}"
