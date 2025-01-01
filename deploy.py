import sagemaker
from sagemaker.model import Model

# Using the python SDK to deploy the model to SageMaker
# Not working, need to resolve the issue
# UserWarning: Field name "json" in "MonitoringDatasetFormat" shadows an attribute in parent "Base"
# Some region setup issue

# Define S3 path for your model artifact
s3_artifact_path = "s3://footballanalysisbucket/football_inference.zip"

# Define SageMaker role and session
role = "Enter_the_role_ARN_here"
session = sagemaker.Session()

# Define the SageMaker model
model = Model(
    model_data=s3_artifact_path,
    role=role,
    entry_point="main.py",  # Entry point script
    source_dir="./",        # Path containing the project files
    image_uri="763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:1.12.1-cpu-py39-ubuntu20.04",  # Prebuilt SageMaker container
)

# Deploy the model
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large"
)
