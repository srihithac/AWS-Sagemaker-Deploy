import sagemaker
from sagemaker import get_execution_role
sagemaker_session = sagemaker.Session()
# Get a SageMaker-compatible role used by this Notebook Instance.
role = get_execution_role()
role
rain_input = sagemaker_session.upload_data("data")
train_input
from sagemaker.sklearn.estimator import SKLearn

script_path = 'startup_prediction.py'

sklearn = SKLearn(
    entry_point=script_path,
    instance_type="ml.m4.xlarge",
    framework_version="0.20.0",
    py_version="py3",
    role=role,
    sagemaker_session=sagemaker_session)
sklearn.fit({'train': train_input})
deployment = sklearn.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
deployment.endpoint
deployment.predict([[1,0,50000,25000,40000]])