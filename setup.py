from setuptools import setup, find_packages

setup(
    name="tf_defect_prediction_ml",
    version="0.1.0",
    description="Machine Learning component for Terraform Defect Prediction",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        "scikit-learn>=1.0.0",
        "pandas>=1.3.0",
        "joblib>=1.1.0",
        "numpy>=1.21.0",
        "lightgbm>=3.3.0",
        "mlflow>=2.0.0",
    ],
    python_requires=">=3.8",
    extras_require={
        "aws": ["boto3"],
        "azure": ["azure-storage-blob", "azure-identity"],
        "gcp": ["google-cloud-storage"],
    },
)
