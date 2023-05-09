from presidio_anonymizer import AnonymizerEngine
from presidio_analyzer import AnalyzerEngine
from typing import List
from sagemaker.huggingface.model import HuggingFaceModel
from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts
from transformers import pipeline
import sagemaker
import boto3
from presidio_analyzer import PatternRecognizer
titles_recognizer = PatternRecognizer(
    supported_entity="TITLE", deny_list=["Mr.", "Mrs.", "Miss"])

sess = sagemaker.Session()
# sagemaker session bucket -> used for uploading data, models and logs
# sagemaker will automatically create this bucket if it not exists
sagemaker_session_bucket = None
if sagemaker_session_bucket is None and sess is not None:
    # set to default bucket if a bucket name is not given
    sagemaker_session_bucket = sess.default_bucket()

try:
    role = sagemaker.get_execution_role()
except ValueError:
    iam = boto3.client('iam')
    role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']

sess = sagemaker.Session(default_bucket=sagemaker_session_bucket)

print(f"sagemaker role arn: {role}")
print(f"sagemaker bucket: {sess.default_bucket()}")
print(f"sagemaker session region: {sess.boto_region_name}")


class TransformersRecognizer(EntityRecognizer):
    def __init__(self, model_id_or_path=None, aggregation_strategy="average", supported_language="en", ignore_labels=["O", "MISC"]):
        # inits transformers pipeline for given mode or path
        self.pipeline = pipeline("token-classification", model=model_id_or_path,
                                 aggregation_strategy="average", ignore_labels=ignore_labels)
        # map labels to presidio labels
        self.label2presidio = {
            "PER": "PERSON",
            "LOC": "LOCATION",
            "ORG": "ORGANIZATION",
        }

        # passes entities from model into parent class
        super().__init__(supported_entities=list(self.label2presidio.values()),
                         supported_language=supported_language)

    def load(self) -> None:
        """No loading is required."""
        pass

    def analyze(
        self, text: str, entities: List[str] = None, nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        Extracts entities using Transformers pipeline
        """
        results = []

        # keep max sequence length in mind
        predicted_entities = self.pipeline(text)
        if len(predicted_entities) > 0:
            for e in predicted_entities:
                converted_entity = self.label2presidio[e["entity_group"]]
                if converted_entity in entities or entities is None:
                    results.append(
                        RecognizerResult(
                            entity_type=converted_entity,
                            start=e["start"],
                            end=e["end"],
                            score=e["score"]
                        )
                    )
        return results


repository = "Jean-Baptiste/roberta-large-ner-english"
model_id = repository.split("/")[-1]
s3_location = f"s3://{sess.default_bucket()}/custom_inference/{model_id}/model.tar.gz"


# create Hugging Face Model Class
huggingface_model = HuggingFaceModel(
    model_data=s3_location,       # path to your model and script
    role=role,                    # iam role with permissions to create an Endpoint
    transformers_version="4.17",  # transformers version used
    pytorch_version="1.10",        # pytorch version used
    py_version='py38',            # python version used
)

# deploy the endpoint endpoint
predictor = huggingface_model.deploy(
    initial_instance_count=1,
    instance_type="ml.g4dn.xlarge"
)

payload = """
Hello, my name is David Johnson and I live in Maine.
I work as a software engineer at Amazon.
You can call me at (123) 456-7890.
My credit card number is 4095-2609-9393-4932 and my crypto wallet id is 16Yeky6GMjeNkAiNcBY7ZhrLoMSgg1BoyZ.

On September 18 I visited microsoft.com and sent an email to test@presidio.site, from the IP 192.168.0.1.
My passport: 191280342 and my phone number: (212) 555-1234.
This is a valid International Bank Account Number: IL150120690000003111111. Can you please check the status on bank account 954567876544?
Kate's social security number is 078-05-1126.  Her driver license? it is 1234567A.

"""

data = {
    "inputs": payload,
}

res = predictor.predict(data=data)
print(res)
