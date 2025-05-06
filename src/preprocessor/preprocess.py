import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.processing import SKLearnProcessor
import boto3
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, role: str, instance_type: str = 'ml.m5.xlarge'):
        self.role = role
        self.instance_type = instance_type
        self.sagemaker_session = sagemaker.Session()
        
    def run_preprocessing(self, 
                         input_data: str,
                         output_data: str,
                         code_path: str = 'preprocess.py'):
        """
        SageMaker Processing Jobを実行してデータの前処理を行います。
        
        Args:
            input_data: 入力データのS3パス
            output_data: 出力データのS3パス
            code_path: 前処理スクリプトのパス
        """
        sklearn_processor = SKLearnProcessor(
            framework_version='0.23-1',
            role=self.role,
            instance_type=self.instance_type,
            instance_count=1,
            base_job_name='data-preprocessing',
            sagemaker_session=self.sagemaker_session
        )
        
        sklearn_processor.run(
            code=code_path,
            inputs=[
                ProcessingInput(
                    source=input_data,
                    destination='/opt/ml/processing/input'
                )
            ],
            outputs=[
                ProcessingOutput(
                    source='/opt/ml/processing/output',
                    destination=output_data
                )
            ],
            arguments=['--input-data', '/opt/ml/processing/input',
                      '--output-data', '/opt/ml/processing/output']
        )

def main():
    # 環境変数から設定を読み込み
    role = os.environ.get('SAGEMAKER_ROLE')
    input_data = os.environ.get('INPUT_DATA')
    output_data = os.environ.get('OUTPUT_DATA')
    
    if not all([role, input_data, output_data]):
        raise ValueError("Required environment variables are missing")
    
    # 前処理の実行
    preprocessor = DataPreprocessor(role=role)
    preprocessor.run_preprocessing(
        input_data=input_data,
        output_data=output_data
    )
    
    logger.info("Preprocessing job completed successfully")

if __name__ == "__main__":
    main() 