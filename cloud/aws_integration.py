"""
AWS cloud service integration for Vidya Quantum Interface
"""
import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from config import config

logger = logging.getLogger(__name__)


class AWSIntegration:
    """AWS cloud services integration"""
    
    def __init__(self):
        self.region = config.aws_region if hasattr(config, 'aws_region') else 'us-west-2'
        self.s3_bucket = config.aws_s3_bucket if hasattr(config, 'aws_s3_bucket') else None
        self.cloudwatch_log_group = config.aws_cloudwatch_log_group if hasattr(config, 'aws_cloudwatch_log_group') else None
        
        # Initialize AWS clients
        self._s3_client = None
        self._lambda_client = None
        self._bedrock_client = None
        self._cloudwatch_client = None
        self._sqs_client = None
    
    @property
    def s3_client(self):
        """Lazy initialization of S3 client"""
        if self._s3_client is None:
            try:
                self._s3_client = boto3.client('s3', region_name=self.region)
            except NoCredentialsError:
                logger.warning("AWS credentials not found, S3 functionality disabled")
                return None
        return self._s3_client
    
    @property
    def lambda_client(self):
        """Lazy initialization of Lambda client"""
        if self._lambda_client is None:
            try:
                self._lambda_client = boto3.client('lambda', region_name=self.region)
            except NoCredentialsError:
                logger.warning("AWS credentials not found, Lambda functionality disabled")
                return None
        return self._lambda_client
    
    @property
    def bedrock_client(self):
        """Lazy initialization of Bedrock client"""
        if self._bedrock_client is None:
            try:
                self._bedrock_client = boto3.client('bedrock-runtime', region_name=self.region)
            except NoCredentialsError:
                logger.warning("AWS credentials not found, Bedrock functionality disabled")
                return None
        return self._bedrock_client
    
    @property
    def cloudwatch_client(self):
        """Lazy initialization of CloudWatch client"""
        if self._cloudwatch_client is None:
            try:
                self._cloudwatch_client = boto3.client('logs', region_name=self.region)
            except NoCredentialsError:
                logger.warning("AWS credentials not found, CloudWatch functionality disabled")
                return None
        return self._cloudwatch_client
    
    @property
    def sqs_client(self):
        """Lazy initialization of SQS client"""
        if self._sqs_client is None:
            try:
                self._sqs_client = boto3.client('sqs', region_name=self.region)
            except NoCredentialsError:
                logger.warning("AWS credentials not found, SQS functionality disabled")
                return None
        return self._sqs_client
    
    async def invoke_lambda_function(
        self, 
        function_name: str, 
        payload: Dict[str, Any],
        invocation_type: str = 'RequestResponse'
    ) -> Optional[Dict[str, Any]]:
        """Invoke AWS Lambda function for scalable processing"""
        
        if not self.lambda_client:
            logger.error("Lambda client not available")
            return None
        
        try:
            response = self.lambda_client.invoke(
                FunctionName=function_name,
                InvocationType=invocation_type,
                Payload=json.dumps(payload)
            )
            
            if invocation_type == 'RequestResponse':
                result = json.loads(response['Payload'].read())
                return result
            else:
                return {"status": "invoked", "request_id": response['ResponseMetadata']['RequestId']}
                
        except ClientError as e:
            logger.error(f"Error invoking Lambda function {function_name}: {e}")
            return None
    
    async def process_sanskrit_with_bedrock(
        self, 
        text: str, 
        model_id: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    ) -> Optional[str]:
        """Process Sanskrit text using AWS Bedrock"""
        
        if not self.bedrock_client:
            logger.error("Bedrock client not available")
            return None
        
        try:
            prompt = f"""
            Analyze this Sanskrit text and provide morphological analysis:
            
            Text: {text}
            
            Please provide:
            1. Word-by-word breakdown
            2. Root analysis
            3. Grammatical forms
            4. Meaning and context
            """
            
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            })
            
            response = self.bedrock_client.invoke_model(
                body=body,
                modelId=model_id,
                accept='application/json',
                contentType='application/json'
            )
            
            result = json.loads(response.get('body').read())
            return result.get('content', [{}])[0].get('text', '')
            
        except ClientError as e:
            logger.error(f"Error processing with Bedrock: {e}")
            return None
    
    async def store_analysis_result(
        self, 
        key: str, 
        data: Dict[str, Any]
    ) -> bool:
        """Store analysis results in S3"""
        
        if not self.s3_client or not self.s3_bucket:
            logger.error("S3 client or bucket not available")
            return False
        
        try:
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=f"analysis/{key}.json",
                Body=json.dumps(data),
                ContentType='application/json'
            )
            return True
            
        except ClientError as e:
            logger.error(f"Error storing to S3: {e}")
            return False
    
    async def retrieve_analysis_result(self, key: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis results from S3"""
        
        if not self.s3_client or not self.s3_bucket:
            logger.error("S3 client or bucket not available")
            return None
        
        try:
            response = self.s3_client.get_object(
                Bucket=self.s3_bucket,
                Key=f"analysis/{key}.json"
            )
            data = json.loads(response['Body'].read())
            return data
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'NoSuchKey':
                logger.info(f"Analysis result not found: {key}")
            else:
                logger.error(f"Error retrieving from S3: {e}")
            return None
    
    async def send_processing_task(
        self, 
        queue_url: str, 
        task_data: Dict[str, Any]
    ) -> bool:
        """Send processing task to SQS queue"""
        
        if not self.sqs_client:
            logger.error("SQS client not available")
            return False
        
        try:
            self.sqs_client.send_message(
                QueueUrl=queue_url,
                MessageBody=json.dumps(task_data),
                MessageAttributes={
                    'TaskType': {
                        'StringValue': task_data.get('type', 'unknown'),
                        'DataType': 'String'
                    }
                }
            )
            return True
            
        except ClientError as e:
            logger.error(f"Error sending SQS message: {e}")
            return False
    
    async def log_to_cloudwatch(
        self, 
        log_stream: str, 
        messages: List[Dict[str, Any]]
    ) -> bool:
        """Send logs to CloudWatch"""
        
        if not self.cloudwatch_client or not self.cloudwatch_log_group:
            logger.error("CloudWatch client or log group not available")
            return False
        
        try:
            # Ensure log stream exists
            try:
                self.cloudwatch_client.create_log_stream(
                    logGroupName=self.cloudwatch_log_group,
                    logStreamName=log_stream
                )
            except ClientError as e:
                if e.response['Error']['Code'] != 'ResourceAlreadyExistsException':
                    raise
            
            # Send log events
            self.cloudwatch_client.put_log_events(
                logGroupName=self.cloudwatch_log_group,
                logStreamName=log_stream,
                logEvents=messages
            )
            return True
            
        except ClientError as e:
            logger.error(f"Error logging to CloudWatch: {e}")
            return False


class ScalableAIProcessor:
    """Scalable AI processing using cloud services"""
    
    def __init__(self):
        self.aws = AWSIntegration()
        self.processing_queue = asyncio.Queue()
        self.results_cache = {}
    
    async def process_batch_sanskrit(
        self, 
        texts: List[str], 
        use_cloud: bool = True
    ) -> List[Dict[str, Any]]:
        """Process multiple Sanskrit texts in parallel"""
        
        if use_cloud and self.aws.lambda_client:
            # Use Lambda for parallel processing
            tasks = []
            for i, text in enumerate(texts):
                task = self.aws.invoke_lambda_function(
                    function_name='vidya-sanskrit-processor',
                    payload={
                        'text': text,
                        'batch_id': f"batch_{i}",
                        'timestamp': asyncio.get_event_loop().time()
                    }
                )
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]
        
        else:
            # Fallback to local processing
            from vidya_quantum_interface.sanskrit_adapter import SanskritAdapter
            adapter = SanskritAdapter()
            
            tasks = [adapter.process_text(text) for text in texts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return [r for r in results if not isinstance(r, Exception)]
    
    async def queue_processing_task(
        self, 
        task_type: str, 
        data: Dict[str, Any]
    ) -> str:
        """Queue a processing task for background execution"""
        
        task_id = f"{task_type}_{asyncio.get_event_loop().time()}"
        
        if config.cloud_provider == "aws":
            # Use SQS for cloud queuing
            queue_url = f"https://sqs.{self.aws.region}.amazonaws.com/account/vidya-processing-queue"
            success = await self.aws.send_processing_task(queue_url, {
                'task_id': task_id,
                'type': task_type,
                'data': data
            })
            
            if success:
                return task_id
        
        # Fallback to local queue
        await self.processing_queue.put({
            'task_id': task_id,
            'type': task_type,
            'data': data
        })
        
        return task_id
    
    async def get_processing_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a processing task"""
        
        # Check local cache first
        if task_id in self.results_cache:
            return self.results_cache[task_id]
        
        # Check cloud storage
        if config.cloud_provider == "aws":
            result = await self.aws.retrieve_analysis_result(task_id)
            if result:
                self.results_cache[task_id] = result
                return result
        
        return None


# Global instances
aws_integration = AWSIntegration()
scalable_processor = ScalableAIProcessor()