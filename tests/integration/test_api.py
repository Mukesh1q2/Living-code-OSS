"""
Comprehensive integration tests for the Sanskrit Rewrite Engine API.

This module tests the FastAPI web server endpoints using TestClient,
ensuring proper request/response handling, validation, and error cases.
"""

import pytest
import json
import time
from fastapi.testclient import TestClient
from unittest.mock import patch, Mock

from src.sanskrit_rewrite_engine.server import app
from tests.fixtures.sample_texts import (
    BASIC_WORDS, MORPHOLOGICAL_EXAMPLES, SANDHI_EXAMPLES, ERROR_CASES
)


class TestAPIEndpoints:
    """Test FastAPI endpoints using TestClient."""
    
    def setup_method(self):
        """Set up test client before each test."""
        self.client = TestClient(app)
        
    def test_root_endpoint(self):
        """Test root endpoint returns API information."""
        response = self.client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "message" in data
        assert "version" in data
        assert "status" in data
        assert data["status"] == "running"
        assert "endpoints" in data
        assert "documentation" in data
        
    def test_health_check_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "version" in data
        assert "timestamp" in data
        assert "components" in data
        
        # Check component status format
        components = data["components"]
        assert "api_server" in components
        assert "sanskrit_processor" in components
        assert "rule_engine" in components
        assert "tokenizer" in components
        
    def test_rules_endpoint(self):
        """Test rules listing endpoint."""
        response = self.client.get("/rules")
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "count" in data
        assert "rules" in data
        assert data["success"] == True
        assert isinstance(data["rules"], list)
        assert data["count"] == len(data["rules"])
        
        # Check rule structure if rules exist
        if data["rules"]:
            rule = data["rules"][0]
            assert "id" in rule
            assert "name" in rule
            assert "description" in rule
            
    def test_process_endpoint_basic(self):
        """Test basic text processing endpoint."""
        request_data = {
            "text": "test text",
            "enable_tracing": False
        }
        
        response = self.client.post("/process", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "input" in data
        assert "output" in data
        assert "converged" in data
        assert "passes" in data
        assert "transformations" in data
        assert "processing_time" in data
        assert "timestamp" in data
        
        assert data["success"] == True
        assert data["input"] == "test text"
        assert isinstance(data["output"], str)
        
    def test_process_endpoint_with_tracing(self):
        """Test text processing with tracing enabled."""
        request_data = {
            "text": "rāma + iti",
            "enable_tracing": True
        }
        
        response = self.client.post("/process", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        assert "traces" in data
        assert isinstance(data["traces"], list)
        
    def test_process_endpoint_with_sanskrit_text(self):
        """Test processing with various Sanskrit texts."""
        test_cases = [
            "rāma + iti",
            "deva + indra", 
            "guru + upadeśa",
            "dharma + artha"
        ]
        
        for text in test_cases:
            request_data = {
                "text": text,
                "enable_tracing": True
            }
            
            response = self.client.post("/process", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] == True
            assert data["input"] == text
            
    def test_process_endpoint_validation_errors(self):
        """Test process endpoint validation errors."""
        # Test empty text
        response = self.client.post("/process", json={"text": ""})
        assert response.status_code == 422
        
        # Test missing text field
        response = self.client.post("/process", json={})
        assert response.status_code == 422
        
        # Test text too long
        long_text = "a" * 20000
        response = self.client.post("/process", json={"text": long_text})
        assert response.status_code == 400
        
        # Test invalid max_passes
        response = self.client.post("/process", json={
            "text": "test",
            "max_passes": -1
        })
        assert response.status_code == 422
        
    def test_process_endpoint_security_validation(self):
        """Test security validation in process endpoint."""
        # Test script injection attempts
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "data:text/html,<script>alert('xss')</script>",
            "vbscript:msgbox('xss')"
        ]
        
        for malicious_input in malicious_inputs:
            response = self.client.post("/process", json={"text": malicious_input})
            assert response.status_code == 422
            
    def test_analyze_endpoint_basic(self):
        """Test basic text analysis endpoint."""
        request_data = {
            "text": "rāma + iti"
        }
        
        response = self.client.post("/api/analyze", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert "success" in data
        assert "input" in data
        assert "analysis" in data
        assert "timestamp" in data
        
        assert data["success"] == True
        assert data["input"] == "rāma + iti"
        
        # Check analysis structure
        analysis = data["analysis"]
        assert "basic_stats" in analysis
        
    def test_analyze_endpoint_validation_errors(self):
        """Test analyze endpoint validation errors."""
        # Test empty text
        response = self.client.post("/api/analyze", json={"text": ""})
        assert response.status_code == 422
        
        # Test missing text field
        response = self.client.post("/api/analyze", json={})
        assert response.status_code == 422
        
        # Test text too long
        long_text = "a" * 20000
        response = self.client.post("/api/analyze", json={"text": long_text})
        assert response.status_code == 400
        
    def test_cors_headers(self):
        """Test CORS headers are properly set."""
        response = self.client.options("/process")
        
        # Should have CORS headers
        assert "access-control-allow-origin" in response.headers
        assert "access-control-allow-methods" in response.headers
        assert "access-control-allow-headers" in response.headers
        
    def test_security_headers(self):
        """Test security headers are properly set."""
        response = self.client.get("/")
        
        # Check for security headers
        assert "x-content-type-options" in response.headers
        assert "x-frame-options" in response.headers
        assert "x-xss-protection" in response.headers
        assert "referrer-policy" in response.headers
        assert "content-security-policy" in response.headers
        
    def test_request_id_header(self):
        """Test that request ID header is added to responses."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        assert "x-request-id" in response.headers
        
        # Request ID should be a valid UUID format
        request_id = response.headers["x-request-id"]
        assert len(request_id) == 36  # UUID length
        assert request_id.count("-") == 4  # UUID format
        
    def test_process_time_header(self):
        """Test that process time header is added to responses."""
        response = self.client.get("/health")
        
        assert response.status_code == 200
        assert "x-process-time" in response.headers
        
        # Process time should be a valid float
        process_time = float(response.headers["x-process-time"])
        assert process_time >= 0
        
    def test_error_handling_404(self):
        """Test 404 error handling."""
        response = self.client.get("/nonexistent-endpoint")
        
        assert response.status_code == 404
        data = response.json()
        
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data
        
    def test_error_handling_405(self):
        """Test 405 method not allowed error handling."""
        response = self.client.put("/health")
        
        assert response.status_code == 405
        
    def test_error_handling_500(self):
        """Test 500 internal server error handling."""
        # Mock the engine to raise an exception
        with patch('sanskrit_rewrite_engine.server.engine.process', side_effect=Exception("Test error")):
            response = self.client.post("/process", json={"text": "test"})
            
            assert response.status_code == 500
            data = response.json()
            
            assert "error" in data
            assert "message" in data
            assert "timestamp" in data
            
    def test_request_size_limit(self):
        """Test request size limits."""
        # Create a very large request
        large_data = {"text": "a" * 15000}  # Larger than MAX_TEXT_LENGTH
        
        response = self.client.post("/process", json=large_data)
        
        # Should be rejected due to size
        assert response.status_code == 400
        
    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        import threading
        import queue
        
        results = queue.Queue()
        
        def make_request(text):
            response = self.client.post("/process", json={"text": f"test {text}"})
            results.put(response.status_code)
            
        # Create multiple threads
        threads = []
        for i in range(10):
            thread = threading.Thread(target=make_request, args=(i,))
            threads.append(thread)
            thread.start()
            
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
            
        # Check all requests succeeded
        status_codes = []
        while not results.empty():
            status_codes.append(results.get())
            
        assert len(status_codes) == 10
        assert all(code == 200 for code in status_codes)
        
    def test_api_performance(self):
        """Test API response times."""
        # Test multiple endpoints for performance
        endpoints = [
            ("GET", "/"),
            ("GET", "/health"),
            ("GET", "/rules"),
            ("POST", "/process", {"text": "test"}),
            ("POST", "/api/analyze", {"text": "test"})
        ]
        
        for method, path, *data in endpoints:
            start_time = time.time()
            
            if method == "GET":
                response = self.client.get(path)
            elif method == "POST":
                response = self.client.post(path, json=data[0] if data else {})
                
            end_time = time.time()
            response_time = end_time - start_time
            
            # Response should be fast (less than 1 second)
            assert response_time < 1.0
            assert response.status_code in [200, 422]  # 422 for invalid POST data
            
    def test_content_type_validation(self):
        """Test content type validation."""
        # Test with wrong content type
        response = self.client.post(
            "/process",
            data="text=test",  # Form data instead of JSON
            headers={"content-type": "application/x-www-form-urlencoded"}
        )
        
        # Should reject non-JSON content
        assert response.status_code == 422
        
    def test_api_documentation_endpoints(self):
        """Test API documentation endpoints."""
        # Test OpenAPI JSON
        response = self.client.get("/openapi.json")
        assert response.status_code == 200
        
        openapi_data = response.json()
        assert "openapi" in openapi_data
        assert "info" in openapi_data
        assert "paths" in openapi_data
        
        # Test Swagger UI
        response = self.client.get("/docs")
        assert response.status_code == 200
        
        # Test ReDoc
        response = self.client.get("/redoc")
        assert response.status_code == 200
        
    def test_api_versioning(self):
        """Test API version consistency."""
        # Check version in root endpoint
        response = self.client.get("/")
        root_data = response.json()
        
        # Check version in health endpoint
        response = self.client.get("/health")
        health_data = response.json()
        
        # Versions should match
        assert root_data["version"] == health_data["version"]
        
    def test_request_logging(self):
        """Test that requests are properly logged."""
        with patch('sanskrit_rewrite_engine.server.logger') as mock_logger:
            response = self.client.get("/health")
            
            assert response.status_code == 200
            
            # Should have logged request start and completion
            assert mock_logger.info.call_count >= 2
            
    def test_error_response_format(self):
        """Test error response format consistency."""
        # Test validation error
        response = self.client.post("/process", json={})
        assert response.status_code == 422
        
        data = response.json()
        assert "detail" in data  # FastAPI validation error format
        
        # Test custom error
        response = self.client.get("/nonexistent")
        assert response.status_code == 404
        
        data = response.json()
        assert "error" in data
        assert "message" in data
        assert "timestamp" in data


class TestAPIIntegrationWithEngine:
    """Test API integration with the actual engine."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
        
    def test_end_to_end_processing(self):
        """Test end-to-end text processing through API."""
        request_data = {
            "text": "rāma + iti",
            "enable_tracing": True,
            "max_passes": 10
        }
        
        response = self.client.post("/process", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        # Should successfully process Sanskrit text
        assert data["success"] == True
        assert data["input"] == "rāma + iti"
        assert isinstance(data["output"], str)
        assert isinstance(data["transformations"], dict)
        assert isinstance(data["traces"], list)
        
    def test_rule_application_through_api(self):
        """Test that rules are actually applied through API."""
        # Test with text that should trigger transformations
        test_cases = [
            ("rāma + iti", "rāmeti"),  # Should apply vowel sandhi
            ("deva + indra", "devendra"),  # Should apply vowel sandhi
        ]
        
        for input_text, expected_pattern in test_cases:
            request_data = {
                "text": input_text,
                "enable_tracing": True
            }
            
            response = self.client.post("/process", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            # Should have applied some transformation
            assert data["success"] == True
            # Output should be different from input (transformation applied)
            # or contain expected pattern
            assert (data["output"] != data["input"] or 
                   expected_pattern in data["output"])
                   
    def test_analysis_integration(self):
        """Test text analysis integration."""
        request_data = {
            "text": "rāma + iti dharma + artha"
        }
        
        response = self.client.post("/api/analyze", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["success"] == True
        analysis = data["analysis"]
        
        # Should detect basic properties
        basic_stats = analysis["basic_stats"]
        assert basic_stats["word_count"] > 0
        assert basic_stats["contains_markers"] == True  # Has + markers
        
    def test_performance_with_real_engine(self):
        """Test API performance with real engine processing."""
        # Test with various text sizes
        test_texts = [
            "short",
            "medium length text with some words",
            "longer text " * 20,
            "rāma + iti " * 50  # Repeated Sanskrit with transformations
        ]
        
        for text in test_texts:
            start_time = time.time()
            
            response = self.client.post("/process", json={
                "text": text,
                "enable_tracing": False  # Disable tracing for performance
            })
            
            end_time = time.time()
            response_time = end_time - start_time
            
            assert response.status_code == 200
            # Should complete within reasonable time
            assert response_time < 2.0  # 2 seconds max
            
            data = response.json()
            assert data["success"] == True