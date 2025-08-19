"""
Unit tests for server security improvements.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.sanskrit_rewrite_engine.server import app


class TestServerSecurity:
    """Test security features of the FastAPI server."""
    
    def setup_method(self):
        """Set up test client."""
        self.client = TestClient(app)
    
    def test_security_headers_present(self):
        """Test that security headers are added to responses."""
        response = self.client.get("/health")
        
        # Check that all required security headers are present
        expected_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains"
        }
        
        for header, expected_value in expected_headers.items():
            assert header in response.headers
            assert response.headers[header] == expected_value
    
    def test_content_security_policy_header(self):
        """Test that CSP header is properly configured."""
        response = self.client.get("/health")
        
        csp_header = response.headers.get("Content-Security-Policy")
        assert csp_header is not None
        assert "default-src 'self'" in csp_header
        assert "script-src 'self' 'unsafe-inline'" in csp_header
        assert "style-src 'self' 'unsafe-inline'" in csp_header
    
    def test_request_id_generation(self):
        """Test that request IDs are generated and included in responses."""
        response = self.client.get("/health")
        
        # Check request ID in headers
        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        assert len(request_id) == 36  # UUID format
        assert "-" in request_id  # UUID contains hyphens
    
    def test_process_time_header(self):
        """Test that processing time is included in response headers."""
        response = self.client.get("/health")
        
        assert "X-Process-Time" in response.headers
        process_time = response.headers["X-Process-Time"]
        assert process_time.replace(".", "").isdigit()  # Should be a number
    
    def test_request_payload_validation_empty_text(self):
        """Test validation of empty text in process request."""
        response = self.client.post(
            "/api/process",
            json={"text": ""}
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_request_payload_validation_whitespace_only(self):
        """Test validation of whitespace-only text."""
        response = self.client.post(
            "/api/process",
            json={"text": "   "}
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_request_payload_validation_suspicious_content(self):
        """Test validation rejects suspicious content."""
        suspicious_payloads = [
            {"text": "<script>alert('xss')</script>"},
            {"text": "javascript:alert('xss')"},
            {"text": "data:text/html,<script>alert('xss')</script>"},
            {"text": "vbscript:msgbox('xss')"}
        ]
        
        for payload in suspicious_payloads:
            response = self.client.post("/api/process", json=payload)
            assert response.status_code == 422
            error_data = response.json()
            assert "Suspicious content detected" in str(error_data)
    
    def test_request_payload_validation_text_length(self):
        """Test validation of text length limits."""
        # Test text that's too long
        long_text = "a" * 10001
        response = self.client.post(
            "/api/process",
            json={"text": long_text}
        )
        
        assert response.status_code == 422
        error_data = response.json()
        assert "detail" in error_data
    
    def test_request_payload_validation_max_passes(self):
        """Test validation of max_passes parameter."""
        # Test invalid max_passes values
        invalid_values = [0, -1, 101, 1000]
        
        for invalid_value in invalid_values:
            response = self.client.post(
                "/api/process",
                json={"text": "test", "max_passes": invalid_value}
            )
            assert response.status_code == 422
    
    def test_request_payload_validation_rules_limit(self):
        """Test validation of rules parameter limits."""
        # Test too many rules
        too_many_rules = ["rule" + str(i) for i in range(51)]
        response = self.client.post(
            "/api/process",
            json={"text": "test", "rules": too_many_rules}
        )
        
        assert response.status_code == 422
    
    def test_cors_headers_present(self):
        """Test that CORS headers are properly configured."""
        # Test a simple GET request to check CORS headers
        response = self.client.get(
            "/health",
            headers={"Origin": "http://localhost:3000"}
        )
        
        # Should allow the request from localhost:3000
        assert response.status_code == 200
        # CORS headers are added by the middleware, check if origin is allowed
        # The actual CORS validation happens during preflight, but we can test basic functionality
    
    def test_error_responses_include_request_id(self):
        """Test that error responses include request IDs."""
        # Trigger a validation error
        response = self.client.post(
            "/api/process",
            json={"text": ""}
        )
        
        assert response.status_code == 422
        assert "X-Request-ID" in response.headers
        
        # For some error types, request_id might be in the response body
        # This depends on the specific error handler
    
    def test_valid_request_includes_request_id(self):
        """Test that valid requests include request ID in response."""
        response = self.client.post(
            "/api/process",
            json={"text": "test text"}
        )
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        
        # Check if request_id is in response body
        data = response.json()
        assert "request_id" in data
        assert data["request_id"] == response.headers["X-Request-ID"]
    
    def test_analyze_endpoint_security(self):
        """Test security features on analyze endpoint."""
        # Test valid request
        response = self.client.post(
            "/api/analyze",
            json={"text": "test analysis"}
        )
        
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers
        
        # Test invalid request
        response = self.client.post(
            "/api/analyze",
            json={"text": "<script>alert('xss')</script>"}
        )
        
        assert response.status_code == 422
    
    def test_content_length_validation(self):
        """Test that content length limits are enforced."""
        # This test would require sending a request with a large content-length header
        # For now, we'll test the logic indirectly through text length validation
        long_text = "a" * 10001
        response = self.client.post(
            "/api/process",
            json={"text": long_text}
        )
        
        # Should be rejected due to text length validation
        assert response.status_code == 422
    
    @pytest.mark.parametrize("endpoint", ["/", "/health", "/api/rules"])
    def test_security_headers_on_all_endpoints(self, endpoint):
        """Test that security headers are present on all endpoints."""
        response = self.client.get(endpoint)
        
        # All endpoints should have security headers
        assert "X-Content-Type-Options" in response.headers
        assert "X-Frame-Options" in response.headers
        assert "X-Request-ID" in response.headers
        assert "X-Process-Time" in response.headers