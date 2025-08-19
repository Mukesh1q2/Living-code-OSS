@echo off
echo Testing Sanskrit Rewrite Engine API with curl
echo =============================================

echo.
echo 1. Testing API info...
curl -s http://localhost:8000/ | python -m json.tool

echo.
echo 2. Testing health check...
curl -s http://localhost:8000/health | python -m json.tool

echo.
echo 3. Testing rules...
curl -s http://localhost:8000/api/rules | python -m json.tool

echo.
echo 4. Testing text processing...
curl -s -X POST http://localhost:8000/api/process ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"rāma + iti\", \"enable_tracing\": true}" | python -m json.tool

echo.
echo 5. Testing text analysis...
curl -s -X POST http://localhost:8000/api/analyze ^
  -H "Content-Type: application/json" ^
  -d "{\"text\": \"deva + rāja\"}" | python -m json.tool

echo.
echo Testing completed!
pause