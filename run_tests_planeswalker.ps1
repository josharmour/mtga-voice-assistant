Write-Host "Running Helper Agent Tests in Voice Assistant Repo (VENV)..."
.\venv\Scripts\pytest tests/integration/test_dynamic_consolidated.py -v -s
if ($LASTEXITCODE -eq 0) {
    Write-Host "All agent tests passed successfully!" -ForegroundColor Green
}
else {
    Write-Host "Some tests failed." -ForegroundColor Red
}
