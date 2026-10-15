# PowerShell Environment Setup Script
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Dental Caries Detection using UNet++ - Setup Script  " -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan

# Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "Creating Python virtual environment in .\venv..." -ForegroundColor Yellow
    python -m venv venv
}

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

Write-Host "Installing dependencies from requirements.txt..." -ForegroundColor Yellow
pip install -r requirements.txt

Write-Host "`nEnvironment setup finished!" -ForegroundColor Green
Write-Host "To start the interactive web application, run:" -ForegroundColor White
Write-Host "  python web_app\app.py`n" -ForegroundColor Cyan
