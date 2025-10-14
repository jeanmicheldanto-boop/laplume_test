# Installation du pipeline de pseudonymisation (Windows)
# =====================================================

Write-Host "🚀 Installation du pipeline de pseudonymisation..." -ForegroundColor Green

# Création de l'environnement virtuel
Write-Host "📦 Création de l'environnement virtuel..." -ForegroundColor Yellow
python -m venv venv_pseudonymize

# Activation de l'environnement
Write-Host "🔧 Activation de l'environnement virtuel..." -ForegroundColor Yellow
& .\venv_pseudonymize\Scripts\Activate.ps1

# Installation des dépendances
Write-Host "📚 Installation des dépendances..." -ForegroundColor Yellow
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "✅ Installation terminée !" -ForegroundColor Green
Write-Host ""
Write-Host "🔧 Pour activer l'environnement :" -ForegroundColor Cyan
Write-Host "   .\venv_pseudonymize\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "🧪 Pour tester le pipeline :" -ForegroundColor Cyan
Write-Host "   python pseudonymize.py --input test_categories.txt --output test_output.txt" -ForegroundColor White