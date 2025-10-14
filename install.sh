#!/bin/bash
# Installation du pipeline de pseudonymisation
# ============================================

echo "🚀 Installation du pipeline de pseudonymisation..."

# Création de l'environnement virtuel
echo "📦 Création de l'environnement virtuel..."
python -m venv venv_pseudonymize

# Activation selon l'OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    source venv_pseudonymize/Scripts/activate
else
    # Linux/Mac
    source venv_pseudonymize/bin/activate
fi

# Installation des dépendances
echo "📚 Installation des dépendances..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Installation terminée !"
echo ""
echo "🔧 Pour activer l'environnement :"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo "   venv_pseudonymize\\Scripts\\Activate.ps1"
else
    echo "   source venv_pseudonymize/bin/activate"
fi
echo ""
echo "🧪 Pour tester le pipeline :"
echo "   python pseudonymize.py --input test_categories.txt --output test_output.txt"