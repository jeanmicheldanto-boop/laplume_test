#!/usr/bin/env python3
"""
Script de test pour valider le pipeline de pseudonymisation
"""
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Exécute une commande et affiche le résultat"""
    print(f"\n🧪 {description}")
    print(f"📋 Commande: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode == 0:
        print("✅ Succès")
        if result.stdout:
            print(f"📤 Sortie:\n{result.stdout}")
    else:
        print("❌ Échec")
        if result.stderr:
            print(f"🚨 Erreur:\n{result.stderr}")
    
    return result.returncode == 0


def check_files():
    """Vérifie la présence des fichiers nécessaires"""
    print("🔍 Vérification des fichiers...")
    
    required_files = [
        "pseudonymize.py",
        "input/exemple_medical.txt",
        "gazetteer",
        "rules"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing:
        print(f"❌ Fichiers manquants: {missing}")
        return False
    
    return True


def test_basic_pseudonymization():
    """Test de pseudonymisation basique"""
    cmd = [
        sys.executable, "pseudonymize.py",
        "--input", "input/exemple_medical.txt",
        "--output", "output/test_basic.txt",
        "--mapping", "output/test_basic_mapping.json",
        "--rules", "rules/rules.yaml",
        "--log-level", "INFO"
    ]
    
    return run_command(cmd, "Test pseudonymisation hybride")


def test_depseudonymization():
    """Test de dépseudonymisation"""
    cmd = [
        sys.executable, "pseudonymize.py",
        "--input", "output/test_basic.txt",
        "--output", "output/test_restored.txt",
        "--load-mapping", "output/test_basic_mapping.json",
        "--depseudonymize"
    ]
    
    return run_command(cmd, "Test dépseudonymisation")


def verify_results():
    """Vérifie les résultats des tests"""
    print("\n🔍 Vérification des résultats...")
    
    # Vérifier que les fichiers ont été créés
    expected_files = [
        "output/test_basic.txt",
        "output/test_basic_mapping.json",
        "output/test_restored.txt"
    ]
    
    for file_path in expected_files:
        if Path(file_path).exists():
            size = Path(file_path).stat().st_size
            print(f"✅ {file_path} ({size} bytes)")
        else:
            print(f"❌ {file_path} - manquant")
            return False
    
    # Vérifier que le texte restauré est identique à l'original
    original = Path("input/exemple_medical.txt").read_text(encoding="utf-8")
    restored = Path("output/test_restored.txt").read_text(encoding="utf-8")
    
    if original.strip() == restored.strip():
        print("✅ Dépseudonymisation parfaite - texte restauré identique")
        return True
    else:
        print("⚠️ Différences entre texte original et restauré")
        print(f"Original: {len(original)} chars")
        print(f"Restauré: {len(restored)} chars")
        return False


def main():
    """Point d'entrée principal des tests"""
    print("🚀 Tests du pipeline de pseudonymisation")
    print("=" * 50)
    
    # Créer le dossier output s'il n'existe pas
    Path("output").mkdir(exist_ok=True)
    
    # Vérifications préliminaires
    if not check_files():
        print("\n❌ Tests interrompus - fichiers manquants")
        sys.exit(1)
    
    # Tests
    tests_passed = 0
    total_tests = 3
    
    if test_basic_pseudonymization():
        tests_passed += 1
    
    if test_depseudonymization():
        tests_passed += 1
    
    if verify_results():
        tests_passed += 1
    
    # Résultats finaux
    print("\n" + "=" * 50)
    print(f"📊 Résultats: {tests_passed}/{total_tests} tests réussis")
    
    if tests_passed == total_tests:
        print("🎉 Tous les tests sont passés avec succès!")
        sys.exit(0)
    else:
        print("⚠️ Certains tests ont échoué")
        sys.exit(1)


if __name__ == "__main__":
    main()