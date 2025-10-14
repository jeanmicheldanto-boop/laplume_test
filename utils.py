#!/usr/bin/env python3
"""
Utilitaires pour le pipeline de pseudonymisation
"""
import json
import argparse
from pathlib import Path


def analyze_mapping(mapping_file: str):
    """Analyse un fichier de mapping et affiche les statistiques"""
    with open(mapping_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"📊 Analyse du mapping: {mapping_file}")
    print("=" * 50)
    
    # Statistiques générales
    stats = data.get('stats', {})
    print(f"📈 Total entités: {stats.get('total_entities', 0)}")
    
    per_type = stats.get('per_type', {})
    for entity_type, count in per_type.items():
        print(f"   - {entity_type}: {count}")
    
    # Top 10 des entités les plus fréquentes
    forward = data.get('forward', {})
    if forward:
        print(f"\n🔝 Échantillon d'entités détectées:")
        for i, (original, pseudo) in enumerate(list(forward.items())[:10]):
            print(f"   {i+1}. '{original}' → {pseudo}")
        
        if len(forward) > 10:
            print(f"   ... et {len(forward) - 10} autres")


def compare_files(file1: str, file2: str):
    """Compare deux fichiers texte"""
    print(f"🔍 Comparaison: {file1} vs {file2}")
    print("=" * 50)
    
    text1 = Path(file1).read_text(encoding='utf-8')
    text2 = Path(file2).read_text(encoding='utf-8')
    
    print(f"📏 Taille fichier 1: {len(text1)} caractères")
    print(f"📏 Taille fichier 2: {len(text2)} caractères")
    
    if text1.strip() == text2.strip():
        print("✅ Les fichiers sont identiques")
    else:
        print("⚠️ Les fichiers diffèrent")
        
        # Analyser les différences
        lines1 = text1.splitlines()
        lines2 = text2.splitlines()
        
        diff_lines = 0
        for i, (line1, line2) in enumerate(zip(lines1, lines2)):
            if line1 != line2:
                diff_lines += 1
                if diff_lines <= 3:  # Afficher les 3 premières différences
                    print(f"   Ligne {i+1}:")
                    print(f"     1: {line1[:100]}...")
                    print(f"     2: {line2[:100]}...")
        
        if diff_lines > 3:
            print(f"   ... et {diff_lines - 3} autres lignes différentes")


def list_entities_in_text(text_file: str, entity_types: list = None):
    """Liste toutes les entités pseudonymisées dans un texte"""
    import re
    
    if entity_types is None:
        entity_types = ['PER', 'ORG', 'LOC']
    
    text = Path(text_file).read_text(encoding='utf-8')
    
    print(f"🔍 Entités pseudonymisées dans: {text_file}")
    print("=" * 50)
    
    for entity_type in entity_types:
        pattern = rf'<{entity_type}_\d+>'
        matches = re.findall(pattern, text)
        unique_matches = sorted(set(matches))
        
        print(f"{entity_type}: {len(unique_matches)} entités uniques")
        if unique_matches:
            print(f"   {', '.join(unique_matches[:10])}")
            if len(unique_matches) > 10:
                print(f"   ... et {len(unique_matches) - 10} autres")


def main():
    parser = argparse.ArgumentParser(description="Utilitaires pour analyser les résultats de pseudonymisation")
    subparsers = parser.add_subparsers(dest='command', help='Commandes disponibles')
    
    # Analyser un mapping
    analyze_parser = subparsers.add_parser('analyze', help='Analyser un fichier de mapping')
    analyze_parser.add_argument('mapping_file', help='Fichier de mapping JSON')
    
    # Comparer deux fichiers
    compare_parser = subparsers.add_parser('compare', help='Comparer deux fichiers texte')
    compare_parser.add_argument('file1', help='Premier fichier')
    compare_parser.add_argument('file2', help='Deuxième fichier')
    
    # Lister les entités
    entities_parser = subparsers.add_parser('entities', help='Lister les entités dans un texte pseudonymisé')
    entities_parser.add_argument('text_file', help='Fichier texte pseudonymisé')
    entities_parser.add_argument('--types', nargs='+', default=['PER', 'ORG', 'LOC'], help='Types d\'entités à chercher')
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_mapping(args.mapping_file)
    elif args.command == 'compare':
        compare_files(args.file1, args.file2)
    elif args.command == 'entities':
        list_entities_in_text(args.text_file, args.types)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()