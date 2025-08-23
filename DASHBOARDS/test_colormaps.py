#!/usr/bin/env python3
"""
Script para testar diferentes colormaps no classification report
"""

import numpy as np
import matplotlib.pyplot as plt
from generate_multiclass_plots import MulticlassVisualizationGenerator

def test_different_colormaps():
    """Testa diferentes colormaps para o classification report"""
    
    # Gerar dados sintéticos
    generator = MulticlassVisualizationGenerator()
    y_true, y_pred = generator.create_synthetic_cdr_predictions(100)
    class_names = ['CDR=0.0', 'CDR=0.5', 'CDR=1.0', 'CDR=2.0']
    
    # Lista de colormaps para testar
    colormaps = [
        ('RdYlGn', 'Vermelho-Amarelo-Verde (Original)'),
        ('viridis', 'Viridis (Científico)'),
        ('plasma', 'Plasma (Roxo-Rosa-Amarelo)'),
        ('Blues', 'Tons de Azul'),
        ('coolwarm', 'Azul-Branco-Vermelho'),
        ('Spectral', 'Arco-íris'),
        ('Greens', 'Tons de Verde'),
    ]
    
    print("Testando diferentes colormaps para o Classification Report...")
    
    for i, (cmap_name, description) in enumerate(colormaps):
        print(f"Testando {i+1}/{len(colormaps)}: {cmap_name} - {description}")
        
        # Criar nome do arquivo
        save_path = f"classification_report_{cmap_name.lower()}.png"
        
        # Gerar com colormap específico
        generator.generate_classification_report_plot_custom_color(
            y_true, y_pred, class_names, save_path, cmap_name
        )
    
    print(f"\n✅ {len(colormaps)} versões geradas!")
    print("Arquivos criados:")
    for cmap_name, _ in colormaps:
        print(f"   - classification_report_{cmap_name.lower()}.png")

if __name__ == "__main__":
    test_different_colormaps()
