#!/usr/bin/env python3
"""
ANALISADOR FASTSURFER COM OPENAI GPT
====================================

Sistema de interpretação clínica dos resultados do FastSurfer usando ChatGPT
Substitui/Complementa a CNN para análise interpretativa dos dados

Funcionalidades:
1. Extração de métricas FastSurfer
2. Análise interpretativa com GPT-4
3. Relatórios clínicos automatizados
4. Detecção de padrões anômalos
5. Recomendações clínicas baseadas em IA

Autor: Raphael Sales - TCC Alzheimer
"""

import os
import json
import pandas as pd
import numpy as np
import openai
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class FastSurferDataExtractor:
    """
    Extrator de dados do FastSurfer para análise com OpenAI
    """
    
    def __init__(self, fastsurfer_dir: str):
        self.fastsurfer_dir = fastsurfer_dir
        self.subjects_data = {}
        
    def extract_subject_metrics(self, subject_id: str) -> Dict:
        """
        Extrai métricas completas de um sujeito do FastSurfer
        """
        subject_dir = os.path.join(self.fastsurfer_dir, subject_id)
        
        if not os.path.exists(subject_dir):
            return None
            
        metrics = {
            'subject_id': subject_id,
            'timestamp': datetime.now().isoformat(),
            'volumes': {},
            'cortical': {},
            'subcortical': {},
            'quality_metrics': {}
        }
        
        # 1. Extrair volumes subcorticais (aseg.stats)
        aseg_file = os.path.join(subject_dir, 'stats', 'aseg.stats')
        if os.path.exists(aseg_file):
            metrics['subcortical'] = self._parse_aseg_stats(aseg_file)
        
        # 2. Extrair métricas corticais (aparc.stats)
        for hemisphere in ['lh', 'rh']:
            aparc_file = os.path.join(subject_dir, 'stats', f'{hemisphere}.aparc.stats')
            if os.path.exists(aparc_file):
                metrics['cortical'][hemisphere] = self._parse_aparc_stats(aparc_file)
        
        # 3. Extrair métricas de qualidade
        metrics['quality_metrics'] = self._extract_quality_metrics(subject_dir)
        
        return metrics
    
    def _parse_aseg_stats(self, aseg_file: str) -> Dict:
        """Parse arquivo aseg.stats do FastSurfer"""
        volumes = {}
        
        try:
            with open(aseg_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        structure = parts[4]  # Nome da estrutura
                        volume = float(parts[3])  # Volume em mm³
                        volumes[structure] = volume
                        
        except Exception as e:
            print(f"Erro ao parse aseg.stats: {e}")
        
        return volumes
    
    def _parse_aparc_stats(self, aparc_file: str) -> Dict:
        """Parse arquivo aparc.stats do FastSurfer"""
        cortical_metrics = {}
        
        try:
            with open(aparc_file, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        region = parts[0]  # Nome da região
                        thickness = float(parts[4])  # Espessura média
                        area = float(parts[2])  # Área de superfície
                        volume = float(parts[3])  # Volume
                        
                        cortical_metrics[region] = {
                            'thickness': thickness,
                            'area': area,
                            'volume': volume
                        }
                        
        except Exception as e:
            print(f"Erro ao parse aparc.stats: {e}")
        
        return cortical_metrics
    
    def _extract_quality_metrics(self, subject_dir: str) -> Dict:
        """Extrai métricas de qualidade do processamento"""
        quality = {}
        
        # Verificar arquivos críticos
        critical_files = [
            'mri/T1.mgz',
            'mri/aparc+aseg.mgz',
            'mri/aseg.mgz',
            'stats/aseg.stats'
        ]
        
        for file_path in critical_files:
            full_path = os.path.join(subject_dir, file_path)
            quality[f"has_{file_path.replace('/', '_')}"] = os.path.exists(full_path)
        
        # Verificar tamanhos dos arquivos
        for file_path in critical_files:
            full_path = os.path.join(subject_dir, file_path)
            if os.path.exists(full_path):
                size = os.path.getsize(full_path)
                quality[f"size_{file_path.replace('/', '_')}"] = size
        
        return quality
    
    def extract_all_subjects(self, max_subjects: Optional[int] = None) -> pd.DataFrame:
        """
        Extrai métricas de todos os sujeitos disponíveis
        """
        print("🔍 Extraindo métricas FastSurfer para análise OpenAI...")
        
        subjects = []
        count = 0
        
        for item in os.listdir(self.fastsurfer_dir):
            if item.startswith('OAS1_') and os.path.isdir(os.path.join(self.fastsurfer_dir, item)):
                if max_subjects and count >= max_subjects:
                    break
                    
                metrics = self.extract_subject_metrics(item)
                if metrics:
                    subjects.append(metrics)
                    count += 1
                    
                if count % 50 == 0:
                    print(f"  Processados: {count} sujeitos")
        
        print(f"✅ Extração concluída: {len(subjects)} sujeitos")
        return pd.DataFrame(subjects)

class OpenAIFastSurferAnalyzer:
    """
    Analisador de dados FastSurfer usando OpenAI GPT
    """
    
    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.analysis_history = []
        
    def analyze_subject(self, subject_metrics: Dict) -> Dict:
        """
        Analisa métricas de um sujeito usando GPT
        """
        
        # Preparar prompt para análise
        prompt = self._create_analysis_prompt(subject_metrics)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": """Você é um neurocientista especialista em análise de neuroimagem e Alzheimer. 
                        Analise os dados do FastSurfer fornecidos e forneça:
                        1. Interpretação clínica das métricas
                        2. Identificação de padrões anômalos
                        3. Risco de comprometimento cognitivo
                        4. Recomendações clínicas
                        5. Comparação com valores normativos
                        
                        Responda em português brasileiro, de forma clara e profissional."""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.3,
                max_tokens=1500
            )
            
            analysis = {
                'subject_id': subject_metrics['subject_id'],
                'timestamp': datetime.now().isoformat(),
                'gpt_analysis': response.choices[0].message.content,
                'model_used': self.model,
                'tokens_used': response.usage.total_tokens
            }
            
            self.analysis_history.append(analysis)
            return analysis
            
        except Exception as e:
            print(f"Erro na análise OpenAI: {e}")
            return {
                'subject_id': subject_metrics['subject_id'],
                'error': str(e)
            }
    
    def _create_analysis_prompt(self, metrics: Dict) -> str:
        """
        Cria prompt estruturado para análise GPT
        """
        
        # Extrair métricas principais
        subcortical = metrics.get('subcortical', {})
        cortical_lh = metrics.get('cortical', {}).get('lh', {})
        cortical_rh = metrics.get('cortical', {}).get('rh', {})
        
        prompt = f"""
ANÁLISE NEUROIMAGEM - SUJEITO: {metrics['subject_id']}

DADOS SUBCORTICAIS (Volumes em mm³):
"""
        
        # Adicionar volumes subcorticais importantes
        important_structures = [
            'Left-Hippocampus', 'Right-Hippocampus',
            'Left-Amygdala', 'Right-Amygdala',
            'Left-Entorhinal', 'Right-Entorhinal',
            'Left-Temporal-Pole', 'Right-Temporal-Pole',
            'Left-Inferior-Temporal', 'Right-Inferior-Temporal',
            'Left-Middle-Temporal', 'Right-Middle-Temporal',
            'Left-Superior-Temporal', 'Right-Superior-Temporal'
        ]
        
        for structure in important_structures:
            if structure in subcortical:
                prompt += f"{structure}: {subcortical[structure]:.0f} mm³\n"
        
        prompt += "\nDADOS CORTICAIS (Espessura em mm):\n"
        
        # Adicionar espessuras corticais importantes
        important_regions = [
            'entorhinal', 'temporalpole', 'inferiortemporal',
            'middletemporal', 'superiortemporal', 'fusiform',
            'parahippocampal', 'lateraloccipital'
        ]
        
        for region in important_regions:
            if region in cortical_lh:
                lh_thick = cortical_lh[region]['thickness']
                rh_thick = cortical_rh.get(region, {}).get('thickness', 0)
                prompt += f"{region} (L/R): {lh_thick:.2f}/{rh_thick:.2f} mm\n"
        
        prompt += f"""
QUALIDADE DO PROCESSAMENTO:
"""
        
        quality = metrics.get('quality_metrics', {})
        for key, value in quality.items():
            if isinstance(value, bool):
                status = "✅" if value else "❌"
                prompt += f"{key}: {status}\n"
        
        prompt += """

INSTRUÇÕES PARA ANÁLISE:
1. Compare os volumes hipocampais com valores normativos (esquerdo: ~3000-4000mm³, direito: ~3200-4200mm³)
2. Analise assimetrias entre hemisférios
3. Identifique padrões sugestivos de atrofia
4. Avalie risco de MCI/Alzheimer baseado nas métricas
5. Forneça recomendações clínicas específicas
6. Use linguagem médica apropriada mas acessível

Por favor, forneça uma análise detalhada e estruturada."""
        
        return prompt
    
    def analyze_cohort(self, subjects_df: pd.DataFrame, max_analyses: int = 10) -> pd.DataFrame:
        """
        Analisa uma coorte de sujeitos
        """
        print(f"🧠 Iniciando análise OpenAI para até {max_analyses} sujeitos...")
        
        analyses = []
        count = 0
        
        for _, subject_data in subjects_df.iterrows():
            if count >= max_analyses:
                break
                
            print(f"  Analisando {subject_data['subject_id']} ({count+1}/{max_analyses})")
            
            analysis = self.analyze_subject(subject_data.to_dict())
            analyses.append(analysis)
            count += 1
            
            # Pausa para evitar rate limiting
            import time
            time.sleep(1)
        
        return pd.DataFrame(analyses)
    
    def generate_cohort_report(self, analyses_df: pd.DataFrame) -> str:
        """
        Gera relatório agregado da coorte usando GPT
        """
        
        # Preparar resumo das análises
        summary = f"""
RELATÓRIO DE COORTE - ANÁLISE OPENAI
====================================

Total de sujeitos analisados: {len(analyses_df)}
Modelo utilizado: {self.model}
Período de análise: {datetime.now().strftime('%d/%m/%Y %H:%M')}

RESUMO DAS ANÁLISES:
"""
        
        for _, analysis in analyses_df.iterrows():
            if 'gpt_analysis' in analysis:
                summary += f"\n--- {analysis['subject_id']} ---\n"
                summary += analysis['gpt_analysis'][:200] + "...\n"
        
        # Gerar análise agregada com GPT
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Você é um neurocientista especialista em análise de coortes para Alzheimer. Analise o resumo fornecido e gere um relatório executivo."
                    },
                    {
                        "role": "user",
                        "content": f"Analise este resumo de coorte e gere um relatório executivo estruturado:\n\n{summary}"
                    }
                ],
                temperature=0.3,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Erro ao gerar relatório de coorte: {e}"

class FastSurferOpenAIVisualizer:
    """
    Visualizador dos resultados da análise OpenAI + FastSurfer
    """
    
    def __init__(self, analyses_df: pd.DataFrame):
        self.analyses_df = analyses_df
        
    def create_analysis_dashboard(self, output_file: str = "openai_fastsurfer_dashboard.png"):
        """
        Cria dashboard visual dos resultados
        """
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('ANÁLISE FASTSURFER + OPENAI GPT', fontsize=16, fontweight='bold')
        
        # 1. Distribuição de tokens utilizados
        if 'tokens_used' in self.analyses_df.columns:
            axes[0,0].hist(self.analyses_df['tokens_used'], bins=10, alpha=0.7, color='skyblue')
            axes[0,0].set_title('Distribuição de Tokens Utilizados')
            axes[0,0].set_xlabel('Tokens')
            axes[0,0].set_ylabel('Frequência')
        
        # 2. Timeline das análises
        if 'timestamp' in self.analyses_df.columns:
            timestamps = pd.to_datetime(self.analyses_df['timestamp'])
            axes[0,1].plot(timestamps, range(len(timestamps)), 'o-', color='green')
            axes[0,1].set_title('Timeline das Análises')
            axes[0,1].set_xlabel('Tempo')
            axes[0,1].set_ylabel('Sujeito (ordem)')
            axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Status das análises
        if 'gpt_analysis' in self.analyses_df.columns:
            success_count = len(self.analyses_df[self.analyses_df['gpt_analysis'].notna()])
            error_count = len(self.analyses_df) - success_count
        else:
            success_count = 0
            error_count = len(self.analyses_df)
        
        axes[1,0].pie([success_count, error_count], 
                     labels=['Sucesso', 'Erro'], 
                     colors=['lightgreen', 'lightcoral'],
                     autopct='%1.1f%%')
        axes[1,0].set_title('Status das Análises OpenAI')
        
        # 4. Resumo estatístico
        axes[1,1].axis('off')
        # Calcular tokens médios de forma segura
        if 'tokens_used' in self.analyses_df.columns:
            avg_tokens = f"{self.analyses_df['tokens_used'].mean():.0f}"
        else:
            avg_tokens = "N/A"
        
        summary_text = f"""
RESUMO DA ANÁLISE:
==================
Total de Sujeitos: {len(self.analyses_df)}
Análises Bem-sucedidas: {success_count}
Análises com Erro: {error_count}
Taxa de Sucesso: {success_count/len(self.analyses_df)*100:.1f}%

MÉTRICAS OPENAI:
================
Modelo Utilizado: GPT-4o-mini
Tokens Médios: {avg_tokens}
Custo Estimado: ~${len(self.analyses_df) * 0.00015:.4f}

BENEFÍCIOS:
===========
• Interpretação clínica automatizada
• Análise linguística natural
• Recomendações personalizadas
• Detecção de padrões sutis
        """
        
        axes[1,1].text(0.05, 0.95, summary_text, transform=axes[1,1].transAxes,
                      fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Dashboard salvo: {output_file}")

def main():
    """
    Pipeline principal: FastSurfer + OpenAI GPT
    """
    
    print("🧠 ANÁLISE FASTSURFER COM OPENAI GPT")
    print("=" * 50)
    print("Sistema de interpretação clínica automatizada")
    print("Substitui/complementa CNN para análise interpretativa")
    print("=" * 50)
    
    # Configurações
    fastsurfer_dir = "/app/alzheimer/oasis_data/outputs_fastsurfer_definitivo_todos"
    
    # IMPORTANTE: Configure sua API key da OpenAI
    try:
        from config_openai import OPENAI_API_KEY, validate_api_key, estimate_cost
        
        if not validate_api_key():
            print("\n💡 SOLUÇÃO RÁPIDA:")
            print("Execute: ./setup_api_key.sh")
            return None
            
        openai_api_key = OPENAI_API_KEY
        
        # Estimar custo
        estimate_cost(5)  # Estimativa para 5 análises
        
    except ImportError:
        # Fallback para variável de ambiente
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            print("❌ ERRO: OPENAI_API_KEY não configurada!")
            print("Execute: ./setup_api_key.sh")
            return None
    
    # ETAPA 1: Extrair dados FastSurfer
    print("\n📊 ETAPA 1: Extraindo dados FastSurfer...")
    extractor = FastSurferDataExtractor(fastsurfer_dir)
    subjects_df = extractor.extract_all_subjects(max_subjects=20)  # Limitar para teste
    
    if len(subjects_df) == 0:
        print("❌ Nenhum dado FastSurfer encontrado!")
        return
    
    print(f"✅ Dados extraídos: {len(subjects_df)} sujeitos")
    
    # ETAPA 2: Análise com OpenAI
    print("\n🤖 ETAPA 2: Análise com OpenAI GPT...")
    analyzer = OpenAIFastSurferAnalyzer(openai_api_key, model="gpt-4o-mini")
    analyses_df = analyzer.analyze_cohort(subjects_df, max_analyses=5)  # Limitar para teste
    
    # ETAPA 3: Gerar relatório de coorte
    print("\n📋 ETAPA 3: Gerando relatório de coorte...")
    cohort_report = analyzer.generate_cohort_report(analyses_df)
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Salvar análises individuais
    analyses_df.to_csv(f"openai_fastsurfer_analyses_{timestamp}.csv", index=False)
    
    # Salvar relatório de coorte
    with open(f"openai_fastsurfer_cohort_report_{timestamp}.txt", 'w') as f:
        f.write(cohort_report)
    
    # ETAPA 4: Visualização
    print("\n📊 ETAPA 4: Criando visualizações...")
    visualizer = FastSurferOpenAIVisualizer(analyses_df)
    visualizer.create_analysis_dashboard(f"openai_fastsurfer_dashboard_{timestamp}.png")
    
    # Resumo final
    print(f"\n🎉 ANÁLISE OPENAI + FASTSURFER CONCLUÍDA!")
    print("=" * 50)
    print("📁 ARQUIVOS GERADOS:")
    print(f"   • openai_fastsurfer_analyses_{timestamp}.csv")
    print(f"   • openai_fastsurfer_cohort_report_{timestamp}.txt")
    print(f"   • openai_fastsurfer_dashboard_{timestamp}.png")
    
    print(f"\n📊 RESUMO:")
    print(f"   • Sujeitos analisados: {len(analyses_df)}")
    print(f"   • Análises bem-sucedidas: {len(analyses_df[analyses_df['gpt_analysis'].notna()])}")
    print(f"   • Modelo utilizado: GPT-4")
    
    print(f"\n💡 VANTAGENS DA ABORDAGEM OPENAI:")
    print("   • Interpretação clínica natural")
    print("   • Análise contextual avançada")
    print("   • Recomendações personalizadas")
    print("   • Linguagem médica apropriada")
    print("   • Detecção de padrões sutis")
    
    return analyses_df

if __name__ == "__main__":
    results = main()
