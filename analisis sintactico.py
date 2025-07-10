#!/usr/bin/env python3
"""
Sistema Forense de Atribución de Autoría - Módulo de Análisis Sintáctico
========================================================================

Módulo autocontenible para análisis forense de textos en español mediante
características sintácticas. Genera perfiles sintácticos y matrices de comparación
para determinar similitudes de autoría.

Autor: Sistema Forense IA
Versión: 1.0
Python: 3.12+
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Dependencias científicas
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon, cosine
from sklearn.feature_extraction.text import CountVectorizer

# Visualización
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# NLP
import spacy
from spacy import displacy

# Configuración de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SyntaxForensics:
    """
    Analizador forense de características sintácticas para atribución de autoría.
    
    Extrae y compara patrones sintácticos de textos conocidos vs. dubitados
    usando distribuciones de POS, dependencias y métricas estructurales.
    """
    
    def __init__(self, model_name: str = "es_core_news_md", top_k: int = 500):
        """
        Inicializa el analizador forense.
        
        Args:
            model_name: Modelo de spaCy para español
            top_k: Número de n-gramas más frecuentes para análisis
        """
        self.model_name = model_name
        self.top_k = top_k
        self.nlp = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Carga el modelo de spaCy optimizado para análisis sintáctico."""
        try:
            self.nlp = spacy.load(self.model_name)
            # Optimización: desactivar componentes innecesarios
            pipes_to_disable = [p for p in ('ner', 'textcat') if p in self.nlp.pipe_names]
            if pipes_to_disable:
                self.nlp.disable_pipes(*pipes_to_disable)
            logger.info(f"Modelo {self.model_name} cargado exitosamente")
        except OSError:
            logger.error(f"Error: Modelo {self.model_name} no encontrado")
            logger.info("Instala con: python -m spacy download es_core_news_md")
            raise
    
    def extract_syntax_features(self, text: str) -> Dict[str, Any]:
        """
        Extrae características sintácticas completas de un texto.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Diccionario con todas las características sintácticas
        """
        doc = self.nlp(text)
        
        # Separar por oraciones
        sentences = list(doc.sents)
        if not sentences:
            return self._empty_features()
        
        # Extraer características
        pos_sequences = []
        dep_labels = []
        sentence_lengths = []
        dependency_distances = []
        subordinate_clauses = 0
        
        for sent in sentences:
            # Secuencia POS de la oración
            pos_seq = [token.pos_ for token in sent if not token.is_space]
            if pos_seq:  # Solo oraciones no vacías
                pos_sequences.append(' '.join(pos_seq))
                sentence_lengths.append(len(pos_seq))
            
            # Etiquetas de dependencia
            for token in sent:
                if not token.is_space:
                    dep_labels.append(token.dep_)
                    
                    # Distancia de dependencia
                    if token.head != token:  # No es raíz
                        dep_dist = abs(token.i - token.head.i)
                        dependency_distances.append(dep_dist)
                    
                    # Contar cláusulas subordinadas
                    if token.dep_ in ['mark', 'advcl', 'ccomp', 'xcomp']:
                        subordinate_clauses += 1
        
        # Calcular métricas escalares
        avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
        avg_dependency_distance = np.mean(dependency_distances) if dependency_distances else 0
        subordinate_ratio = subordinate_clauses / len(sentences) if sentences else 0
        
        # Generar n-gramas POS
        pos_trigrams = self._extract_pos_ngrams(pos_sequences, n=3)
        pos_unigrams = self._extract_pos_ngrams(pos_sequences, n=1)
        
        # Distribuciones
        dep_distribution = self._calculate_distribution(dep_labels)
        
        return {
            'pos_trigrams': pos_trigrams,
            'pos_unigrams': pos_unigrams,
            'dep_distribution': dep_distribution,
            'avg_sentence_length': avg_sentence_length,
            'avg_dependency_distance': avg_dependency_distance,
            'subordinate_ratio': subordinate_ratio,
            'total_sentences': len(sentences),
            'total_tokens': len([t for t in doc if not t.is_space])
        }
    
    def _extract_pos_ngrams(self, pos_sequences: List[str], n: int) -> Dict[str, float]:
        """Extrae n-gramas de etiquetas POS normalizados."""
        if not pos_sequences:
            return {}
        
        vectorizer = CountVectorizer(
            analyzer='word', 
            ngram_range=(n, n), 
            max_features=self.top_k
        )
        
        try:
            X = vectorizer.fit_transform(pos_sequences)
            feature_names = vectorizer.get_feature_names_out()
            frequencies = X.sum(axis=0).A1
            
            # Normalizar a frecuencias relativas
            total = frequencies.sum()
            if total > 0:
                relative_freq = {name: freq/total for name, freq in zip(feature_names, frequencies)}
                return dict(sorted(relative_freq.items(), key=lambda x: x[1], reverse=True))
            return {}
        except ValueError:
            return {}
    
    def _calculate_distribution(self, labels: List[str]) -> Dict[str, float]:
        """Calcula distribución normalizada de etiquetas."""
        if not labels:
            return {}
        
        unique, counts = np.unique(labels, return_counts=True)
        total = counts.sum()
        return {label: count/total for label, count in zip(unique, counts)}
    
    def _empty_features(self) -> Dict[str, Any]:
        """Retorna características vacías para textos sin contenido."""
        return {
            'pos_trigrams': {},
            'pos_unigrams': {},
            'dep_distribution': {},
            'avg_sentence_length': 0,
            'avg_dependency_distance': 0,
            'subordinate_ratio': 0,
            'total_sentences': 0,
            'total_tokens': 0
        }
    
    def build_syntax_profiles(self, texts: Dict[str, str]) -> Dict[str, pd.Series]:
        """
        Construye perfiles sintácticos para múltiples textos.
        
        Args:
            texts: Diccionario {nombre_texto: contenido}
            
        Returns:
            Diccionario con perfiles sintácticos por texto
        """
        profiles = {}
        
        for name, text in texts.items():
            logger.info(f"Procesando: {name}")
            features = self.extract_syntax_features(text)
            
            # Convertir a Series para facilitar análisis
            scalar_features = {
                'avg_sentence_length': features['avg_sentence_length'],
                'avg_dependency_distance': features['avg_dependency_distance'],
                'subordinate_ratio': features['subordinate_ratio'],
                'total_sentences': features['total_sentences'],
                'total_tokens': features['total_tokens']
            }
            
            # Agregar top POS trigrams como características escalares
            top_trigrams = dict(list(features['pos_trigrams'].items())[:20])
            for trigram, freq in top_trigrams.items():
                scalar_features[f'trigram_{trigram}'] = freq
            
            # Agregar distribuciones de dependencias principales
            top_deps = dict(list(features['dep_distribution'].items())[:15])
            for dep, freq in top_deps.items():
                scalar_features[f'dep_{dep}'] = freq
            
            profiles[name] = pd.Series(scalar_features).fillna(0)
        
        return profiles
    
    def compare_syntax(self, 
                      profiles_known: Dict[str, pd.Series], 
                      profiles_query: Dict[str, pd.Series],
                      metrics: List[str] = ["cosine", "delta", "jsd"]) -> pd.DataFrame:
        """
        Compara perfiles sintácticos usando múltiples métricas.
        
        Args:
            profiles_known: Perfiles de textos conocidos
            profiles_query: Perfiles de textos dubitados
            metrics: Lista de métricas a calcular
            
        Returns:
            DataFrame con matriz de distancias
        """
        results = []
        
        for query_name, query_profile in profiles_query.items():
            for known_name, known_profile in profiles_known.items():
                
                # Alinear características
                common_features = query_profile.index.intersection(known_profile.index)
                if len(common_features) == 0:
                    logger.warning(f"No hay características comunes entre {query_name} y {known_name}")
                    continue
                
                q_aligned = query_profile[common_features].fillna(0)
                k_aligned = known_profile[common_features].fillna(0)
                
                distances = {'query': query_name, 'known': known_name}
                
                # Calcular métricas solicitadas
                if "cosine" in metrics:
                    distances['cosine'] = self._cosine_distance(q_aligned, k_aligned)
                
                if "delta" in metrics:
                    distances['delta'] = self._burrows_delta(q_aligned, k_aligned, profiles_known)
                
                if "jsd" in metrics:
                    distances['jsd'] = self._jensen_shannon_distance(q_aligned, k_aligned)
                
                results.append(distances)
        
        return pd.DataFrame(results)
    
    def _cosine_distance(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """Calcula distancia coseno entre dos vectores."""
        try:
            return cosine(vec1.values, vec2.values)
        except:
            return 1.0  # Máxima distancia en caso de error
    
    def _burrows_delta(self, vec1: pd.Series, vec2: pd.Series, all_profiles: Dict) -> float:
        """Implementa Delta de Burrows para estilometría."""
        try:
            # Combinar todos los perfiles para calcular estadísticas globales
            all_values = pd.concat(all_profiles.values(), axis=1, ignore_index=True)
            
            # Calcular media y desviación estándar por característica
            means = all_values.mean(axis=1)
            stds = all_values.std(axis=1).replace(0, 1)  # Evitar división por cero
            
            # Normalizar usando z-score
            vec1_norm = (vec1 - means) / stds
            vec2_norm = (vec2 - means) / stds
            
            # Delta de Burrows: media de diferencias absolutas
            delta = np.mean(np.abs(vec1_norm - vec2_norm))
            return delta
        except:
            return float('inf')
    
    def _jensen_shannon_distance(self, vec1: pd.Series, vec2: pd.Series) -> float:
        """Calcula divergencia Jensen-Shannon al cuadrado."""
        try:
            # Normalizar para que sumen 1 (distribuciones de probabilidad)
            v1 = vec1.values
            v2 = vec2.values
            
            v1 = v1 / v1.sum() if v1.sum() > 0 else v1
            v2 = v2 / v2.sum() if v2.sum() > 0 else v2
            
            # Agregar pequeño epsilon para evitar log(0)
            epsilon = 1e-10
            v1 = v1 + epsilon
            v2 = v2 + epsilon
            
            # Renormalizar
            v1 = v1 / v1.sum()
            v2 = v2 / v2.sum()
            
            js_distance = jensenshannon(v1, v2) ** 2
            return js_distance
        except:
            return 1.0
    
    def export_syntax_results(self,
                            profiles: Dict[str, pd.Series],
                            distances: pd.DataFrame,
                            out_dir: Path,
                            author: str = "",
                            query: str = "") -> Dict[str, Path]:
        """
        Exporta todos los resultados del análisis sintáctico.
        
        Args:
            profiles: Perfiles sintácticos
            distances: Matriz de distancias
            out_dir: Directorio de salida
            
        Returns:
            Diccionario con rutas de artefactos generados
        """
        if author and query:
            out_dir = Path(out_dir) / author / f"Resultados_{author}_{query}"
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        artifacts = {}
        
        # 1. Exportar perfiles como CSV
        profiles_df = pd.DataFrame(profiles).T
        profiles_path = out_dir / "syntax_profiles.csv"
        profiles_df.to_csv(profiles_path, encoding='utf-8')
        artifacts['profiles'] = profiles_path
        
        # 2. Exportar matriz de distancias
        distances_path = out_dir / "syntax_distances.csv"
        distances.to_csv(distances_path, index=False, encoding='utf-8')
        artifacts['distances'] = distances_path
        
        # 3. Generar gráficos individuales por texto
        individual_plots = []
        for name, profile in profiles.items():
            plot_path = self._plot_individual_profile(name, profile, out_dir)
            if plot_path:
                individual_plots.append(plot_path)
        artifacts['individual_plots'] = individual_plots
        
        # 4. Generar heatmap global de distancias
        if not distances.empty and 'cosine' in distances.columns:
            heatmap_path = self._plot_distance_heatmap(distances, out_dir)
            if heatmap_path:
                artifacts['heatmap'] = heatmap_path
        
        # 5. Generar reporte resumen
        report_path = self._generate_summary_report(profiles, distances, out_dir)
        artifacts['report'] = report_path
        
        logger.info(f"Resultados exportados en: {out_dir}")
        return artifacts
    
    def _plot_individual_profile(self, name: str, profile: pd.Series, out_dir: Path) -> Optional[Path]:
        """Genera gráfico de barras para perfil individual."""
        # Filtrar solo trigramas POS para visualización
        trigram_features = profile.filter(regex=r'^trigram_').head(15)
        
        if trigram_features.empty:
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Limpiar nombres de características
        clean_names = [name.replace('trigram_', '') for name in trigram_features.index]
        
        bars = plt.barh(range(len(clean_names)), trigram_features.values)
        plt.yticks(range(len(clean_names)), clean_names)
        plt.xlabel('Frecuencia Relativa')
        plt.title(f'Top 15 POS-Trigramas: {name}')
        plt.gca().invert_yaxis()
        
        # Colorear barras
        colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        plt.tight_layout()
        
        plot_path = out_dir / f"profile_{name.replace(' ', '_')}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_path
    
    def _plot_distance_heatmap(self, distances: pd.DataFrame, out_dir: Path) -> Optional[Path]:
        """Genera heatmap de matriz de distancias coseno."""
        if 'cosine' not in distances.columns:
            return None
        
        # Crear matriz pivoteada
        try:
            pivot_matrix = distances.pivot(index='query', columns='known', values='cosine')
        except:
            logger.warning("No se pudo crear matriz pivot para heatmap")
            return None
        
        plt.figure(figsize=(12, 8))
        
        # Crear heatmap
        im = plt.imshow(pivot_matrix.values, cmap='RdYlBu_r', aspect='auto')
        
        # Configurar ejes
        plt.xticks(range(len(pivot_matrix.columns)), pivot_matrix.columns, rotation=45, ha='right')
        plt.yticks(range(len(pivot_matrix.index)), pivot_matrix.index)
        
        # Añadir colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Distancia Coseno')
        
        # Añadir valores en las celdas
        for i in range(len(pivot_matrix.index)):
            for j in range(len(pivot_matrix.columns)):
                value = pivot_matrix.iloc[i, j]
                if not pd.isna(value):
                    plt.text(j, i, f'{value:.3f}', ha='center', va='center',
                           color='white' if value > 0.5 else 'black')
        
        plt.title('Matriz de Distancias Coseno - Análisis Sintáctico')
        plt.xlabel('Textos Conocidos')
        plt.ylabel('Textos Dubitados')
        plt.tight_layout()
        
        heatmap_path = out_dir / "distance_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return heatmap_path
    
    def _generate_summary_report(self, 
                               profiles: Dict[str, pd.Series], 
                               distances: pd.DataFrame, 
                               out_dir: Path) -> Path:
        """Genera reporte resumen del análisis."""
        report_path = out_dir / "syntax_analysis_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ANÁLISIS SINTÁCTICO FORENSE\n")
            f.write("=" * 50 + "\n\n")
            
            # Información general
            f.write(f"Textos analizados: {len(profiles)}\n")
            f.write(f"Comparaciones realizadas: {len(distances)}\n\n")
            
            # Estadísticas de perfiles
            f.write("ESTADÍSTICAS DE PERFILES SINTÁCTICOS:\n")
            f.write("-" * 40 + "\n")
            
            profiles_df = pd.DataFrame(profiles).T
            for metric in ['avg_sentence_length', 'avg_dependency_distance', 'subordinate_ratio']:
                if metric in profiles_df.columns:
                    f.write(f"{metric}:\n")
                    f.write(f"  Media: {profiles_df[metric].mean():.4f}\n")
                    f.write(f"  Desv. Std: {profiles_df[metric].std():.4f}\n")
                    f.write(f"  Min-Max: {profiles_df[metric].min():.4f} - {profiles_df[metric].max():.4f}\n\n")
            
            # Comparaciones más similares
            if not distances.empty:
                f.write("COMPARACIONES MÁS SIMILARES:\n")
                f.write("-" * 40 + "\n")
                
                for metric in ['cosine', 'delta', 'jsd']:
                    if metric in distances.columns:
                        top_similar = distances.nsmallest(3, metric)
                        f.write(f"\nTop 3 según {metric.upper()}:\n")
                        for _, row in top_similar.iterrows():
                            f.write(f"  {row['query']} ↔ {row['known']}: {row[metric]:.4f}\n")
        
        return report_path


def load_texts_from_directory(directory: Path, combine_subdirs: bool = False) -> Dict[str, str]:
    """Carga archivos de texto de un directorio.

    Si ``combine_subdirs`` es ``True`` y el directorio contiene subcarpetas, los
    archivos de cada subdirectorio se combinan y se devuelven bajo el nombre de
    dicha carpeta. Esto es útil para manejar colecciones como ``Dubitados/A1``.
    """

    directory = Path(directory)
    texts: Dict[str, str] = {}

    subdirs = [d for d in directory.iterdir() if d.is_dir()]
    if combine_subdirs and subdirs:
        for sub in subdirs:
            sub_texts = load_texts_from_directory(sub)
            if sub_texts:
                texts[sub.name] = "\n".join(sub_texts.values())
        return texts

    for file_path in directory.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                if content:
                    texts[file_path.stem] = content
                    logger.info(f"Cargado: {file_path.name} ({len(content)} caracteres)")
        except Exception as e:
            logger.warning(f"Error cargando {file_path}: {e}")

    return texts


def main():
    """Función principal CLI."""
    parser = argparse.ArgumentParser(
        description="Sistema Forense de Análisis Sintáctico",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplo de uso:
  python syntax_mod.py --known textos_conocidos/ --query textos_dubitados/ --out resultados/
  
El sistema generará:
  - syntax_profiles.csv: Perfiles sintácticos por texto
  - syntax_distances.csv: Matriz de comparaciones
  - profile_*.png: Gráficos individuales por texto
  - distance_heatmap.png: Heatmap de similitudes
  - syntax_analysis_report.txt: Reporte resumen
        """
    )
    
    parser.add_argument("--known", type=Path, required=True,
                       help="Directorio con textos conocidos (.txt)")
    parser.add_argument("--query", type=Path, required=True,
                       help="Directorio con textos dubitados (.txt)")
    parser.add_argument("--out", type=Path, required=True,
                       help="Directorio de salida para resultados")
    parser.add_argument("--top_k", type=int, default=500,
                       help="Número máximo de características (default: 500)")
    parser.add_argument("--model", type=str, default="es_core_news_md",
                       help="Modelo spaCy (default: es_core_news_md)")
    parser.add_argument("--metrics", nargs='+', default=["cosine", "delta", "jsd"],
                       choices=["cosine", "delta", "jsd"],
                       help="Métricas de comparación (default: todas)")
    
    args = parser.parse_args()
    
    # Validar directorios de entrada
    if not args.known.exists() or not args.known.is_dir():
        logger.error(f"Directorio de textos conocidos no existe: {args.known}")
        return
    
    if not args.query.exists() or not args.query.is_dir():
        logger.error(f"Directorio de textos dubitados no existe: {args.query}")
        return
    
    # Cargar textos
    logger.info("Cargando textos conocidos...")
    known_texts = load_texts_from_directory(args.known)
    
    logger.info("Cargando textos dubitados...")
    query_texts = load_texts_from_directory(args.query)
    
    if not known_texts:
        logger.error("No se encontraron textos conocidos válidos")
        return
    
    if not query_texts:
        logger.error("No se encontraron textos dubitados válidos")
        return
    
    # Ejecutar análisis forense
    logger.info("Iniciando análisis sintáctico forense...")
    
    forensics = SyntaxForensics(model_name=args.model, top_k=args.top_k)
    
    # Construir perfiles
    logger.info("Construyendo perfiles sintácticos...")
    known_profiles = forensics.build_syntax_profiles(known_texts)
    query_profiles = forensics.build_syntax_profiles(query_texts)
    
    # Comparar perfiles
    logger.info("Comparando perfiles...")
    distances = forensics.compare_syntax(known_profiles, query_profiles, args.metrics)
    
    # Exportar resultados
    logger.info("Exportando resultados...")
    all_profiles = {**known_profiles, **query_profiles}
    artifacts = forensics.export_syntax_results(all_profiles, distances, args.out)
    
    # Mostrar resumen
    print("\n" + "="*60)
    print("ANÁLISIS SINTÁCTICO FORENSE COMPLETADO")
    print("="*60)
    print(f"Textos conocidos procesados: {len(known_texts)}")
    print(f"Textos dubitados procesados: {len(query_texts)}")
    print(f"Comparaciones realizadas: {len(distances)}")
    print(f"Artefactos generados: {len(artifacts)}")
    print(f"\nResultados disponibles en: {args.out}")
    
    # Mostrar mejores coincidencias
    if not distances.empty and 'cosine' in distances.columns:
        print("\nMEJORES COINCIDENCIAS (Distancia Coseno):")
        print("-" * 50)
        top_matches = distances.nsmallest(5, 'cosine')
        for _, row in top_matches.iterrows():
            similarity = 1 - row['cosine']  # Convertir distancia a similitud
            print(f"{row['query']} ↔ {row['known']}: {similarity:.3f} ({row['cosine']:.3f})")


if __name__ == "__main__":
    main()
