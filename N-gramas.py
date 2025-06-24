#!/usr/bin/env python3
"""
Sistema Forense de Atribución de Autoría
Análisis estilométrico mediante n-gramas de carácter y palabra
"""

import argparse
import csv
import logging
import re
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suprimir warnings de sklearn
warnings.filterwarnings('ignore', category=UserWarning)


class ForensicAnalyzer:
    """Analizador forense de autoría mediante n-gramas"""
    
    def __init__(self, min_text_length: int = 300):
        self.min_text_length = min_text_length
        self.profiles_char = {}
        self.profiles_word = {}
        self.text_metadata = {}
    
    def normalize_text(self, text: str) -> str:
        """Normaliza el texto para análisis"""
        # Minúsculas y eliminar retornos de carro excesivos
        text = text.lower().replace('\r', '').strip()
        # Normalizar espacios múltiples pero mantener puntuación
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        return text
    
    def validate_text(self, text: str, text_id: str = "unknown") -> None:
        """Valida que el texto sea suficientemente largo"""
        if len(text) < self.min_text_length:
            raise ValueError(f"Texto '{text_id}' demasiado corto para análisis n-gramas "
                           f"({len(text)} < {self.min_text_length} caracteres)")
    
    def build_char_profile(self, text: str, n_range: Tuple[int, int] = (3, 5), 
                          top_k: int = 500) -> pd.Series:
        """Construye perfil de n-gramas de carácter"""
        vectorizer = CountVectorizer(
            analyzer='char',
            ngram_range=n_range,
            max_features=top_k,
            lowercase=False  # Ya normalizamos antes
        )
        
        # Ajustar y transformar
        char_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        # Crear serie con frecuencias
        frequencies = char_matrix.toarray()[0]
        profile = pd.Series(frequencies, index=feature_names, dtype=int)
        
        # Ordenar por frecuencia descendente
        return profile.sort_values(ascending=False)
    
    def build_word_profile(self, text: str, n_range: Tuple[int, int] = (1, 3), 
                          top_k: int = 500) -> pd.Series:
        """Construye perfil de n-gramas de palabra"""
        vectorizer = CountVectorizer(
            analyzer='word',
            ngram_range=n_range,
            max_features=top_k,
            lowercase=False,  # Ya normalizamos
            token_pattern=r'\b\w+\b'  # Solo palabras, mantiene puntuación separada
        )
        
        word_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        
        frequencies = word_matrix.toarray()[0]
        profile = pd.Series(frequencies, index=feature_names, dtype=int)
        
        return profile.sort_values(ascending=False)
    
    def build_profiles(self, texts: Dict[str, str], level: str = "char", 
                      n_range: Tuple[int, int] = None, top_k: int = 500) -> Dict[str, pd.Series]:
        """
        Construye perfiles de n-gramas para múltiples textos
        
        Args:
            texts: Diccionario {id_texto: contenido}
            level: "char" o "word"
            n_range: Rango de n-gramas
            top_k: Número máximo de n-gramas más frecuentes
        """
        if n_range is None:
            n_range = (3, 5) if level == "char" else (1, 3)
        
        profiles = {}
        
        for text_id, content in texts.items():
            logger.info(f"Construyendo perfil {level} para: {text_id}")
            
            # Normalizar y validar
            normalized = self.normalize_text(content)
            self.validate_text(normalized, text_id)
            
            # Guardar metadata
            self.text_metadata[text_id] = {
                'length': len(normalized),
                'words': len(normalized.split()),
                'level': level
            }
            
            # Construir perfil según tipo
            if level == "char":
                profile = self.build_char_profile(normalized, n_range, top_k)
            elif level == "word":
                profile = self.build_word_profile(normalized, n_range, top_k)
            else:
                raise ValueError(f"Nivel no soportado: {level}")
            
            profiles[text_id] = profile
        
        return profiles
    
    def cosine_tfidf_similarity(self, profiles_a: Dict[str, pd.Series], 
                               profiles_b: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calcula similitud coseno con TF-IDF"""
        # Obtener todos los n-gramas únicos
        all_ngrams = set()
        all_profiles = {**profiles_a, **profiles_b}
        
        for profile in all_profiles.values():
            all_ngrams.update(profile.index)
        
        all_ngrams = sorted(list(all_ngrams))
        
        # Crear matriz de frecuencias
        def profile_to_vector(profile: pd.Series) -> np.ndarray:
            return np.array([profile.get(ngram, 0) for ngram in all_ngrams])
        
        # Vectores TF-IDF
        vectors_a = np.array([profile_to_vector(prof) for prof in profiles_a.values()])
        vectors_b = np.array([profile_to_vector(prof) for prof in profiles_b.values()])
        
        # Aplicar TF-IDF manualmente (más control)
        def apply_tfidf(matrix: np.ndarray) -> np.ndarray:
            # TF: frecuencias normalizadas
            tf = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-10)
            
            # IDF: log(N / df)
            df = (matrix > 0).sum(axis=0)
            n_docs = matrix.shape[0]
            idf = np.log(n_docs / (df + 1))
            
            return tf * idf
        
        # Aplicar TF-IDF
        all_vectors = np.vstack([vectors_a, vectors_b])
        tfidf_vectors = apply_tfidf(all_vectors)
        
        tfidf_a = tfidf_vectors[:len(vectors_a)]
        tfidf_b = tfidf_vectors[len(vectors_a):]
        
        # Calcular similitudes coseno
        similarities = cosine_similarity(tfidf_a, tfidf_b)
        
        return pd.DataFrame(
            similarities,
            index=list(profiles_a.keys()),
            columns=list(profiles_b.keys())
        )
    
    def burrows_delta(self, profiles_a: Dict[str, pd.Series], 
                     profiles_b: Dict[str, pd.Series], top_features: int = 500) -> pd.DataFrame:
        """Calcula Burrows's Delta"""
        # Combinar todos los perfiles
        all_profiles = {**profiles_a, **profiles_b}
        
        # Obtener los top_features más frecuentes globalmente
        global_counts = pd.Series(dtype=int)
        for profile in all_profiles.values():
            global_counts = global_counts.add(profile, fill_value=0)
        
        top_ngrams = global_counts.nlargest(top_features).index.tolist()
        
        # Crear matriz de frecuencias relativas
        freq_matrix = []
        text_ids = []
        
        for text_id, profile in all_profiles.items():
            # Frecuencias relativas (por mil)
            total_ngrams = profile.sum()
            rel_freqs = [(profile.get(ngram, 0) / total_ngrams) * 1000 
                        for ngram in top_ngrams]
            freq_matrix.append(rel_freqs)
            text_ids.append(text_id)
        
        freq_matrix = np.array(freq_matrix)
        
        # Calcular z-scores (Burrows's Delta)
        # log-transform para estabilizar varianza
        log_freqs = np.log(freq_matrix + 1e-6)  # Evitar log(0)
        
        # Media y desviación estándar por feature
        means = np.mean(log_freqs, axis=0)
        stds = np.std(log_freqs, axis=0, ddof=1)
        
        # Z-scores
        z_scores = (log_freqs - means) / (stds + 1e-10)
        
        # Calcular distancias Delta (Manhattan en espacio z-score)
        deltas = np.zeros((len(profiles_a), len(profiles_b)))
        
        for i, text_a in enumerate(profiles_a.keys()):
            idx_a = text_ids.index(text_a)
            for j, text_b in enumerate(profiles_b.keys()):
                idx_b = text_ids.index(text_b)
                # Burrows's Delta: media de diferencias absolutas de z-scores
                delta = np.mean(np.abs(z_scores[idx_a] - z_scores[idx_b]))
                deltas[i, j] = delta
        
        return pd.DataFrame(
            deltas,
            index=list(profiles_a.keys()),
            columns=list(profiles_b.keys())
        )
    
    def jensen_shannon_divergence(self, profiles_a: Dict[str, pd.Series], 
                                 profiles_b: Dict[str, pd.Series]) -> pd.DataFrame:
        """Calcula divergencia Jensen-Shannon"""
        # Obtener vocabulario común
        all_ngrams = set()
        for profile in {**profiles_a, **profiles_b}.values():
            all_ngrams.update(profile.index)
        all_ngrams = sorted(list(all_ngrams))
        
        def profile_to_distribution(profile: pd.Series) -> np.ndarray:
            """Convierte perfil a distribución de probabilidad"""
            vector = np.array([profile.get(ngram, 0) for ngram in all_ngrams])
            # Suavizado de Laplace
            vector = vector + 1
            return vector / vector.sum()
        
        # Calcular divergencias
        jsd_matrix = np.zeros((len(profiles_a), len(profiles_b)))
        
        for i, (text_a, prof_a) in enumerate(profiles_a.items()):
            dist_a = profile_to_distribution(prof_a)
            for j, (text_b, prof_b) in enumerate(profiles_b.items()):
                dist_b = profile_to_distribution(prof_b)
                # Jensen-Shannon divergence (squared distance)
                jsd = jensenshannon(dist_a, dist_b) ** 2
                jsd_matrix[i, j] = jsd
        
        return pd.DataFrame(
            jsd_matrix,
            index=list(profiles_a.keys()),
            columns=list(profiles_b.keys())
        )
    
    def compare_profiles(self, profiles_a: Dict[str, pd.Series], 
                        profiles_b: Dict[str, pd.Series],
                        metrics: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Compara perfiles usando múltiples métricas"""
        if metrics is None:
            metrics = ["cosine", "delta", "jsd"]
        
        results = {}
        
        if "cosine" in metrics:
            logger.info("Calculando similitud coseno TF-IDF...")
            results["cosine"] = self.cosine_tfidf_similarity(profiles_a, profiles_b)
        
        if "delta" in metrics:
            logger.info("Calculando Burrows's Delta...")
            results["delta"] = self.burrows_delta(profiles_a, profiles_b)
        
        if "jsd" in metrics:
            logger.info("Calculando Jensen-Shannon Divergence...")
            results["jsd"] = self.jensen_shannon_divergence(profiles_a, profiles_b)
        
        return results
    
    def plot_top_ngrams(self, profile: pd.Series, text_id: str, 
                       top_n: int = 20, save_path: Optional[Path] = None) -> Path:
        """Genera gráfico de barras con los n-gramas más frecuentes"""
        plt.figure(figsize=(12, 8))
        
        # Top n-gramas
        top_ngrams = profile.head(top_n)
        
        # Frecuencias relativas
        total = profile.sum()
        rel_freqs = (top_ngrams / total) * 100
        
        # Gráfico de barras
        bars = plt.bar(range(len(rel_freqs)), rel_freqs.values, 
                      color='steelblue', alpha=0.7)
        
        # Personalización
        plt.title(f'Top {top_n} n-gramas más frecuentes: {text_id}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('n-gramas', fontsize=12)
        plt.ylabel('Frecuencia relativa (%)', fontsize=12)
        
        # Etiquetas en x (rotadas para legibilidad)
        plt.xticks(range(len(rel_freqs)), rel_freqs.index, 
                  rotation=45, ha='right', fontsize=10)
        
        # Valores sobre las barras
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.2f}%', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.grid(axis='y', alpha=0.3)
        
        # Guardar
        if save_path is None:
            save_path = Path(f"ngrams_{text_id}.png")
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def export_results(self, profiles_known: Dict[str, pd.Series],
                      profiles_query: Dict[str, pd.Series],
                      distances: Dict[str, pd.DataFrame],
                      out_dir: Path) -> Dict[str, Path]:
        """Exporta resultados a archivos"""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        exported_files = {}
        
        # 1. Exportar perfiles como CSV
        logger.info("Exportando perfiles...")
        profiles_data = []
        
        all_profiles = {**profiles_known, **profiles_query}
        for text_id, profile in all_profiles.items():
            category = "known" if text_id in profiles_known else "query"
            
            for ngram, freq in profile.head(100).items():  # Top 100 por texto
                profiles_data.append({
                    'text_id': text_id,
                    'category': category,
                    'ngram': ngram,
                    'frequency': freq,
                    'rel_frequency': freq / profile.sum() * 100
                })
        
        profiles_df = pd.DataFrame(profiles_data)
        profiles_path = out_dir / "profiles.csv"
        profiles_df.to_csv(profiles_path, index=False, encoding='utf-8')
        exported_files['profiles'] = profiles_path
        
        # 2. Exportar matrices de distancia
        logger.info("Exportando matrices de distancia...")
        for metric, matrix in distances.items():
            matrix_path = out_dir / f"distances_{metric}.csv"
            matrix.to_csv(matrix_path, encoding='utf-8')
            exported_files[f'distances_{metric}'] = matrix_path
        
        # 3. Exportar resumen ejecutivo
        logger.info("Generando resumen ejecutivo...")
        summary_data = []
        
        for metric, matrix in distances.items():
            for known_text in matrix.index:
                for query_text in matrix.columns:
                    value = matrix.loc[known_text, query_text]
                    summary_data.append({
                        'known_text': known_text,
                        'query_text': query_text,
                        'metric': metric,
                        'value': value,
                        'rank': 0  # Se calculará después
                    })
        
        summary_df = pd.DataFrame(summary_data)
        
        # Calcular rankings por métrica
        for metric in distances.keys():
            metric_data = summary_df[summary_df['metric'] == metric].copy()
            
            # Para cada texto query, rankear por similitud/distancia
            for query_text in metric_data['query_text'].unique():
                query_data = metric_data[metric_data['query_text'] == query_text].copy()
                
                # Cosine: mayor es mejor (similitud)
                # Delta y JSD: menor es mejor (distancia)
                ascending = metric != 'cosine'
                query_data = query_data.sort_values('value', ascending=ascending)
                query_data['rank'] = range(1, len(query_data) + 1)
                
                # Actualizar en summary_df
                mask = (summary_df['metric'] == metric) & (summary_df['query_text'] == query_text)
                summary_df.loc[mask, 'rank'] = query_data['rank'].values
        
        summary_path = out_dir / "summary.csv"
        summary_df.to_csv(summary_path, index=False, encoding='utf-8')
        exported_files['summary'] = summary_path
        
        # 4. Generar gráficos de n-gramas
        logger.info("Generando gráficos de n-gramas...")
        for text_id, profile in all_profiles.items():
            plot_path = out_dir / f"ngrams_{text_id}.png"
            self.plot_top_ngrams(profile, text_id, save_path=plot_path)
            exported_files[f'plot_{text_id}'] = plot_path
        
        # 5. Metadata
        metadata = {
            'analysis_type': 'forensic_authorship',
            'known_texts': list(profiles_known.keys()),
            'query_texts': list(profiles_query.keys()),
            'metrics_used': list(distances.keys()),
            'total_files_generated': len(exported_files)
        }
        
        metadata_path = out_dir / "metadata.json"
        import json
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        exported_files['metadata'] = metadata_path
        
        logger.info(f"Exportación completa. {len(exported_files)} archivos generados en {out_dir}")
        return exported_files


def load_texts_from_directory(directory: Path) -> Dict[str, str]:
    """Carga todos los archivos de texto de un directorio"""
    texts = {}
    directory = Path(directory)
    
    if not directory.exists():
        raise FileNotFoundError(f"Directorio no encontrado: {directory}")
    
    # Formatos soportados
    extensions = ['.txt', '.md', '.text']
    
    for file_path in directory.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # Solo textos no vacíos
                        texts[file_path.stem] = content
                        logger.info(f"Cargado: {file_path.name} ({len(content)} caracteres)")
            except UnicodeDecodeError:
                logger.warning(f"Error de codificación en: {file_path}")
            except Exception as e:
                logger.error(f"Error cargando {file_path}: {e}")
    
    if not texts:
        raise ValueError(f"No se encontraron textos válidos en: {directory}")
    
    return texts


def main():
    """Función principal para uso como script"""
    parser = argparse.ArgumentParser(
        description="Sistema Forense de Atribución de Autoría",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python ngram_mod.py --known ./conocidos --query ./dubitados --out ./resultados
  python ngram_mod.py --known ./conocidos --query ./dubitados --level word --metrics cosine delta
        """
    )
    
    parser.add_argument('--known', type=str, required=True,
                       help='Directorio con textos de autor conocido')
    parser.add_argument('--query', type=str, required=True,
                       help='Directorio con textos dubitados')
    parser.add_argument('--out', type=str, required=True,
                       help='Directorio de salida para resultados')
    parser.add_argument('--level', choices=['char', 'word'], default='char',
                       help='Tipo de n-gramas: char o word (default: char)')
    parser.add_argument('--n-range', nargs=2, type=int, metavar=('MIN', 'MAX'),
                       help='Rango de n-gramas (default: 3-5 para char, 1-3 para word)')
    parser.add_argument('--top-k', type=int, default=500,
                       help='Número máximo de n-gramas más frecuentes (default: 500)')
    parser.add_argument('--metrics', nargs='+', choices=['cosine', 'delta', 'jsd'],
                       default=['cosine', 'delta', 'jsd'],
                       help='Métricas a calcular (default: todas)')
    parser.add_argument('--min-length', type=int, default=300,
                       help='Longitud mínima de texto en caracteres (default: 300)')
    
    args = parser.parse_args()
    
    try:
        # Inicializar analizador
        analyzer = ForensicAnalyzer(min_text_length=args.min_length)
        
        # Cargar textos
        logger.info("Cargando textos conocidos...")
        known_texts = load_texts_from_directory(args.known)
        logger.info(f"Cargados {len(known_texts)} textos conocidos")
        
        logger.info("Cargando textos dubitados...")
        query_texts = load_texts_from_directory(args.query)
        logger.info(f"Cargados {len(query_texts)} textos dubitados")
        
        # Configurar rango de n-gramas
        n_range = args.n_range
        if n_range is None:
            n_range = (3, 5) if args.level == 'char' else (1, 3)
        
        # Construir perfiles
        logger.info(f"Construyendo perfiles de n-gramas ({args.level})...")
        profiles_known = analyzer.build_profiles(
            known_texts, level=args.level, n_range=tuple(n_range), top_k=args.top_k
        )
        profiles_query = analyzer.build_profiles(
            query_texts, level=args.level, n_range=tuple(n_range), top_k=args.top_k
        )
        
        # Comparar perfiles
        logger.info("Comparando perfiles...")
        distances = analyzer.compare_profiles(
            profiles_known, profiles_query, metrics=args.metrics
        )
        
        # Exportar resultados
        logger.info("Exportando resultados...")
        exported_files = analyzer.export_results(
            profiles_known, profiles_query, distances, Path(args.out)
        )
        
        # Mostrar resumen
        print("\n" + "="*60)
        print("ANÁLISIS FORENSE COMPLETADO")
        print("="*60)
        print(f"Textos conocidos: {len(known_texts)}")
        print(f"Textos dubitados: {len(query_texts)}")
        print(f"Métricas calculadas: {', '.join(args.metrics)}")
        print(f"Archivos generados: {len(exported_files)}")
        print(f"Directorio de salida: {args.out}")
        print("\nArchivos principales:")
        for key, path in exported_files.items():
            if not key.startswith('plot_'):
                print(f"  - {key}: {path.name}")
        
        return exported_files
        
    except Exception as e:
        logger.error(f"Error durante el análisis: {e}")
        raise


if __name__ == "__main__":
    main()
