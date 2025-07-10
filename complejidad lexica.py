
"""
Módulo de Análisis de Complejidad y Riqueza Léxica para Atribución de Autoría

Este módulo implementa 11 indicadores contemporáneos de complejidad y riqueza léxica
para textos en español, diseñado específicamente para proyectos de atribución de autoría.

MIT License

Copyright (c) 2025

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import spacy
from lexicalrichness import LexicalRichness
from readability_es import FernandezHuerta, SzigrisztPazos
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler

# Suprimir warnings innecesarios
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class LexicalComplexityAnalyzer:
    """
    Analizador de complejidad y riqueza léxica para textos en español.
    
    Implementa 11 métricas contemporáneas organizadas en 4 categorías:
    - Riqueza léxica (5 métricas)
    - Complejidad superficial (2 métricas)  
    - Lecturabilidad (2 métricas)
    - Léxico avanzado (2 métricas)
    """
    
    METRIC_NAMES = [
        'TTR', 'RTTR', 'CTTR', 'MTLD', 'MaasA',
        'avg_word_length', 'pct_long_words',
        'fernandez_huerta', 'szigriszt_pazos',
        'pct_rare_words', 'pct_advanced_words'
    ]
    
    def __init__(self, spacy_model: str = "es_core_news_md"):
        """
        Inicializa el analizador.
        
        Args:
            spacy_model: Modelo de spaCy a utilizar (por defecto es_core_news_md)
        """
        self.spacy_model_name = spacy_model
        self._nlp = None
        self._freq_vocab = None
        
    @property
    def nlp(self):
        """Pipeline de spaCy cargado de forma lazy."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.spacy_model_name, disable=['ner'])
            except OSError:
                raise OSError(
                    f"Modelo spaCy '{self.spacy_model_name}' no encontrado. "
                    f"Instálalo con: python -m spacy download {self.spacy_model_name}"
                )
        return self._nlp
    
    @property
    def freq_vocab(self):
        """Vocabulario de frecuencias de spaCy cargado de forma lazy."""
        if self._freq_vocab is None:
            # Extraer las 2000 palabras más frecuentes del vocabulario de spaCy
            vocab_items = [(token.text.lower(), token.prob) 
                          for token in self.nlp.vocab 
                          if token.prob != 0 and token.is_alpha]
            vocab_items.sort(key=lambda x: x[1], reverse=True)
            self._freq_vocab = set(word for word, _ in vocab_items[:2000])
        return self._freq_vocab
    
    def _preprocess_text(self, text: str) -> List[str]:
        """
        Preprocesa el texto y extrae tokens alfabéticos.
        
        Args:
            text: Texto de entrada
            
        Returns:
            Lista de tokens alfabéticos en minúsculas
        """
        doc = self.nlp(text)
        tokens = [token.text.lower() for token in doc 
                 if token.is_alpha and not token.is_stop]
        return tokens
    
    def _calculate_richness_metrics(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calcula métricas de riqueza léxica usando lexicalrichness.
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            Diccionario con métricas TTR, RTTR, CTTR, MTLD, MaasA
        """
        if not tokens:
            return {metric: 0.0 for metric in ['TTR', 'RTTR', 'CTTR', 'MTLD', 'MaasA']}
        
        # Crear texto para lexicalrichness
        text_for_lr = ' '.join(tokens)
        lr = LexicalRichness(text_for_lr)
        
        try:
            ttr = lr.ttr
        except:
            ttr = 0.0
            
        try:
            rttr = lr.rttr
        except:
            rttr = 0.0
            
        try:
            cttr = lr.cttr
        except:
            cttr = 0.0
            
        try:
            mtld = lr.mtld(threshold=0.72)
        except:
            mtld = 0.0
            
        try:
            maas_a = lr.maas
        except:
            maas_a = 0.0
        
        return {
            'TTR': ttr,
            'RTTR': rttr,
            'CTTR': cttr,
            'MTLD': mtld,
            'MaasA': maas_a
        }
    
    def _calculate_surface_complexity(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calcula métricas de complejidad superficial.
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            Diccionario con longitud media de palabra y % palabras largas
        """
        if not tokens:
            return {'avg_word_length': 0.0, 'pct_long_words': 0.0}
        
        word_lengths = [len(token) for token in tokens]
        avg_length = np.mean(word_lengths)
        pct_long = (sum(1 for length in word_lengths if length >= 7) / len(tokens)) * 100
        
        return {
            'avg_word_length': avg_length,
            'pct_long_words': pct_long
        }
    
    def _calculate_readability(self, text: str) -> Dict[str, float]:
        """
        Calcula índices de lecturabilidad para español.
        
        Args:
            text: Texto original
            
        Returns:
            Diccionario con índices Fernández-Huerta y Szigriszt-Pazos
        """
        try:
            fh = FernandezHuerta(text)
            fh_score = fh.score()
        except:
            fh_score = 0.0
            
        try:
            sp = SzigrisztPazos(text)
            sp_score = sp.score()
        except:
            sp_score = 0.0
        
        return {
            'fernandez_huerta': fh_score,
            'szigriszt_pazos': sp_score
        }
    
    def _calculate_advanced_lexicon(self, tokens: List[str]) -> Dict[str, float]:
        """
        Calcula métricas de léxico avanzado.
        
        Args:
            tokens: Lista de tokens
            
        Returns:
            Diccionario con % palabras raras y % palabras avanzadas
        """
        if not tokens:
            return {'pct_rare_words': 0.0, 'pct_advanced_words': 0.0}
        
        # Contar frecuencias en el texto
        token_counts = Counter(tokens)
        
        # Palabras con frecuencia <= 5 en el texto
        rare_words = sum(1 for count in token_counts.values() if count <= 5)
        pct_rare = (rare_words / len(set(tokens))) * 100
        
        # Palabras fuera de las 2000 más frecuentes
        advanced_words = sum(1 for token in tokens if token not in self.freq_vocab)
        pct_advanced = (advanced_words / len(tokens)) * 100
        
        return {
            'pct_rare_words': pct_rare,
            'pct_advanced_words': pct_advanced
        }
    
    def analyze_text(self, text: str) -> pd.Series:
        """
        Analiza un texto y calcula todas las métricas de complejidad léxica.
        
        Args:
            text: Texto a analizar
            
        Returns:
            Serie de pandas con las 11 métricas
        """
        tokens = self._preprocess_text(text)
        
        # Calcular todas las métricas
        richness = self._calculate_richness_metrics(tokens)
        surface = self._calculate_surface_complexity(tokens)
        readability = self._calculate_readability(text)
        advanced = self._calculate_advanced_lexicon(tokens)
        
        # Combinar resultados
        metrics = {**richness, **surface, **readability, **advanced}
        
        return pd.Series(metrics, index=self.METRIC_NAMES)


def build_lexical_profiles(
    texts: List[str],
    spacy_model: str = "es_core_news_md"
) -> Dict[str, pd.Series]:
    """
    Calcula las 11 métricas de complejidad léxica para una lista de textos.
    
    Args:
        texts: Lista de textos a analizar
        spacy_model: Modelo de spaCy a utilizar
        
    Returns:
        Diccionario con perfiles léxicos {id_texto: Serie con métricas}
    """
    analyzer = LexicalComplexityAnalyzer(spacy_model)
    profiles = {}
    
    for i, text in enumerate(texts):
        text_id = f"texto_{i+1:03d}"
        profiles[text_id] = analyzer.analyze_text(text)
    
    return profiles


def compare_lexical_profiles(
    profiles_A: Dict[str, pd.Series],
    profiles_B: Dict[str, pd.Series],
    metrics: List[str] = ["euclidean", "manhattan", "cosine"]
) -> pd.DataFrame:
    """
    Calcula distancias entre perfiles léxicos de dos grupos.
    
    Args:
        profiles_A: Perfiles del grupo A (textos conocidos)
        profiles_B: Perfiles del grupo B (textos dubitados)
        metrics: Lista de métricas de distancia a calcular
        
    Returns:
        DataFrame con distancias entre cada par <A, B>
    """
    # Crear DataFrame combinado
    all_profiles = {**profiles_A, **profiles_B}
    df = pd.DataFrame(all_profiles).T
    
    # Normalizar características (z-score)
    scaler = StandardScaler()
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df),
        index=df.index,
        columns=df.columns
    )
    
    # Separar grupos normalizados
    df_A = df_normalized.loc[list(profiles_A.keys())]
    df_B = df_normalized.loc[list(profiles_B.keys())]
    
    results = []
    
    for text_a in df_A.index:
        for text_b in df_B.index:
            row = {'text_A': text_a, 'text_B': text_b}
            
            vec_a = df_A.loc[text_a].values
            vec_b = df_B.loc[text_b].values
            
            if "euclidean" in metrics:
                row['euclidean'] = np.linalg.norm(vec_a - vec_b)
            
            if "manhattan" in metrics:
                row['manhattan'] = np.sum(np.abs(vec_a - vec_b))
            
            if "cosine" in metrics:
                dot_product = np.dot(vec_a, vec_b)
                norm_a = np.linalg.norm(vec_a)
                norm_b = np.linalg.norm(vec_b)
                
                if norm_a > 0 and norm_b > 0:
                    cosine_sim = dot_product / (norm_a * norm_b)
                    row['cosine'] = 1 - cosine_sim  # Distancia coseno
                else:
                    row['cosine'] = 1.0
            
            results.append(row)
    
    return pd.DataFrame(results)


def export_lexical_results(
    profiles: Dict[str, pd.Series],
    distances: pd.DataFrame,
    out_dir: Path,
    author: str = "",
    query: str = ""
) -> Dict[str, Path]:
    """
    Exporta resultados del análisis léxico a archivos CSV y PNG.
    
    Args:
        profiles: Perfiles léxicos de todos los textos
        distances: DataFrame con distancias entre textos
        out_dir: Directorio de salida
        
    Returns:
        Diccionario con rutas de archivos generados
    """
    if author and query:
        out_dir = Path(out_dir) / author / f"Resultados_{author}_{query}"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    files_created = {}
    
    # 1. Exportar perfiles a CSV
    profiles_df = pd.DataFrame(profiles).T
    profiles_path = out_dir / "lex_complexity_profiles.csv"
    profiles_df.to_csv(profiles_path, index=True, encoding='utf-8')
    files_created['profiles_csv'] = profiles_path
    
    # 2. Exportar distancias a CSV
    distances_path = out_dir / "lex_complexity_distances.csv"
    distances.to_csv(distances_path, index=False, encoding='utf-8')
    files_created['distances_csv'] = distances_path
    
    # 3. Crear gráficos individuales (barras horizontales normalizadas)
    # Normalizar perfiles para visualización (0-1)
    scaler_viz = StandardScaler()
    profiles_normalized = pd.DataFrame(
        scaler_viz.fit_transform(profiles_df),
        index=profiles_df.index,
        columns=profiles_df.columns
    )
    
    # Convertir a rango 0-1
    profiles_viz = profiles_normalized.copy()
    for col in profiles_viz.columns:
        min_val = profiles_viz[col].min()
        max_val = profiles_viz[col].max()
        if max_val > min_val:
            profiles_viz[col] = (profiles_viz[col] - min_val) / (max_val - min_val)
        else:
            profiles_viz[col] = 0.5
    
    individual_plots = {}
    
    for text_id in profiles.keys():
        fig, ax = plt.subplots(figsize=(10, 8))
        
        values = profiles_viz.loc[text_id]
        y_pos = np.arange(len(values))
        
        bars = ax.barh(y_pos, values, color='steelblue', alpha=0.7)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(values.index, fontsize=10)
        ax.set_xlabel('Valor Normalizado (0-1)', fontsize=12)
        ax.set_title(f'Perfil de Complejidad Léxica - {text_id}', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Añadir valores en las barras
        for i, (bar, value) in enumerate(zip(bars, values)):
            ax.text(value + 0.01, bar.get_y() + bar.get_height()/2, f'{value:.3f}',
                   va='center', fontsize=9)
        
        plt.tight_layout()
        
        plot_path = out_dir / f"profile_{text_id}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        individual_plots[f'plot_{text_id}'] = plot_path
    
    files_created.update(individual_plots)
    
    # 4. Crear heatmap de distancias euclidianas
    if 'euclidean' in distances.columns:
        # Crear matriz de distancias
        texts_A = sorted(distances['text_A'].unique())
        texts_B = sorted(distances['text_B'].unique())
        
        distance_matrix = np.zeros((len(texts_B), len(texts_A)))
        
        for _, row in distances.iterrows():
            i = texts_B.index(row['text_B'])
            j = texts_A.index(row['text_A'])
            distance_matrix[i, j] = row['euclidean']
        
        fig, ax = plt.subplots(figsize=(max(8, len(texts_A)), max(6, len(texts_B))))
        
        im = ax.imshow(distance_matrix, cmap='viridis', aspect='auto')
        
        ax.set_xticks(np.arange(len(texts_A)))
        ax.set_yticks(np.arange(len(texts_B)))
        ax.set_xticklabels(texts_A, rotation=45, ha='right')
        ax.set_yticklabels(texts_B)
        
        ax.set_xlabel('Textos Conocidos', fontsize=12)
        ax.set_ylabel('Textos Dubitados', fontsize=12)
        ax.set_title('Distancias Euclidianas entre Perfiles Léxicos', fontsize=14, fontweight='bold')
        
        # Añadir valores en las celdas
        for i in range(len(texts_B)):
            for j in range(len(texts_A)):
                text = ax.text(j, i, f'{distance_matrix[i, j]:.2f}',
                              ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im, ax=ax, label='Distancia Euclidiana')
        plt.tight_layout()
        
        heatmap_path = out_dir / "distances_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        files_created['heatmap'] = heatmap_path
    
    return files_created


def load_texts_from_directory(directory: Path, combine_subdirs: bool = False) -> Dict[str, str]:
    """Carga textos desde un directorio.

    Cuando ``combine_subdirs`` es ``True`` se combinan los archivos de cada
    subcarpeta y se devuelven bajo el nombre de la carpeta. Esto permite
    procesar estructuras como ``Dubitados/A1`` o similares.
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
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts[file_path.stem] = content
        except Exception:
            pass

    return texts


def analyze_authorship_corpus(
    known_texts: List[str],
    disputed_texts: List[str],
    output_dir: Union[str, Path],
    spacy_model: str = "es_core_news_md"
) -> Dict[str, Path]:
    """
    Función principal para análisis completo de atribución de autoría.
    
    Args:
        known_texts: Lista de textos de autoría conocida
        disputed_texts: Lista de textos dubitados
        output_dir: Directorio donde guardar resultados
        spacy_model: Modelo de spaCy a utilizar
        
    Returns:
        Diccionario con rutas de todos los archivos generados
    """
    print("🔍 Iniciando análisis de complejidad léxica...")
    
    # Calcular perfiles
    print("📊 Calculando perfiles léxicos...")
    profiles_known = build_lexical_profiles(known_texts, spacy_model)
    profiles_disputed = build_lexical_profiles(disputed_texts, spacy_model)
    
    # Renombrar para claridad
    profiles_known_renamed = {f"conocido_{k}": v for k, v in profiles_known.items()}
    profiles_disputed_renamed = {f"dubitado_{k}": v for k, v in profiles_disputed.items()}
    
    all_profiles = {**profiles_known_renamed, **profiles_disputed_renamed}
    
    # Calcular distancias
    print("📏 Calculando distancias entre perfiles...")
    distances = compare_lexical_profiles(
        profiles_known_renamed,
        profiles_disputed_renamed,
        metrics=["euclidean", "manhattan", "cosine"]
    )
    
    # Exportar resultados
    print("💾 Exportando resultados...")
    files_created = export_lexical_results(all_profiles, distances, Path(output_dir))
    
    print(f"✅ Análisis completado. {len(files_created)} archivos generados en {output_dir}")
    
    return files_created


if __name__ == "__main__":
    # Ejemplo de uso
    
    # Textos de ejemplo (en un caso real, cargar desde archivos)
    textos_conocidos = [
        """El análisis de la complejidad léxica constituye una herramienta fundamental 
        en los estudios de atribución de autoría. Las métricas contemporáneas permiten 
        caracterizar el estilo lingüístico de manera objetiva y cuantificable.""",
        
        """La riqueza vocabular y la sofisticación sintáctica representan indicadores 
        esenciales para determinar patrones autoriales. Mediante algoritmos avanzados 
        podemos identificar particularidades estilísticas individuales."""
    ]
    
    textos_dubitados = [
        """Los indicadores de diversidad léxica ofrecen perspectivas valiosas sobre 
        la autoría textual. Las metodologías modernas facilitan la identificación 
        de características distintivas en la producción discursiva."""
    ]
    
    # Ejecutar análisis completo
    archivos_generados = analyze_authorship_corpus(
        known_texts=textos_conocidos,
        disputed_texts=textos_dubitados,
        output_dir="./resultados_lexicos",
        spacy_model="es_core_news_md"
    )
    
    print("\nArchivos generados:")
    for nombre, ruta in archivos_generados.items():
        print(f"  {nombre}: {ruta}")
