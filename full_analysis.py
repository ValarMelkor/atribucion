#!/usr/bin/env python3
"""Ejecuta N-gramas, análisis sintáctico y complejidad léxica de forma unificada."""

from pathlib import Path
import importlib.util
import subprocess
import sys


def _run_pip(args):
    """Run pip with given arguments and handle PEP 668 errors."""
    cmd = [sys.executable, "-m", "pip", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode == 0:
        return
    if "externally-managed-environment" in proc.stderr:
        print("Entorno administrado externamente, reintentando con --user...")
        cmd.insert(3, "--user")
        subprocess.check_call(cmd)
    else:
        raise RuntimeError(proc.stderr)


def install_requirements() -> None:
    """Instala los paquetes listados en requirements.txt si faltan."""
    req_file = Path(__file__).with_name("requirements.txt")
    if not req_file.exists():
        return

    try:
        import pkg_resources
    except ImportError:
        try:
            _run_pip(["install", "-r", str(req_file)])
        except Exception as e:  # pragma: no cover - solo se usa en ejecución directa
            print(f"Advertencia: no se pudo instalar dependencias automáticamente: {e}")
        return

    try:
        with open(req_file, "r", encoding="utf-8") as f:
            packages = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        missing = []
        for pkg in packages:
            try:
                pkg_resources.require(pkg)
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                missing.append(pkg)

        if missing:
            print(f"Instalando dependencias: {', '.join(missing)}")
            _run_pip(["install", *missing])
    except Exception as e:  # pragma: no cover - solo se usa en ejecución directa
        print(f"Advertencia: no se pudo instalar dependencias automáticamente: {e}")




# Instalar dependencias antes de cargar los módulos
install_requirements()

# Cargar módulos con nombres que contienen espacios

def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module

ngram_mod = load_module(Path(__file__).with_name('N-gramas.py'), 'ngrams')
syntax_mod = load_module(Path(__file__).with_name('analisis sintactico.py'), 'syntax')
lex_mod = load_module(Path(__file__).with_name('complejidad lexica.py'), 'lexical')


def _prompt(msg: str, default: str = "") -> str:
    """Helper to prompt the user with a default value."""
    prompt = f"{msg} [{'Enter' if default == '' else default}]: "
    resp = input(prompt).strip()
    return resp or default


def main() -> None:
    print("=== Análisis Forense Interactivo ===")
    known_dir = _prompt("Directorio con textos conocidos")
    query_dir = _prompt("Directorio con textos dubitados")
    out_dir = Path(_prompt("Directorio de salida", "resultados"))

    print("\n--- Parámetros de n-gramas ---")
    print("Opciones de nivel: 'char' = caracteres, 'word' = palabras")
    levels_in = _prompt("Niveles (separados por espacio)", "char")
    levels = levels_in.split()

    rng_in = _prompt("Rango n-gram (min max, vacío=defecto)")
    n_range = tuple(map(int, rng_in.split())) if rng_in else None
    top_k_ng = int(_prompt("top_k n-gramas", "500"))
    min_len = int(_prompt("Longitud mínima de texto", "300"))

    print("\n--- Parámetros de análisis sintáctico ---")
    print("Modelos spaCy comunes: es_core_news_sm, es_core_news_md, es_core_news_lg")
    spacy_model_syn = _prompt("Modelo spaCy", "es_core_news_md")
    top_k_syn = int(_prompt("top_k características sintácticas", "500"))

    print("\n--- Parámetros de complejidad léxica ---")
    spacy_model_lex = _prompt("Modelo spaCy", "es_core_news_md")

    out_dir.mkdir(parents=True, exist_ok=True)

    known_texts = ngram_mod.load_texts_by_author(Path(known_dir))
    query_groups = ngram_mod.load_texts_from_directory(Path(query_dir), combine_subdirs=True)

    summary_entries = []

    for query_name, query_text in query_groups.items():
        for author, known_text in known_texts.items():
            pair_out = out_dir / author / query_name

            # N-gramas
            analyzer = ngram_mod.ForensicAnalyzer(min_text_length=min_len)
            pk = analyzer.build_profiles({author: known_text}, level=levels, n_range=n_range, top_k=top_k_ng)
            pq = analyzer.build_profiles({query_name: query_text}, level=levels, n_range=n_range, top_k=top_k_ng)
            dist_ng = analyzer.compare_profiles(pk, pq)
            files_ng = analyzer.export_results(pk, pq, dist_ng, pair_out / 'ngrams')

            # Sintaxis
            syntax = syntax_mod.SyntaxForensics(model_name=spacy_model_syn, top_k=top_k_syn)
            sk = syntax.build_syntax_profiles({author: known_text})
            sq = syntax.build_syntax_profiles({query_name: query_text})
            dist_syn = syntax.compare_syntax(sk, sq)
            files_syn = syntax.export_syntax_results({**sk, **sq}, dist_syn, pair_out / 'syntax')

            # Léxica
            lex_an = lex_mod.LexicalComplexityAnalyzer(spacy_model_lex)
            lk = {author: lex_an.analyze_text(known_text)}
            lq = {query_name: lex_an.analyze_text(query_text)}
            dist_lex = lex_mod.compare_lexical_profiles(lk, lq)
            files_lex = lex_mod.export_lexical_results({**lk, **lq}, dist_lex, pair_out / 'lexical')

            for d in (files_ng, files_syn, files_lex):
                for p in d.values():
                    if isinstance(p, list):
                        for sub in p:
                            summary_entries.append(str(sub))
                    else:
                        summary_entries.append(str(p))

    summary_path = out_dir / 'full_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('ARCHIVOS GENERADOS\n')
        for path in summary_entries:
            if path.endswith('.pdf') or path.endswith('.md'):
                f.write(f"{path}\n")

    print(f"Resumen guardado en {summary_path}")


if __name__ == '__main__':
    main()
