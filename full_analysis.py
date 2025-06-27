#!/usr/bin/env python3
"""Ejecuta N-gramas, análisis sintáctico y complejidad léxica de forma unificada."""

import argparse
from pathlib import Path
import importlib.util
import subprocess
import sys


def install_requirements() -> None:
    """Instala los paquetes listados en requirements.txt si faltan."""
    req_file = Path(__file__).with_name("requirements.txt")
    if not req_file.exists():
        return

    try:
        import pkg_resources

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
            subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
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


def main() -> None:
    parser = argparse.ArgumentParser(description='Análisis forense combinado')
    parser.add_argument('--known', required=True, help='Dir textos conocidos (subdirectorios=autor)')
    parser.add_argument('--query', required=True, help='Dir textos dubitados')
    parser.add_argument('--out', required=True, help='Directorio de salida')
    parser.add_argument('--levels', nargs='+', choices=['char', 'word'], default=['char'],
                        help='Niveles de n-gramas a calcular')
    parser.add_argument('--n-range', nargs=2, type=int, metavar=('MIN','MAX'))
    parser.add_argument('--top-k', type=int, default=500)
    parser.add_argument('--min-length', type=int, default=300)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- N-GRAMAS ----
    analyzer = ngram_mod.ForensicAnalyzer(min_text_length=args.min_length)
    known_texts = ngram_mod.load_texts_by_author(Path(args.known))
    query_texts = ngram_mod.load_texts_from_directory(Path(args.query))
    n_range = tuple(args.n_range) if args.n_range else None
    pk = analyzer.build_profiles(known_texts, level=args.levels, n_range=n_range, top_k=args.top_k)
    pq = analyzer.build_profiles(query_texts, level=args.levels, n_range=n_range, top_k=args.top_k)
    dist_ng = analyzer.compare_profiles(pk, pq)
    files_ng = analyzer.export_results(pk, pq, dist_ng, out_dir / 'ngrams')

    # ---- SINTAXIS ----
    syntax = syntax_mod.SyntaxForensics()
    sk = syntax.build_syntax_profiles(known_texts)
    sq = syntax.build_syntax_profiles(query_texts)
    dist_syn = syntax.compare_syntax(sk, sq)
    files_syn = syntax.export_syntax_results({**sk, **sq}, dist_syn, out_dir / 'syntax')

    # ---- LÉXICA ----
    lex_an = lex_mod.LexicalComplexityAnalyzer()
    lk = {n: lex_an.analyze_text(t) for n, t in known_texts.items()}
    lq = {n: lex_an.analyze_text(t) for n, t in query_texts.items()}
    dist_lex = lex_mod.compare_lexical_profiles(lk, lq)
    files_lex = lex_mod.export_lexical_results({**lk, **lq}, dist_lex, out_dir / 'lexical')

    # ---- Resumen ----
    summary_path = out_dir / 'full_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('ARCHIVOS GENERADOS\n')
        for d in (files_ng, files_syn, files_lex):
            for k, p in d.items():
                f.write(f"{k}: {p}\n")

    print(f"Resumen guardado en {summary_path}")


if __name__ == '__main__':
    main()
