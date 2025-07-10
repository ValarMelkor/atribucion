#!/usr/bin/env python3
"""Ejecuta N-gramas, análisis sintáctico y complejidad léxica de forma unificada."""

import argparse
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

    known_texts = ngram_mod.load_texts_by_author(Path(args.known))
    query_groups = ngram_mod.load_texts_from_directory(Path(args.query), combine_subdirs=True)

    n_range = tuple(args.n_range) if args.n_range else None

    summary_entries = []

    for query_name, query_text in query_groups.items():
        for author, known_text in known_texts.items():
            # N-gramas
            analyzer = ngram_mod.ForensicAnalyzer(min_text_length=args.min_length)
            pk = analyzer.build_profiles({author: known_text}, level=args.levels, n_range=n_range, top_k=args.top_k)
            pq = analyzer.build_profiles({query_name: query_text}, level=args.levels, n_range=n_range, top_k=args.top_k)
            dist_ng = analyzer.compare_profiles(pk, pq)
            files_ng = analyzer.export_results(pk, pq, dist_ng, out_dir / 'ngrams', author=author, query=query_name)

            # Sintaxis
            syntax = syntax_mod.SyntaxForensics()
            sk = syntax.build_syntax_profiles({author: known_text})
            sq = syntax.build_syntax_profiles({query_name: query_text})
            dist_syn = syntax.compare_syntax(sk, sq)
            files_syn = syntax.export_syntax_results({**sk, **sq}, dist_syn, out_dir / 'syntax', author=author, query=query_name)

            # Léxica
            lex_an = lex_mod.LexicalComplexityAnalyzer()
            lk = {author: lex_an.analyze_text(known_text)}
            lq = {query_name: lex_an.analyze_text(query_text)}
            dist_lex = lex_mod.compare_lexical_profiles(lk, lq)
            files_lex = lex_mod.export_lexical_results({**lk, **lq}, dist_lex, out_dir / 'lexical', author=author, query=query_name)

            for d in (files_ng, files_syn, files_lex):
                for p in d.values():
                    summary_entries.append(str(p))

    summary_path = out_dir / 'full_summary.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write('ARCHIVOS GENERADOS\n')
        for path in summary_entries:
            f.write(f"{path}\n")

    print(f"Resumen guardado en {summary_path}")


if __name__ == '__main__':
    main()
