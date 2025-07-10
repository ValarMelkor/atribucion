# Sistema Forense de Atribución de Autoría

Este proyecto reúne tres módulos de análisis de texto en español:

- **N-gramas**: perfiles de carácter y palabra.
- **Análisis sintáctico**: distribución de etiquetas gramaticales y dependencias.
- **Complejidad léxica**: métricas de diversidad y sofisticación del vocabulario.

El script `full_analysis.py` ejecuta los tres módulos de forma consecutiva y reúne los resultados en un solo directorio.

Los cargadores de texto aceptan archivos con extensiones `.txt`, `.md` y `.text`.

## Organización de carpetas

1. **Textos conocidos** (`--known`)
   - Debe ser un directorio que contenga **subdirectorios por autor**. Cada subdirectorio incluye los archivos de texto (`.txt`, `.md` o `.text`) que pertenecen a dicho autor.
   - Ejemplo:
     ```
     conocidos/
       autor1/
         a1.txt
         a2.txt
       autor2/
         b1.txt
     ```

2. **Textos dubitados** (`--query`)
   - Directorio con archivos de texto individuales (`.txt`, `.md` o `.text`). No requiere subdirectorios.
   - Ejemplo:
     ```
     dubitados/
       q1.txt
       q2.txt
     ```

3. **Salida** (`--out`)
   - Directorio donde se guardarán los resultados generados por el script. Se crearán subcarpetas para cada tipo de análisis.

## Ejecución rápida

1. Asegúrate de tener Python instalado.
2. (Opcional pero recomendado) crea un entorno virtual para evitar problemas con
   entornos gestionados externamente:

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Ejecuta el script principal desde la línea de comandos:

   ```bash
   python full_analysis.py --known ruta/conocidos --query ruta/dubitados --out salida
   ```

    Al iniciarse, `full_analysis.py` comprobará e instalará automáticamente los paquetes indicados en `requirements.txt` si alguno falta.

    Si la instalación automática falla (por ejemplo por el mensaje
    `externally-managed-environment`), instala las dependencias manualmente con:

    ```bash
    pip install -r requirements.txt
    ```

Los resultados incluirán archivos CSV con perfiles, gráficas PNG y un archivo `full_summary.txt` con las rutas de todos los artefactos generados.

## Estructura de resultados

Al ejecutarse, cada par **autor - texto dubitado** genera un directorio con el patrón `Resultados_<autor>_<query>` dentro de la ruta de salida. En él se crean subcarpetas por módulo:

```
salida/
  autor1/
    Resultados_autor1_q1/
      ngrams/
      syntax/
      lexical/
    Resultados_autor1_q2/
      ...
  autor2/
    Resultados_autor2_q1/
      ...
```

Cada subcarpeta contiene archivos CSV con perfiles y distancias, gráficas PNG, un resumen en Markdown (`*_summary.md`) y, cuando `pandoc` o `fpdf` están disponibles, su versión en PDF.

## Flujo interactivo de `full_analysis.py`

El script principal no usa argumentos en la línea de comandos. Al ejecutarlo, se muestra un asistente que solicita rutas y parámetros clave:

1. Directorios con textos conocidos y dubitados.
2. Ubicación donde guardar los resultados.
3. Opciones de los tres módulos (rangos de n‑gramas, modelo spaCy, etc.).

Tras responder a las preguntas, comienza el procesamiento y al final se crea `full_summary.txt` con las rutas de todos los resúmenes Markdown/PDF generados.
