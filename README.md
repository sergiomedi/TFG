# GRAIA — Granada Retrieval-Augmented Intelligent Assistant

Implementación de referencia del sistema conversacional RAG descrito en el TFG
*"Desarrollo y evaluación de un sistema conversacional inteligente con arquitectura
RAG para el dominio académico de la ETSIIT"* (Sergio Medina Muñoz, UGR, 2026).

---

## 1. Visión general

GRAIA responde consultas académicas de estudiantes de la ETSIIT-UGR(horarios,
calendario, exámenes, plan de estudios, normativa, TFG, movilidad, etc.) 
con las asignaturas acotadas al Grado de Ingeniería Informática, mediante
una arquitectura *Retrieval-Augmented Generation* **íntegramente local**, sin
servicios de pago ni envío de datos a terceros:

- **Recuperación híbrida**: densa con FAISS sobre *embeddings* `multilingual-e5-base`
  (768 dim, producto interno sobre vectores normalizados) + léxica con BM25, fusionadas
  con *Reciprocal Rank Fusion* (RRF).
- **Reranking neuronal** con un *cross-encoder* multilingüe (`mmarco-mMiniLMv2`) como
  tercera etapa, que decide la precisión final del contexto.
- **Contextual Retrieval** (Anthropic, 2024): cada *chunk* se enriquece offline con un
  breve contexto generado por el LLM y cacheado en disco.
- **Generación** con Llama 3.1 8B (Q4_K_M) servido por Ollama, decodificación *greedy*
  (`temperature=0`) para máxima reproducibilidad factual.
- **Citas verificadas**: validación *post-hoc* de los marcadores `[n]` y recuperación de
  fuentes por solapamiento léxico, con trazabilidad al documento y fragmento originales.
- **Control de calidad en 3 capas**: *gate* de ámbito (out-of-domain) previo a la
  generación, filtrado por umbral de similitud y deduplicación de contenido redundante.

La justificación completa de cada decisión (modelo, *chunking*, umbrales, etc.) está en
el Capítulo 5 de la memoria; la evaluación experimental, en el Capítulo 7.

---

## 2. Estructura del repositorio

```
codigo/
├── graia/                    # Paquete principal (importable)
│   ├── ingesta/              # Fetcher, Parser (tablas→MD, OCR), Cleaner, Chunker, Contextualizer
│   ├── indexacion/           # Embeddings E5 + índice FAISS
│   ├── recuperacion/         # Retriever híbrido (BM25+denso), MMR, reranker, router, scope gate
│   ├── generacion/           # Cliente Ollama, prompt builder, validación/recuperación de citas, dedup
│   ├── interfaz/             # Streamlit (app.py) + historial conversacional
│   └── registro/             # Logger JSONL estructurado
├── config/
│   ├── default.yaml          # Configuración declarativa (parámetros del Cap. 5)
│   └── sources.yaml          # Fuentes web a rastrear (crawl)
├── scripts/                  # Orquestación: build, index, review, evaluate, etc.
├── tests/                    # Pruebas unitarias y de integración (pytest)
├── eval/                     # Dataset de evaluación + informes .json
├── data/                     # Corpus y artefactos (fuera de control de versiones)
│   ├── raw/                  # HTML/PDF originales descargados
│   ├── processed/            # Corpus normalizado (corpus.jsonl) + caché de contexto
│   ├── verificado/           # Fuentes verificadas a mano (horario, calendario, exámenes…)
│   └── index/                # Índice FAISS + BM25 + metadatos
├── requirements.txt          # Dependencias de la aplicación
└── requirements-ragas.txt    # Dependencias SOLO para la evaluación RAGAS (venv aparte)
```

---

## 3. Requisitos

- **Python ≥ 3.10**
- **GPU NVIDIA** con ≥ 8 GB de VRAM (desarrollado en RTX 4070 Super, 12 GB). Funciona en
  CPU, pero la recuperación y la generación son mucho más lentas.
- **[Ollama](https://ollama.com)** en ejecución local con, al menos, el modelo
  `llama3.1:8b-instruct-q4_K_M`. Para la evaluación comparativa también
  `qwen2.5:14b-instruct` y `gemma2:9b`.
- **Tesseract OCR** (binario de sistema, *no* se instala con pip) + idioma español `spa`,
  necesario para leer PDFs escaneados durante la ingesta:
  - **Windows 11**: instalador de [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
    (marcar *Spanish*) y añadir `C:\Program Files\Tesseract-OCR` al PATH.
  - **Linux**: `sudo apt install tesseract-ocr tesseract-ocr-spa`
  - **macOS**: `brew install tesseract tesseract-lang`
  - Comprobar: `tesseract --list-langs` debe incluir `spa`.

---

## 4. Instalación

El entorno principal de la aplicación es el **Python de la propia máquina** (instalación
global); todos los comandos se ejecutan con ese intérprete. La única excepción es la
evaluación RAGAS, que va en un entorno aislado `.venv-ragas` (ver sección 6.3) para no
contaminar las dependencias de la app.

```powershell
# Windows 11 (PowerShell). Equivalentes Linux/macOS indicados en comentarios.
cd codigo
pip install -r requirements.txt
copy .env.example .env          # Linux/macOS: cp .env.example .env

# GPU (recomendado): instalar la build de torch con CUDA DESPUÉS del requirements.
# Si se omite, PyPI instala el torch de CPU y todo va mucho más lento.
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Descargar el LLM (y, para evaluar, los demás modelos)
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull qwen2.5:14b-instruct      # opcional (evaluación comparativa)
ollama pull gemma2:9b                 # opcional (evaluación comparativa)
```

El fichero `.env` permite ajustar `OLLAMA_HOST`, el modelo LLM/embeddings, la ruta de
configuración y el nivel de *logging*. Los parámetros del sistema (umbrales, *chunking*,
`k`, etc.) se editan en `config/default.yaml`.

> **Opcional (mayor aislamiento de paquetes):** si prefieres no instalar las dependencias
> en el Python global, puedes crear un entorno virtual como entorno principal y trabajar
> dentro de él. Solo afecta a cómo se invoca `python`/`pip`; el resto de comandos del
> README es idéntico.
>
> ```powershell
> python -m venv .venv ; .venv\Scripts\activate   # Linux/macOS: python3 -m venv .venv && source .venv/bin/activate
> pip install -r requirements.txt
> ```

> Todos los comandos de las secciones siguientes se ejecutan **desde la propia carpeta del sistema asociada a `codigo/` en Estructura del repositorio**.

---
> [!IMPORTANT]
> A partir de este punto ya estás preparado para iniciar el sistema con
> `streamlit run graia\interfaz\app.py`. Lo que viene a continuación es una guía de
> construcción y evaluación que **reproduce** lo que el repositorio ya trae hecho. Ten en
> cuenta que, una vez construido el corpus de forma automática, parte de su contenido fue
> depurado y eliminado a mano.
---

## 5. Guía de ejecución paso a paso (de cero a asistente)

### Paso 0 — Verificar el entorno

Comprueba que parser, OCR y *chunker* funcionan en tu máquina antes de procesar nada:

```powershell
python scripts\verify_pipeline.py
```

### Paso 1 — Construir el corpus (`build_corpus.py`)

Rastrea las webs de la ETSIIT/UGR de `config/sources.yaml`, descarga páginas y PDFs
permitidos, los parsea (HTML sin duplicados, **tablas → Markdown**, **OCR** en PDFs
escaneados), los limpia y escribe `data/processed/corpus.jsonl`. Es el paso más lento
(usa red).

```powershell
python scripts\build_corpus.py                              # crawl completo
python scripts\build_corpus.py --categories calendario normativa   # solo categorías
python scripts\build_corpus.py --max-pages 50              # límite para pruebas rápidas
python scripts\build_corpus.py --dry-run                   # solo descubrir enlaces (diagnóstico)
```

Salida: `data/processed/corpus.jsonl`, `data/processed/crawl_report.txt` y
`data/raw/crawl_manifest.json`.

### Paso 2 — Fuentes verificadas

Algunas fuentes (rejillas visuales como el horario o el calendario, y datos cuya extracción
automática es poco fiable) se mantienen como **documentos verificados a mano** en
`data/verificado/`, que son la única fuente de verdad para esa información. Se inyectan en
el corpus con `inject_doc.py`, que **conserva la URL oficial original para las citas** y
crea una copia de seguridad `corpus.jsonl.bak` antes de modificar. Requiere haber ejecutado
`build_corpus.py` antes.

Las fuentes verificadas incluidas en `data/verificado/` son:

| Fichero | Contenido |
|---|---|
| `horario_2025-2026.txt` | Horario del Grado (teoría y prácticas por grupo/subgrupo y aula). |
| `calendario_TFG_2025-2026.txt` | Calendario académico y plazos del TFG. |
| `calendario_ETSIIT_2025-2026.txt` | Calendario académico general de la ETSIIT. |
| `examenes_2025-2026.txt` | Fechas de exámenes (convocatorias ordinaria y extraordinaria). |
| `plan_estudios_2025-2026.txt` | Plan de estudios (asignaturas por curso y semestre). |
| `plazos_interes_2025-2026.txt` | Plazos administrativos de interés. |
| `tutores_movilidad_ETSIIT.txt` | Tutores de programas de movilidad. |

Se inyecta cada una con su URL oficial y su título. Parámetros: `--text` (fichero
verificado), `--url` (enlace para citar), `--tipo` (`horario`, `calendario` o
`plan_estudios`), `--title` (título mostrado en las citas) y `--category` (opcional; por
defecto igual que `--tipo`). Para fuentes que no encajan exactamente en un `--tipo`
(exámenes, plazos, movilidad) se usa el `--tipo` más cercano (`calendario`) y se afina la
etiqueta con `--category`. Los comandos completos para las 7 fuentes son:

```powershell
python scripts\inject_doc.py --text data\verificado\horario_2025-2026.txt --tipo horario ^
  --url "https://etsiit.ugr.es/sites/centros/etsiit/public/inline-files/Horarios%20GII%20%2825-26%29.pdf" ^
  --title "Horarios Grado en Ingeniería Informática 2025-2026"

python scripts\inject_doc.py --text data\verificado\calendario_TFG_2025-2026.txt --tipo calendario ^
  --url "https://etsiit.ugr.es/sites/centros/etsiit/public/inline-files/Calendario%20TFG%202025-2026.pdf" ^
  --title "Calendario TFG 2025-2026"

python scripts\inject_doc.py --text data\verificado\calendario_ETSIIT_2025-2026.txt --tipo calendario ^
  --url "https://etsiit.ugr.es/sites/centros/etsiit/public/inline-files/Calendario%20Academico%202025-2026.pdf" ^
  --title "Calendario ETSIIT 2025-2026"

python scripts\inject_doc.py --text data\verificado\examenes_2025-2026.txt --tipo calendario ^
  --url "https://etsiit.ugr.es/sites/centros/etsiit/public/inline-files/CalendarioExamenes25-26-GII.pdf" ^
  --title "Calendario de examenes 2025-2026"

python scripts\inject_doc.py --text data\verificado\plan_estudios_2025-2026.txt --tipo plan_estudios ^
  --url "https://grados.ugr.es/informatica/docencia/plan-estudios" ^
  --title "Plan de estudios Grado Ing. Informática 2025-2026"

python scripts\inject_doc.py --text data\verificado\plazos_interes_2025-2026.txt --tipo calendario --category tramites ^
  --url "https://etsiit.ugr.es/sites/centros/etsiit/public/inline-files/PLAZOS%20DE%20INTERES%20CURSO%202025-26_1.pdf" ^
  --title "Plazos de interés del curso 2025-2026"

python scripts\inject_doc.py --text data\verificado\tutores_movilidad_ETSIIT.txt --tipo calendario --category movilidad ^
  --url "https://etsiit.ugr.es/sites/centros/etsiit/public/ficheros/docmovilidad/Tutores%20movilidad%2026-27.pdf" ^
  --title "Tutores de movilidad internacional de la ETSIIT (Erasmus+ y SICUE)"
```

> En PowerShell el carácter de continuación de línea es `^` (arriba) o `` ` ``; en
> Linux/macOS es `\`.

### Paso 3 — Revisar y limpiar el corpus (`review_corpus.py`) *(opcional pero recomendado)*

```powershell
python scripts\review_corpus.py                       # informe (nº docs por categoría)
python scripts\review_corpus.py --full                # vista previa del contenido
python scripts\review_corpus.py --category normativa  # filtrar por categoría
python scripts\review_corpus.py --search "TFG"        # buscar término
python scripts\review_corpus.py --export-csv          # exportar a CSV para Excel

# Limpieza (usa SIEMPRE --dry-run antes de un --remove-*)
python scripts\review_corpus.py --remove-url "https://www.ugr.es/universidad/normativa"
python scripts\review_corpus.py --remove-min-chars 100
python scripts\review_corpus.py --remove-search "2023" --dry-run
python scripts\review_corpus.py --add-url "https://etsiit.ugr.es/..." --add-category normativa

python scripts\analyze_corpus.py                      # (opcional) análisis de calidad
```

### Paso 4 — Indexar (`index_corpus.py`)

Fragmenta los documentos (horario y calendario: un registro = un *chunk*; el resto,
*chunking* recursivo), aplica **Contextual Retrieval** (cacheado por hash del texto),
genera los *embeddings* E5 y construye los índices **FAISS** + **BM25** en `data/index/`.

```powershell
python scripts\index_corpus.py            # indexación completa
python scripts\index_corpus.py --dry-run  # solo estadísticas, sin generar índice
```

> La primera indexación con Contextual Retrieval llama a Ollama una vez por *chunk*
> (lento la primera vez; luego reutiliza `data/processed/context_cache.json`). Para una
> prueba rápida, poner `contextual_retrieval.enabled: false` en `config/default.yaml`.

### Paso 5 — Lanzar el asistente (`app.py`)

```powershell
streamlit run graia\interfaz\app.py
```

### Resumen del flujo completo

```powershell
python scripts\build_corpus.py                                    # 1) Corpus
python scripts\inject_doc.py --text data\verificado\horario_2025-2026.txt --tipo horario --url "..." --title "..."   # 2) Verificadas
python scripts\inject_doc.py --text data\verificado\calendario_TFG_2025-2026.txt --tipo calendario --url "..." --title "..."
python scripts\review_corpus.py                                   # 3) Revisión (opcional)
python scripts\index_corpus.py                                    # 4) Indexar
streamlit run graia\interfaz\app.py                               # 5) Lanzar
```

---

## 6. Evaluación (Capítulo 7)

La batería de evaluación tiene **tres bloques** sobre el mismo dataset
(`eval/eval_dataset.jsonl`, 74 preguntas: factuales, de agregación y de
no-información/fuera de ámbito). Requisitos: Ollama en ejecución, modelos descargados e
índice construido (`python scripts\index_corpus.py`).

### 6.1. Evaluación del sistema RAG (`evaluate.py`)

Lanza las preguntas por el **pipeline completo** y mide *recall* de recuperación, acierto
de respuesta, rechazos correctos, fugas de razonamiento y latencia.

```powershell
python scripts\evaluate.py --model llama3.1:8b-instruct-q4_K_M --out eval\rag_llama.json
python scripts\evaluate.py --model qwen2.5:14b-instruct        --out eval\rag_qwen.json
python scripts\evaluate.py --model gemma2:9b                   --out eval\rag_gemma.json
```

Opciones útiles: `--limit N` (solo N preguntas), `--show` (imprime cada respuesta),
`--dataset` (otro fichero). Deja terminar las 74 preguntas para que la comparación sea
sobre el mismo *n*.

### 6.2. Evaluación *baseline* sin RAG (`evaluate_baseline.py`)

Mide H2: el mismo LLM respondiendo **sin** recuperar contexto (solo conocimiento
paramétrico). Añade la métrica `hallucination_noinfo` (proporción de preguntas sin
respuesta en el corpus que el modelo no rehúsa, es decir, inventa).

```powershell
python scripts\evaluate_baseline.py --model llama3.1:8b-instruct-q4_K_M --out eval\base_llama.json
python scripts\evaluate_baseline.py --model qwen2.5:14b-instruct        --out eval\base_qwen.json
python scripts\evaluate_baseline.py --model gemma2:9b                   --out eval\base_gemma.json
```

### 6.3. Evaluación RAGAS (`faithfulness` / `answer_relevancy`) — **en venv aislado**

RAGAS arrastra `datasets/langchain/pyarrow/transformers` que, mezclados con `torch+faiss`
bajo Streamlit, rompen la aplicación. Por eso se hace en **dos fases** y RAGAS se instala
**SOLO** en `.venv-ragas`, nunca en el entorno de la app.

> **Regla de oro:** no instales `ragas` en tu entorno global. El `.venv-ragas` es un
> cajón aislado; tu entorno de siempre (donde corre la app) queda intacto.

**Fase A — exportar entradas (en tu entorno global, el de siempre).** Ejecuta el pipeline
RAG y vuelca por pregunta la respuesta y los contextos recuperados. No instala nada nuevo.

```powershell
python scripts\export_for_ragas.py --model llama3.1:8b-instruct-q4_K_M --out eval\ragas_input_llama.jsonl
python scripts\export_for_ragas.py --model gemma2:9b                   --out eval\ragas_input_gemma.jsonl
python scripts\export_for_ragas.py --model qwen2.5:14b-instruct        --out eval\ragas_input_qwen.jsonl
```

> Por defecto usa `eval\eval_dataset.jsonl` (las 74 preguntas, el mismo conjunto de la
> evaluación del Cap. 7), de modo que el denominador coincide. Puedes forzar otro fichero
> con `--dataset`.

**Crear el entorno aislado de RAGAS (una sola vez):**

```powershell
python -m venv .venv-ragas
.\.venv-ragas\Scripts\python.exe -m pip install --upgrade pip
.\.venv-ragas\Scripts\python.exe -m pip install -r requirements-ragas.txt
```

En Linux/macOS: `python3 -m venv .venv-ragas` y luego
`.venv-ragas/bin/python -m pip install -r requirements-ragas.txt`.

**Fase B — calcular RAGAS (con el python del `.venv-ragas`):**

```powershell
.\.venv-ragas\Scripts\python.exe scripts\evaluate_ragas.py --in eval\ragas_input_llama.jsonl --out eval\ragas_llama.json
.\.venv-ragas\Scripts\python.exe scripts\evaluate_ragas.py --in eval\ragas_input_gemma.jsonl --out eval\ragas_gemma.json
.\.venv-ragas\Scripts\python.exe scripts\evaluate_ragas.py --in eval\ragas_input_qwen.jsonl  --out eval\ragas_qwen.json
```

Por defecto calcula `faithfulness` (no necesita embeddings). Para añadir
`answer_relevancy` (requiere `ollama pull nomic-embed-text`):

```powershell
.\.venv-ragas\Scripts\python.exe scripts\evaluate_ragas.py --in eval\ragas_input_llama.jsonl --out eval\ragas_llama.json --metrics faithfulness,answer_relevancy
```

El juez por defecto es `llama3.1:8b-instruct-q4_K_M`; para una comparación justa fija el
mismo juez para todos los modelos con `--judge qwen2.5:14b-instruct` y sube el límite con
`--timeout 600` si Qwen va lento.

> Nota metodológica: algunos `faithfulness=0` son falsos negativos del juez (no pudo
> descomponer la respuesta); se ignoran en la media. Si son muchos, usa un juez más capaz.

---

## 7. Pruebas

```powershell
pytest                 # toda la batería (tests/)
pytest -q              # salida resumida
pytest --cov=graia     # con cobertura
pytest tests\test_recuperacion.py   # un módulo concreto
```

---

## 8. Scripts de diagnóstico

| Script | Para qué sirve |
|---|---|
| `scripts\verify_pipeline.py` | Comprueba que parser, OCR y *chunker* funcionan en la máquina. |
| `scripts\debug_retrieval.py` | Inspecciona qué *chunks* recupera una consulta y sus *scores*. |
| `scripts\debug_gate.py` | Depura el *gate* de ámbito (margen del *cross-encoder* y decisión). |
| `scripts\analyze_corpus.py` | Estadísticas de calidad del corpus (longitudes, categorías…). |

---

## 9. Referencias

- Configuración y justificación de parámetros: `config/default.yaml` y Capítulo 5 de la memoria.
- Diseño experimental, metodología de evaluación y resultados: Capítulo 7 de la memoria (los datos de partida están en `eval/`: dataset `eval_dataset.jsonl` e informes `rag_*`, `base_*`, `ragas_*` sin necesidad de ejecutar dichas evaluaciones en una máquina ya que tardan bastante).
- Autor: Sergio Medina Muñoz · Grado en Ingeniería Informática · ETSIIT, Universidad de Granada · 2026.
