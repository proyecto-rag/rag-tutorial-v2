# Sistema RAG con Hugging Face

Este proyecto implementa un sistema de Generación Aumentada por Recuperación (RAG) utilizando modelos de Hugging Face para el procesamiento de documentos PDF.

## Requisitos

- Python 3.9+
- Entorno virtual (recomendado)

## Configuración

1. Clona este repositorio:
```bash
git clone <url-del-repositorio>
cd rag-tutorial-v2
```

2. Crea y activa un entorno virtual:
```bash
# En macOS/Linux
python3 -m venv venv
source venv/bin/activate

# En Windows
python -m venv venv
venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### 1. Poblar la base de datos

Primero, se debe cargar los documentos PDF en la base de datos vectorial:

```bash
python populate_database.py
```

Para reiniciar la base de datos:

```bash
python populate_database.py --reset
```

### 2. Consultar documentos

Para hacer consultas al sistema RAG:

```bash
python query_data.py "tu pregunta aquí"
```

#### Opciones adicionales:

- **Seleccionar tamaño del modelo**:
  ```bash
  python query_data.py "tu pregunta aquí" --model base
  ```
  Opciones disponibles: `small` (predeterminado), `base`, `large`

- **Definir número de documentos de contexto**:
  ```bash
  python query_data.py "tu pregunta aquí" --docs 3
  ```
  Valores aceptados: 1-5 (predeterminado: 2)

- **Combinación de opciones**:
  ```bash
  python query_data.py "tu pregunta aquí" --model large --docs 4
  ```

## Estructura del proyecto

- `populate_database.py`: Script para cargar y procesar documentos PDF
- `query_data.py`: Script para consultar el sistema RAG
- `get_embedding_function.py`: Funciones para generar embeddings
- `chroma/`: Directorio donde se almacena la base de datos vectorial
- `data/`: Directorio donde se deben colocar los archivos PDF

## Notas técnicas

- El sistema utiliza embeddings de HuggingFace ("all-MiniLM-L6-v2")
- Para la generación de respuestas se usa modelos T5 de Google (flan-t5-small/base/large)
- Los modelos se ejecutan localmente, no requieren conexión a internet después de la descarga inicial

## EJEMPLO COMÚN
- ```source venv/bin/activate``` para arrancar el ambiente
- ```python3 populate_database.py``` para volver a llenar la base de datos si se agrego un nuevo PDF
- ```python3 query_data.py "PREGUNTA" --model xl --docs 5``` para usar el segundo mejor modelo