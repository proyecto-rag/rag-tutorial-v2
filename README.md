# RAG (Retrieval-Augmented Generation) System

Este proyecto implementa un sistema RAG que permite consultar documentos PDF utilizando diferentes modelos de embedding y lenguaje.

## Características

- Procesamiento de documentos PDF
- Múltiples opciones de modelos de embedding:
  - all-MiniLM-L12-v2 (ligero y eficiente)
  - BAAI/bge-m3 (multilingüe y potente)
  - hkunlp/instructor-large (con capacidad de instrucciones)
- Soporte para diferentes modelos de lenguaje:
  - Modelos FLAN-T5 locales (small, base, large, xl)
  - API externa para generación de respuestas
- Sistema de chunking inteligente con superposición
- Base de datos vectorial con Chroma

## Requisitos

```bash
pip install langchain-community langchain-huggingface chromadb sentence-transformers transformers torch requests
```

## Estructura del Proyecto

- `populate_database.py`: Procesa los documentos PDF y los almacena en la base de datos
- `query_data.py`: Maneja las consultas y genera respuestas
- `get_embedding_function.py`: Gestiona los diferentes modelos de embedding

## Uso

### 1. Preparar los Documentos

Coloca tus archivos PDF en el directorio `data/`.

### 2. Poblar la Base de Datos

```bash
# Crear nueva base de datos multilingual por default
python populate_database.py --reset

# Actualizar base de datos existente
python populate_database.py
```

### 3. Realizar Consultas

```bash
# Consulta básica
python query_data.py "Tu pregunta aquí"

# Opciones adicionales
python query_data.py "Tu pregunta" \
    --model xl \           # Modelo a utilizar (small, base, large, xl)
    --docs 3 \            # Número de documentos para contexto (1-5)
    --use-local \         # Usar modelo local en lugar de API
    --max-tokens 2048     # Límite de tokens para la respuesta (solo aplica API)
```

## Configuración de Modelos

### Modelos de Embedding Disponibles

1. **minil12** (all-MiniLM-L12-v2)
   - Ligero y eficiente
   - Buen balance entre rendimiento y recursos

2. **multilingual** (BAAI/bge-m3)
   - Soporte multilingüe
   - Mayor potencia y precisión
   - Configurado para CPU por defecto

3. **instruction** (hkunlp/instructor-large)
   - Permite instrucciones personalizadas
   - Mayor consumo de recursos
   - Optimizado para documentos en español

### Modelos de Lenguaje

- **Locales** (FLAN-T5):
  - small: 512 tokens máx.
  - base: 768 tokens máx.
  - large: 1024 tokens máx.
  - xl: 2048 tokens máx.

- **API Externa**:
  - Límite configurable de tokens
  - Requiere API key válida

## Notas Importantes

- El sistema ajusta automáticamente el contexto si se excede el límite de tokens
- Los documentos se dividen en chunks de 800 caracteres con 80 caracteres de superposición
- La base de datos se puede resetear o actualizar según necesidad
- Se recomienda usar el modelo de embedding según el caso de uso:
  - minil12: para casos generales y recursos limitados
  - multilingual: para documentos multilingües
  - instruction: para casos que requieren instrucciones específicas
