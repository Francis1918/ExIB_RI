# Buscador SciFact

Sistema de recuperación de información que indexa **5.183 documentos científicos** del corpus SciFact de BEIR. Busca por términos en los títulos y resúmenes de los documentos y ordena los resultados con **TF-IDF y similitud coseno** o **BM25**. El notebook prepara el índice y compara ambos modelos. La API de FastAPI carga el índice al iniciar.

**API en Vercel** · [https://scifact-api.vercel.app/](https://scifact-api.vercel.app/)

[Documentación interactiva (Swagger UI)](https://scifact-api.vercel.app/docs) · [Estado del servicio](https://scifact-api.vercel.app/health)

## Uso de la API

| Método y ruta | Descripción |
| --- | --- |
| `GET /health` | Indica si el índice está cargado y cuántos documentos contiene. |
| `POST /search` | Devuelve los documentos ordenados por el puntaje del modelo elegido. |

Para buscar desde Swagger UI o Postman, envía una solicitud `POST` a `https://scifact-api.vercel.app/search` con `Content-Type: application/json` y este cuerpo.

```json
{
  "query": "0-dimensional biomaterials show inductive properties.",
  "model": "bm25",
  "top_k": 3
}
```

El campo `model` acepta `bm25` o `tfidf`. El campo `top_k` es un entero mayor o igual que cero y limita la cantidad de resultados. Cada resultado incluye su posición (`rank`), el identificador del documento (`document_id`), el puntaje (`score`), el título (`title`) y un fragmento del resumen de hasta 240 caracteres (`snippet`). Si no hay coincidencias con puntaje positivo o `top_k` es `0`, `results` es una lista vacía.

## Ejecución local

Con **Python 3.13**, ejecuta estos comandos desde la raíz del repositorio.

```bash
cd despliegue
python -m pip install -r requirements.txt
python -m uvicorn main:app
```

La API usa el índice comprimido y el tokenizador de NLTK incluidos en el repositorio. Puedes probar las rutas en la [documentación interactiva local](http://127.0.0.1:8000/docs).

## Evaluación de los modelos

El notebook calcula el promedio de las métricas en las **300 consultas de prueba** con juicios de relevancia de SciFact.

| Modelo | Precision@10 | Recall@10 | NDCG@10 |
| --- | ---: | ---: | ---: |
| TF-IDF + coseno | 0.0750 | 0.6892 | 0.5338 |
| BM25 | 0.0843 | 0.7670 | 0.6325 |

BM25 obtuvo mejores resultados en las tres métricas. La mayor diferencia fue en NDCG@10, que da más peso a los documentos relevantes que aparecen en las primeras posiciones.

## Archivos del proyecto

- Notebook de preparación y evaluación. La sección 9 exporta el índice. [`examen1bim.ipynb`](examen1bim.ipynb)
- Rutas de FastAPI. [`despliegue/main.py`](despliegue/main.py)
- Carga del índice y cálculo de los rankings de TF-IDF y BM25. [`despliegue/buscador.py`](despliegue/buscador.py)
- Índice comprimido y recursos de NLTK. [`despliegue/data/`](despliegue/data/)
