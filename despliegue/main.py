from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from buscador import cargar_indice


Modelo = Literal['tfidf', 'bm25']
ARCHIVO_INDICE = Path(__file__).resolve().parent / 'data' / 'indice_scifact.json.gz'


class Consulta(BaseModel):
    query: str
    model: Modelo
    top_k: int = Field(ge=0, strict=True)


class Resultado(BaseModel):
    rank: int
    document_id: str
    score: float
    title: str
    snippet: str


class RespuestaBusqueda(BaseModel):
    model: Modelo
    results: list[Resultado]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.buscador = cargar_indice(ARCHIVO_INDICE)
    yield


app = FastAPI(title='Buscador SciFact', lifespan=lifespan)


@app.get('/health')
def health(response: Response):
    buscador = getattr(app.state, 'buscador', None)
    if buscador is None:
        response.status_code = 503
        return {'status': 'unavailable', 'index_loaded': False, 'documents': 0}
    return {'status': 'ok', 'index_loaded': True, 'documents': len(buscador.doc_ids)}


@app.post('/search', response_model=RespuestaBusqueda)
def search(consulta: Consulta):
    buscador = app.state.buscador
    ranking = buscador.search_tfidf if consulta.model == 'tfidf' else buscador.search_bm25
    results = []
    for rank, (doc_id, score) in enumerate(ranking(consulta.query, consulta.top_k), start=1):
        texto = buscador.doc_texts[doc_id]
        snippet = texto[:239].rstrip() + '…' if len(texto) > 240 else texto
        results.append(Resultado(
            rank=rank,
            document_id=doc_id,
            score=score,
            title=buscador.doc_titles[doc_id],
            snippet=snippet,
        ))
    return RespuestaBusqueda(model=consulta.model, results=results)
