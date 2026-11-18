import contextlib
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener

import nbformat
import psutil


path = Path('examen1bim.ipynb')
original = nbformat.read(path, as_version=4)
kernels = []
kernels_examen = []
for proc in psutil.process_iter(['name', 'cmdline']):
    if 'python' in (proc.info['name'] or '').lower() or 'jupyter' in (proc.info['name'] or '').lower():
        args = proc.info['cmdline'] or []
        roles = [arg for arg in args if any(label in arg.lower() for label in ('jupyter', 'ipykernel', 'uvicorn'))]
        if roles:
            if any('jupyter' in role.lower() or 'ipykernel' in role.lower() for role in roles):
                kernels.append(proc.pid)
            try:
                print(f'Proceso de notebook o servidor existente: {proc.pid} {roles}, carpeta {proc.cwd()}')
                if 'ipykernel_launcher' in args and Path(proc.cwd()) == Path.cwd():
                    kernels_examen.append((proc, args))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                print(f'Proceso de notebook o servidor existente: {proc.pid} {roles}')

if '--apagar-kernel' in sys.argv and kernels_examen:
    from jupyter_client import BlockingKernelClient

    archivos = set()
    for _, args in kernels_examen:
        if '-f' in args:
            archivos.add(args[args.index('-f') + 1])
        else:
            archivos.update(arg.split('=', 1)[1] for arg in args if arg.startswith('--f='))
    assert len(archivos) == 1, 'No se identificó un único kernel del examen'
    client = BlockingKernelClient(connection_file=archivos.pop())
    client.load_connection_file()
    client.start_channels()
    try:
        client.shutdown(restart=False)
    finally:
        client.stop_channels()
    _, activos = psutil.wait_procs([proc for proc, _ in kernels_examen], timeout=12)
    assert not activos, 'El kernel del examen no completó su apagado'
    kernels = [pid for pid in kernels if pid not in {proc.pid for proc, _ in kernels_examen}]

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
if '--arranque-local' in sys.argv:
    print(f'Procesos de Jupyter presentes: {kernels}')
    cliente = build_opener(ProxyHandler({}))
    base = 'http://127.0.0.1:8000'
    with tempfile.TemporaryDirectory() as cache_vacia, tempfile.TemporaryFile() as registro:
        entorno = dict(os.environ, HF_HOME=cache_vacia, NLTK_DATA=cache_vacia, PYTHONDONTWRITEBYTECODE='1')
        servidor = subprocess.Popen(
            [sys.executable, '-m', 'uvicorn', 'main:app'],
            cwd=Path('despliegue'), env=entorno, stdout=registro, stderr=subprocess.STDOUT,
        )
        try:
            limite = time.monotonic() + 20
            while servidor.poll() is None and time.monotonic() < limite:
                try:
                    with cliente.open(base + '/health', timeout=1) as response:
                        assert response.status == 200 and json.load(response) == {'status': 'ok', 'index_loaded': True, 'documents': 5183}
                    break
                except URLError:
                    time.sleep(0.2)
            else:
                raise RuntimeError('Uvicorn no arrancó en el puerto 8000')
            print('GET /health 200 en el puerto 8000')
            for modelo, primer_id in [('tfidf', '10906636'), ('bm25', '21257564')]:
                datos = {'query': '0-dimensional biomaterials show inductive properties.', 'model': modelo, 'top_k': 3}
                peticion = Request(base + '/search', data=json.dumps(datos).encode(), headers={'Content-Type': 'application/json'})
                with cliente.open(peticion, timeout=5) as response:
                    assert response.status == 200
                    resultado = json.load(response)
                    assert resultado['model'] == modelo and len(resultado['results']) == 3
                    assert resultado['results'][0]['document_id'] == primer_id
                print(f'POST /search {modelo}: 200, primer ID {primer_id}')
        finally:
            if servidor.poll() is None:
                servidor.terminate()
                servidor.wait(timeout=10)
            registro.seek(0)
            print(registro.read().decode(errors='replace'))
    if '--registrar' in sys.argv:
        assert not kernels, 'Todavía hay un proceso de Jupyter abierto'
        nota = (
            'Verificación desde una terminal con Jupyter cerrado. '
            '`python -m uvicorn main:app` inició el servicio en el puerto 8000 con una caché de SciFact vacía y en modo sin conexión. '
            '`GET /health` y `POST /search` para ambos modelos devolvieron HTTP 200.'
        )
        celda = next(cell for cell in original.cells if cell.id == 'pruebas-http-md')
        celda.source += '\n\n' + nota
        nbformat.validate(original)
        nbformat.write(original, path)
    sys.exit(0)

if '--repetir' not in sys.argv:
    runpy.run_path('.completar_examen.py')
notebook = nbformat.read(path, as_version=4)
nbformat.validate(notebook)

namespace = {'__name__': '__main__'}
execution_count = 0
for cell in notebook.cells:
    if cell.cell_type != 'code':
        continue
    execution_count += 1
    print(f'Ejecutando {cell.id}', flush=True)
    output = io.StringIO()
    try:
        with contextlib.redirect_stdout(output):
            exec(compile(cell.source, cell.id, 'exec'), namespace)
    finally:
        print(output.getvalue(), end='')
    cell.execution_count = execution_count
    cell.outputs = [nbformat.v4.new_output('stream', name='stdout', text=output.getvalue())] if output.getvalue() else []

baseline = {cell.id: cell for cell in original.cells}
for cell_id in ('tfidf-code', 'bm25-code', 'eval-code'):
    cell = next(cell for cell in notebook.cells if cell.id == cell_id)
    assert cell.outputs[0].text == baseline[cell_id].outputs[0].text, (cell_id, cell.outputs[0].text)

nbformat.validate(notebook)
nbformat.write(notebook, path)
print(f'Notebook válido, {execution_count} celdas de código ejecutadas. Rankings y métricas conservados.')
