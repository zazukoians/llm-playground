import logging
import os
from collections import OrderedDict
from hashlib import md5

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.lib import (LoggingHandler, create_cube_selection_chain,
                     create_query_generation_chain, fetch_cube_sample,
                     fetch_cubes_descriptions, fetch_dimensions_triplets,
                     parse_all_cubes)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
handler = LoggingHandler(logger)

# Simple LRU cache with a maximum size
class LRUCache:
    def __init__(self, max_size: int):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key: str):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: str):
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)

cache = LRUCache(max_size=20)

class CubeBody(BaseModel):
    question: str

class GenerateBody(CubeBody):
    cube: str

class FullBody(CubeBody):
    pass

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

def get_cache_key(question: str, cube: str = None) -> str:
    key = question if cube is None else f"{question}-{cube}"
    return md5(key.encode()).hexdigest()

async def _select_cube(question: str) -> str:
    cube_selection_settings = {
        "temperature": 0.2,
        "top_p": 0.1
    }
    cubes = fetch_cubes_descriptions()
    cube_selection_chain = create_cube_selection_chain(api_key=OPENAI_API_KEY, handler=handler, **cube_selection_settings)

    cube_selection_response = await cube_selection_chain.ainvoke({
        "cubes": cubes,
        "question": question,
    })
    cube_selection_response = cube_selection_response['text']

    logger.info("========== CUBES RESPONSE ================")
    logger.info(f"{cube_selection_response}")

    selected_cubes = parse_all_cubes(cube_selection_response)

    if not selected_cubes:
        logger.warning("Failed at parsing cube id from response. Returning 404 and response as a result")
        raise HTTPException(status_code=404, detail=cube_selection_response)

    return selected_cubes[0]

async def _select_cube_cached(question: str) -> str:
    key = get_cache_key(question)
    cached = cache.get(key)
    if cached:
        return cached
    cube = await _select_cube(question)
    cache.set(key, cube)
    return cube


async def _generate_query(question: str, cube: str) -> str:
    cube_and_sample = fetch_cube_sample(cube)
    dimensions_triplets = fetch_dimensions_triplets(cube)

    query_generation_settings = {
        "temperature": 0.2,
        "top_p": 0.1
    }

    generation_chain = create_query_generation_chain(api_key=OPENAI_API_KEY, handler=handler, **query_generation_settings)

    query_generation_response = await generation_chain.ainvoke({
        "cube_and_sample": cube_and_sample,
        "dimensions_triplets": dimensions_triplets,
        "cube": cube,
        "question": question,
    })
    query_generation_response = query_generation_response['text']

    logger.info("========== QUERY GENERATION RESPONSE ================")
    logger.info(f"{query_generation_response}")

    return query_generation_response

async def _generate_query_cached(question: str, cube: str) -> str:
    key = get_cache_key(question, cube)
    cached = cache.get(key)
    if cached:
        return cached
    query = await _generate_query(question, cube)
    cache.set(key, query)
    return query


@app.post("/cube")
async def select_cube(body: CubeBody):
    logger.info(f"Select cube request: {body}")
    return {
        "result": await _select_cube(body.question)
    }


@app.post("/query")
async def select_cube(body: GenerateBody):
    logger.info(f"Generate query request: {body}")
    return {
        "result": await _generate_query(body.question, body.cube)
    }


@app.post("/")
async def select_cube_and_generate_query(body: FullBody):
    logger.info(f"Full generate request: {body}")

    selected_cube = await _select_cube(body.question)

    query = await _generate_query(body.question, selected_cube)
    return {
        "result": query
    }

    # NOTE: currently only query is returned. Returning query result is possible.

@app.get("/")
def get_status():
    return {
        "status": "Service is up and running"
    }

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("app/static/favicon.ico")

@app.get("/ui", response_class=HTMLResponse)
async def get_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ui", response_class=HTMLResponse)
async def handle_form_query(request: Request, question: str = Form(...)):

    forwarded_for = request.headers.get("X-Forwarded-For", "not set")
    logger.info("========== BROWSER DATA ================")
    x_forwarded_for = request.headers.get("X-Forwarded-For", "not set")
    x_real_ip = request.headers.get("X-Real-IP", "not set")
    client_host = request.client.host if request.client else "no client"

    logger.info(
        f"Form query request: question={question}, "
        f"X-Forwarded-For={x_forwarded_for}, "
        f"X-Real-IP={x_real_ip}, "
        f"client_host={client_host}"
    )

    try:
        body = FullBody(question=question)
        cube = await _select_cube_cached(body.question)
        query = await _generate_query_cached(body.question, cube)
        return templates.TemplateResponse("index.html", {
            "request": request,
            "cube": cube.strip('<>'),
            "question": question,
            "query": query
        })
    except HTTPException as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "question": question,
            "error": e.detail
        })
