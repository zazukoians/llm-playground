import asyncio
import logging
import os

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

class CubeBody(BaseModel):
    question: str

class GenerateBody(CubeBody):
    cube: str

class FullBody(CubeBody):
    pass

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

async def _select_cube(question: str) -> str:
    cube_selection_settings = {
        "temperature": 0.5,
        "top_p": 0.5
    }
    cubes = fetch_cubes_descriptions()
    cube_selection_chain = create_cube_selection_chain(api_key=OPENAI_API_KEY, handler=handler, **cube_selection_settings)

    # question = f"sum of emission of CO2 for industry between year 2009 and 2011"
    # question = f"get average of emission of Methane for transport between years 2007 and 2005"
    #question = "What percentage of emission was from N2O and CH4 compared to total emission?"

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
        raise HTTPException(status_code=404, detail=f"Service was unable to select proper cube. Full response: {cube_selection_response}")

    return selected_cubes[0]


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
    logger.info(f"Form query request: question={question}")
    body = FullBody(question=question)
    cube = await _select_cube(body.question)
    query = await _generate_query(body.question, cube)
    return templates.TemplateResponse("index.html", {"request": request, "cube": cube.strip('<>'), "question": question, "query": query })
