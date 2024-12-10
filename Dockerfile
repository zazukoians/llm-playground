FROM python:3.10 AS build-stage

WORKDIR /build
ENV VIRTUAL_ENV=/build/venv
RUN python -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
RUN pip install poetry
COPY ./pyproject.toml ./poetry.lock* /build/
RUN poetry install --without dev --no-root

# COPY ./pyproject.toml ./poetry.lock* /build/
# RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

FROM python:3.10

WORKDIR /code
COPY --from=build-stage /build/venv /code/venv
# RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt
COPY ./app /code/app
ENV VIRTUAL_ENV=/code/venv
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
EXPOSE 80

ENTRYPOINT ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
