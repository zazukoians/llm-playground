# Important

This code base uses OpenAI API Key. To make te code functional you need to set environment variable OPENAI_API_KEY to proper key for OpenAI service.

# Installation guide (tested on Python 3.10)

1. Install python
2. Install poetry globally

```
pip install poetry
```

3. Setup virtual environment in local directory:

```
python -m venv .venv
```

4. Activate virtual environment

```
source .venv/bin/activate
```

5. Install dependencies using poetry in virtual environment

```
poetry install --no-root
```

6. Recommended way to run: Visual Studio Code with Python etensions installed. Allows to run playground/full_pipelne.ipynb notebook.

# Installation guid for API version

1. Build image:

```
docker build -t zazuko/sparql-ai-api .
```

2. Run image (note that OPENAI_API_KEY should be placed in .env file to make this work):

```
docker run -p 8080:80 --env-file .env zazuko/sparql-ai-api
```

# Next steps

## Productize the model.
The goal is to make our prototype accessible to everyone. Steps:
- Stabilize by selecting detailed GPT model versions
- Logging - keep track of all the questions users asked
- Build UI
- Error handling - return error when cube with relevant information cannot be found, or when subject is out of scope
- Introduce caching - question that was already asked, will be answered based on cached results of GPT API
- Add third step in pipeline - generate answer based on data returned by the generated SPARQL query


## Stabilize model performance:
The goal is to improve the accuracy of LLB-based SPARQL queries. Steps:
- Build test datasets of various questions and prepare testing environment to assess overall performance
- Prepare an environment for statistical testing of model performance with respect to parameter or query changes (e.g. how changing the temperature affects model performance, how changing prompt formulation affects the performance etc.)
- Check model behavior for corner cases, where there is no way to give proper answer to the question (i.e. missing cube or dimension)
- Add a "feedback loop" - if result was not correct, try to fix it using LLM, or other techniques
- Improve current usage of RAG using retrievers in langchain (filter cube list based on semantic similarity etc.)

## Experimental:
The goal is to test whether alternative approaches can improve model performance. Ideas:
- Test alternative implementation approach using Assistants API
- Test soft prompt generation
- Add preprocessing to user question to remove unnecessary words and leave only words that have any valuable meaning
- Extension for langchain library for knowledge graphs: make KGs work with Retrievers mechanism in langchain
