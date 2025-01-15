import re
from typing import Any, Dict, Optional, Union

import SPARQLWrapper
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.schema import AgentAction, AgentFinish


class LoggingHandler(BaseCallbackHandler):
    """Callback Handler that writes logger"""

    def __init__(
        self, logger
    ) -> None:
        """Initialize callback handler."""
        self.logger = logger

    def __del__(self) -> None:
        """Destructor to cleanup when done."""
        pass

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Print out that we are entering a chain."""
        class_name = serialized.get("name", serialized.get("id", ["<unknown>"])[-1])
        self.logger.info(f"Entering new {class_name} chain...")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Print out that we finished a chain."""
        self.logger.info(f"Finished chain.")

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run on agent action."""
        self.logger.info(action.log)

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """If not the final action, print out observation."""
        if observation_prefix is not None:
            self.logger.info(observation_prefix)
        self.logger.info(output)
        if llm_prefix is not None:
            self.logger.info(llm_prefix)

    def on_text(
        self, text: str, color: Optional[str] = None, end: str = "", **kwargs: Any
    ) -> None:
        """Run when agent ends."""
        self.logger.info(text)

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run on agent end."""
        self.logger.info(finish.log)


def run_query(query: str, return_format: str = SPARQLWrapper.JSON):
    sparql = SPARQLWrapper.SPARQLWrapper(endpoint="https://ld.stadt-zuerich.ch/query")
    sparql.setReturnFormat(return_format)
    sparql.setHTTPAuth(SPARQLWrapper.DIGEST)
    sparql.setMethod(SPARQLWrapper.POST)
    sparql.setQuery(query)
    return sparql.queryAndConvert()


def fetch_cubes_descriptions() -> str:
    cubes_query = """
        PREFIX cube: <https://cube.link/>
        PREFIX sh: <http://www.w3.org/ns/shacl#>
        PREFIX schema: <http://schema.org/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX qudt: <http://qudt.org/schema/qudt/>
        PREFIX dct: <http://purl.org/dc/terms/>

        CONSTRUCT {
        ?cube a cube:Cube;
                schema:name ?label;
                schema:description ?description.
        }
        WHERE {
            ?cube a cube:Cube ;
                    schema:name ?label ;
                    schema:description ?description ;
                    dct:creator <https://register.ld.admin.ch/opendataswiss/org/bundesamt-fur-umwelt-bafu> .


            FILTER(lang(?label) = 'en')
            FILTER(lang(?description) = 'en')

            MINUS {
                ?cube schema:expires ?date .
            }
        }
    """

    raw_result = run_query(cubes_query, return_format=SPARQLWrapper.N3)
    return raw_result.decode()


def create_cube_selection_chain(api_key: str, handler: BaseCallbackHandler, temperature: float = 0.5, top_p: float = 0.5) -> LLMChain:
    cube_selection_model = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=temperature, top_p=top_p)

    cubes_description = """
    Given following data cubes with its labels and description:
    {cubes}
    """

    human_template = """
    For this question: {question}
    Return ONLY the cube ID that best matches this question.
    If no cube matches even with these mandatory rules, return 'Unable to select proper cube'
    and list all topics that ARE available in the cubes. Format the available topics as a list ith bullet points.
      """
    cube_selection_prompt = ChatPromptTemplate.from_messages([
        ("system", cubes_description),
        ("human", human_template),
    ])

    cube_selection_chain = LLMChain(prompt=cube_selection_prompt, llm=cube_selection_model, callbacks=[handler])

    return cube_selection_chain


def create_query_generation_chain(api_key: str, handler: BaseCallbackHandler, temperature: float = 0.2, top_p: float = 0.1) -> LLMChain:
    model = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=temperature, top_p=top_p)

    sample_description = """
    Given cube and its sample observation::
    {cube_and_sample}
    """

    structure_description = """
    Dimensions labels:
    {dimensions_triplets}
    """

    system_instructions = """
    You are a SPARQL query generator. Generate only the SPARQL query without any additional text or explanations.
    Important rules for query generation:
    1. Do not add any explanatory text before or after the query
    2. Do not wrap the output in code blocks or sparql tags
    3. The query should start directly with the PREFIX declarations
    4. For year/time filtering, use these exact patterns based on the type of time constraint:

    For a specific year range:
    FILTER(?year >= "2005"^^xsd:gYear && ?year <= "2007"^^xsd:gYear)

    For years after a specific year:
    FILTER(?year >= "2003"^^xsd:gYear)

    For years before a specific year:
    FILTER(?year <= "2005"^^xsd:gYear)

    For a specific year:
    FILTER(?year = "2004"^^xsd:gYear)
    """

    query_template = """
    PREFIX cube: <https://cube.link/>
    PREFIX schema: <http://schema.org/>
    PREFIX qudt: <http://qudt.org/schema/qudt/>
    PREFIX sh: <http://www.w3.org/ns/shacl#>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

    SELECT *
    WHERE {{
    {cube} a cube:Cube;
        cube:observationSet ?observationSet.

    ?observationSet a cube:ObservationSet;
        cube:observation ?observation.

    ?observation a cube:Observation.
    }}
    """
    human_template = f"Modify this query: {query_template}\n to get {{question}} for this cube {{cube}}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", sample_description),
        ("system", structure_description),
        ("human", human_template),
        ("system", system_instructions),
    ])

    chain = LLMChain(prompt=prompt, llm=model, callbacks=[handler])

    return chain



def parse_all_cubes(ai_response: str) -> list[str]:
    words = ai_response.split()
    cube_pattern = r'<.+>'
    groups_of_cubes = map(lambda s: re.findall(cube_pattern, s), words)

    return [cube for group in groups_of_cubes for cube in group]


def fetch_cube_sample(cube: str) -> str:
    query = f"""
        PREFIX cube: <https://cube.link/>
        PREFIX sh: <http://www.w3.org/ns/shacl#>
        PREFIX schema: <http://schema.org/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX qudt: <http://qudt.org/schema/qudt/>

        CONSTRUCT {{
        ?cube a cube:Cube;
            cube:observationSet ?observationSet.
        ?observationSet a cube:observationSet;
            cube:observation ?s.
        ?s ?p ?o.
        }}
        WHERE {{
        {{
            SELECT ?cube ?observationSet (SAMPLE(?observation) AS ?s)
            WHERE {{
            VALUES ?cube {{ {cube} }}
            ?cube a cube:Cube ;
                    cube:observationSet ?observationSet.
            ?observationSet cube:observation ?observation.
            }}
            GROUP BY ?cube ?observationSet
        }}
        ?s ?p ?o .
        }}
    """

    raw_result = run_query(query, return_format=SPARQLWrapper.N3)
    return raw_result.decode()


def fetch_dimensions_triplets(cube: str) -> str:
    query = f"""
        PREFIX cube: <https://cube.link/>
        PREFIX sh: <http://www.w3.org/ns/shacl#>
        PREFIX schema: <http://schema.org/>
        PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX qudt: <http://qudt.org/schema/qudt/>

        CONSTRUCT {{
        ?values schema:name ?label.
        }}
        WHERE {{
            SELECT ?values ?label
            WHERE {{
                VALUES ?cube {{ {cube} }}

                ?cube a cube:Cube ;
                    cube:observationConstraint ?shape .

                ?shape a cube:Constraint;
                    sh:property ?property .

                ?property sh:path ?dimensions ;
                sh:in ?list .

                ?list rdf:rest*/rdf:first ?values .

                ?values schema:name ?label .

                FILTER( lang(?label) = 'en')
            }}
        }}
    """

    raw_result = run_query(query, return_format=SPARQLWrapper.N3)
    return raw_result.decode()
