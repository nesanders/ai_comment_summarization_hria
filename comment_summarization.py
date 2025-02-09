"""Use a LangChain AI agent to answer questions about the extracted testimonies.

This will search the utils.DATA_DIR for json files previously extracted by `document_extraction.py`.
It is assumed that these files are organized into state-specific subdirectories, and the state is extracted
as metadata based on that path structure.

One output file, in markdown and rendered HTML format, will be generated for each issue specified in the
constant ISSUE_CONFIGURATIONS.
"""

from copy import deepcopy
import logging
from pathlib import PosixPath
import subprocess
from typing import Optional

from langchain.cache import SQLiteCache
from langchain.llms.base import LLM
from langchain.globals import set_llm_cache, set_verbose
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain.prompts import Prompt
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from prompt_library import (
    EVIDENCE_PROMPT, 
    FROM_X_PROMPT, 
    MOST_X_PROMPT, 
    MISINFORMATION_PROMPT,
    STORYTELLING_PROMPT,
    VALUES_PROMPT,
    CitedAnswer
)
from utils import DATA_DIR, OUTPUT_DIR, START_TIME, get_git_hash, get_openai_secret, log_time, setup_logging


LOGGER = setup_logging('comment_summarization')

# Vector store for embedded comments
COMMENT_VECTOR_STORE = DATA_DIR / 'vector_store.faiss_index'

# Cache for LLM completions; saving these to disk prevents repeated re-calling
LLM_CACHE = DATA_DIR / 'llm_cache.db'

# See here for options: https://platform.openai.com/docs/guides/embeddings/what-are-embeddings
EMBEDDING_MODEL_CHOICE = 'text-embedding-3-small'

# See here for guidance: https://platform.openai.com/docs/guides/text-generation/which-model-should-i-use
LLM_MODEL_CHOICE = 'gpt-4-turbo-2024-04-09'
LLM_TOKEN_LIMIT = 120000

# How many documents to load per input json file. Set lower to do smaller test runs.
# Set to None to run on all documents
MAX_DOCS = None

# Temperature to use for LLM
OPENAI_TEMPERATURE = 0

# By default, how many comments to return from the RAG query
DEFAULT_RAG_K = 50

# Bill prompt configurations
ISSUE_CONFIGURATIONS = {
    #"trans_issues": {
        #"issue_description": "transgender people and health",
        #"pro_description": "pro-transgender-inclusion",
        #"con_description": "anti-trans",
        #"states": ["UT", "MT"],
        #"speaker_types": [
            #'people who work in government agencies', 
            #'parents', 
            #'youths', 
            #'Medical and public health practitioners'
        #]
    #},
    "dei_issues": {
        "issue_description": "teaching divisive concepts related to race, diversity, equity, and inclusion (DEI)",
        "pro_description": "pro-teaching DEI",
        "con_description": "anti-teaching DEI",
        "states": ["TN", "ND"],
        "speaker_types": [
            'educators', 
            'parents', 
            'youths', 
        ]
    }
}

def get_output_path(issue_name: str) -> PosixPath:
    """Return an output path for the markdown output file for a given issue_name.
    """
    return OUTPUT_DIR / f'comment_summarization_{issue_name}_{START_TIME}.md'

def get_embeddings_model() -> OpenAIEmbeddings:
    """Return the default Embeddings class.
    """
    return OpenAIEmbeddings(
        openai_api_key=get_openai_secret(), model=EMBEDDING_MODEL_CHOICE)


def load_embeddings(
    data: list[Document], 
    vector_store: PosixPath=COMMENT_VECTOR_STORE
) -> FAISS:
    """Return an embedding database for a document set, loading from disk if
    possible.
    """
    LOGGER.info('Starting load_embeddings')

    embeddings_model = get_embeddings_model()
    if vector_store.exists():
        LOGGER.info(
            f'Vector store already exists; loading from disk at {vector_store}')
        db = FAISS.load_local(COMMENT_VECTOR_STORE, embeddings_model, allow_dangerous_deserialization=True)
    else:
        LOGGER.info('Vector store does not exist; embedding via API')
        data_formatted = format_docs(data)
        #embeddings = embeddings_model.embed_documents([doc.page_content for doc in data_formatted])
        db = FAISS.from_documents(data_formatted, embeddings_model)
        LOGGER.info(f'Saving vector store to disk at {vector_store}')
        db.save_local(vector_store)
    
    return db

def format_docs(docs: list[Document]) -> list[Document]:
    """Add metadata about the document directly to the document text for visibility to the LLM.:"""
    out_docs = deepcopy(docs)
    for doc in out_docs:
        doc.page_content = f"""Document ID: {doc.metadata['doc_id']}
        State: {doc.metadata['state']}
        Hearing Transcript Title: {doc.metadata['transcript_name']}
        Transcript: ```{doc.metadata['text']}```
        Source URL: ```{doc.metadata['source_url']}```
        """
    return out_docs

def set_my_llm_cache(cache_file: PosixPath=LLM_CACHE) -> SQLiteCache:
    """Set an LLM cache, which allows for previously executed completions to be
    loaded from disk instead of repeatedly queried.
    """
    LOGGER.info(f'Setting LLM cache to {cache_file}')
    set_llm_cache(SQLiteCache(database_path=cache_file))

DOC_ID = 0
def comment_metadata_func(sample: dict, additional_fields: dict) -> dict:
    """Extract comment metadata from the json record (`sample`).
    
    See https://python.langchain.com/docs/modules/data_connection/document_loaders/json#the-metadata_func
    """
    global DOC_ID
    metadata = additional_fields
    metadata['speaker_id'] = sample.get('speaker_id')
    metadata['timestamp'] = sample.get('timestamp')
    metadata['source_url'] = sample.get('source_url')
    metadata['doc_id'] = DOC_ID
    DOC_ID += 1
    # Parse the state from the filepath
    if 'source' in additional_fields:
        metadata['state'] = PosixPath(additional_fields['source']).parent.name
        metadata['transcript_name'] = PosixPath(additional_fields['source']).stem
        metadata['text'] = f"STATE LEGISLATURE: {metadata['state'] }\n\nTESTIMONY: {sample['text']}"
    return metadata

def load_documents(
    data_dir: PosixPath=DATA_DIR, 
    max_docs: Optional[int]=MAX_DOCS
) -> list[Document]:
    """Load documents from a son `data_file` and return a document list.
    """
    LOGGER.info('Starting load_documents')
    all_doc_paths = data_dir.glob('**/*.json')
    data = []
    for doc_path in all_doc_paths:
        LOGGER.info(f'Loading from {doc_path=}')
        loader = JSONLoader(
            file_path=doc_path,
            jq_schema='.[]',
            content_key='text',
            metadata_func=comment_metadata_func)
        doc_data = loader.load()
        LOGGER.info(f'Loaded N={len(doc_data)} comments from {doc_path=}')
        if max_docs and max_docs < len(doc_data):
            LOGGER.info(f'Subsetting documents list to {max_docs=}')
            doc_data = doc_data[:max_docs]
        data += doc_data
    LOGGER.info(f'Loaded a total of N={len(data)} comments')
    
    llm = get_llm_model()
    total_tokens = llm.get_num_tokens(' '.join([d.dict()['page_content'] for d in data]))
    LOGGER.info(f'Constitutes a total of N={total_tokens} tokens')
    
    return data

LLM_MODEL = None
def get_llm_model(temperature: float=OPENAI_TEMPERATURE, 
    model: str=LLM_MODEL_CHOICE) -> LLM:
    """Return a configured LLM model.
    """
    if LLM_MODEL:
        return LLM_MODEL
    LOGGER.info('Starting get_llm_model')
    return ChatOpenAI(
        temperature=temperature, 
        openai_api_key=get_openai_secret(), 
        model_name=model
    )

def get_citing_llm_model() -> LLM:
    """Give an LLM model the ability to cite sources
    
    Following https://python.langchain.com/docs/use_cases/question_answering/citations
    """
    llm = get_llm_model()
    return llm.bind_tools(
        [CitedAnswer],
        tool_choice="CitedAnswer",
    )

def get_metadata_by_id(data, doc_id) -> dict:
    for doc in data:
        if doc.metadata['doc_id'] == doc_id:
            return doc.metadata

def rag_qa(
    data: list[Document], 
    prompt: Prompt,
    template_args: dict,
    states: list[str],
    k: int=DEFAULT_RAG_K,
) -> tuple[str, list[dict]]:
    """Answer the prompts using RAG.
    
    Parameters
    ----------
    data: list[Document]
        The set of documents to do RAG with
    prompt: PromptTemplate
        Prompt to run. Its template text will be used as a the RAG retrival query,
        and it will also be used to invoke the chain.
    template_args: dict
        Any parameters other than 'text' to be passed to the prompt.
    k: int=DEFAULT_RAG_K
        How many documents to return from the RAG query.
        
    Returns
    -------
    str
        Response to the query
    citations
        Citations referenced in response (the metadata dictionary from each doc)
    """
    citing_llm = get_citing_llm_model()
    vector_db = load_embeddings(data)
    
    def filter_on_state(metadata: dict):
        return metadata["state"] in states
    
    # TODO how can we max out the number of documents to put in the context window?
    # Here we fetch k*8 documents, filter down to ones from the right state, and return k that match
    # See https://python.langchain.com/docs/integrations/vectorstores/faiss/#similarity-search-with-filtering
    retriever = vector_db.as_retriever(search_kwargs={"k": k, "fetch_k": k * 8, "filter": filter_on_state})
    prompt = prompt.partial(**template_args)
    
    output_parser = JsonOutputKeyToolsParser(key_name="CitedAnswer", return_single=True)
    retrieval_chain = (
        {
            "text": retriever, 
        } 
        | prompt 
        | citing_llm 
        | output_parser
    )
    # We use the prompt template itself as the query to the vector store.
    response = retrieval_chain.invoke(prompt.pretty_repr())
    cited_docs = [get_metadata_by_id(data, doc_id) for doc_id in response[0]['citations']]
    
    ## Validate the cited sources by Doc ID
    retrieved_docs = retriever.invoke(prompt.pretty_repr())
    retrieved_states = sorted([str(doc).split('State: ')[1].split('\\n')[0] for doc in retrieved_docs])
    logging.info(f'Retrieved a total of {len(retrieved_states)} documents from states: {retrieved_states}')
    queried_doc_ids = sorted([int(str(doc).split('Document ID: ')[1].split('\\n')[0]) for doc in retrieved_docs])
    assert all([doc['doc_id'] in queried_doc_ids for doc in cited_docs]), \
        f"Cited document was not in retrieved context. Cited {[doc['doc_id'] for doc in cited_docs]}; Retrieved {queried_doc_ids}"
    
    return response[0]['answer'], cited_docs

def log_rag_qa(title: str, response: tuple[str, list[dict]]):
    """Log a `rag_qa` output to the standard LOGGER.
    """
    LOGGER.info("")
    LOGGER.info("")
    LOGGER.info(title)
    LOGGER.info("---------")
    LOGGER.info(response[0])
    LOGGER.info("")
    LOGGER.info("CITATIONS")
    LOGGER.info("---------")
    LOGGER.info(response[1])
    LOGGER.info("")
    LOGGER.info("")

def append_output(output_file: PosixPath, title: str, content: str):
    """Append some results `content` to the output_file, under a heading `title`.
    """
    output_file.parent.mkdir(exist_ok=True, parents=True)
    if not output_file.exists():
        with open(output_file, 'w') as f:
            f.write(
f"""# comment_summarization.py

Execution time: `{START_TIME}`

git hash: `{get_git_hash()}`
            
"""
)
    with open(output_file, 'a') as f:
        f.write(f'\n\n## {title}\n{content}\n')

def compile_output(output_file: PosixPath):
    """Use pandoc to compile the output file to HTML.
    """
    subprocess.run(
        f"pandoc {output_file} -V colorlinks=true -o {output_file.with_suffix('.html')}", 
        shell=True, check=True)

def do_all_rag_qa(
    issue_name: str,
    data: list[Document], 
) -> tuple[str, list[dict]]:
    """Execute each RAG prompt for issue `issue_name` and output the markdown and HTML files.
    """
    logging.info(f"Starting do_all_rag_qa for issue {do_all_rag_qa}")
    
    output_file = get_output_path(issue_name)
    
    rag_most_personal = rag_qa(
        data, 
        MOST_X_PROMPT, 
        {"issue_description": ISSUE_CONFIGURATIONS[issue_name]["issue_description"], "N": 10, "adjective": "personal, heartfelt, based-on-lived experience"},
        ISSUE_CONFIGURATIONS[issue_name]["states"]
    )
    log_rag_qa("# MOST PERSONAL", rag_most_personal)
    append_output(output_file, "Most personal testimonials", rag_most_personal[0])
    
    rag_values = rag_qa(
        data, 
        VALUES_PROMPT,
        {"issue_description": ISSUE_CONFIGURATIONS[issue_name]["issue_description"], "N": 10},
        ISSUE_CONFIGURATIONS[issue_name]["states"]
    )
    log_rag_qa("# VALUES", rag_values)
    append_output(output_file, 'Expressions of Values', rag_values[0])
    
    
    for speaker_type in ISSUE_CONFIGURATIONS[issue_name]["speaker_types"]:
        rag_speaker = rag_qa(
            data, 
            FROM_X_PROMPT,
            {"issue_description": ISSUE_CONFIGURATIONS[issue_name]["issue_description"], "N": 10, "speaker_type": speaker_type},
            ISSUE_CONFIGURATIONS[issue_name]["states"]
        )
        log_rag_qa(f"# FROM {speaker_type}", rag_speaker)
        append_output(output_file, f'Testimonials from {speaker_type}', rag_speaker[0])
    
    rag_evidence = rag_qa(
        data, 
        EVIDENCE_PROMPT,
        {"issue_description": ISSUE_CONFIGURATIONS[issue_name]["issue_description"], "N": 10},
        ISSUE_CONFIGURATIONS[issue_name]["states"]
    )
    log_rag_qa(f"# EVIDENCE", rag_evidence)
    append_output(output_file, 'Lines of evidence cited', rag_evidence[0])
    
    rag_stories = rag_qa(
        data,
        STORYTELLING_PROMPT,
        {
            "issue_description": ISSUE_CONFIGURATIONS[issue_name]["issue_description"], 
            "N": 10, 
            "pro_description": ISSUE_CONFIGURATIONS[issue_name]["pro_description"],
            "con_description": ISSUE_CONFIGURATIONS[issue_name]["con_description"]
        },
        ISSUE_CONFIGURATIONS[issue_name]["states"]
    )
    log_rag_qa(f"# STORYTELLING", rag_stories)
    append_output(output_file, 'Examples of storytelling', rag_stories[0])
    
    rag_misinfo = rag_qa(
        data,
        MISINFORMATION_PROMPT,
        {"issue_description": ISSUE_CONFIGURATIONS[issue_name]["issue_description"], "N": 10},
        ISSUE_CONFIGURATIONS[issue_name]["states"]
    )
    log_rag_qa(f"# MISINFORMATION", rag_misinfo)
    append_output(output_file, 'Examples of misinformation', rag_misinfo[0])

def main():
    """Main logic for comment_summarization.py.
    """
    LOGGER.info('Begin comment_summarization.py')
    set_my_llm_cache()
    set_verbose(True)
    data = load_documents()
    
    for issue_name in ISSUE_CONFIGURATIONS:
        do_all_rag_qa(issue_name, data)
        compile_output(get_output_path(issue_name))
    
    log_time(LOGGER, "comment_summarization complete")
    
if __name__ == '__main__':
    main()
