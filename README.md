# Legislative comment summarization using AI agents

This repository demonstrates using AI (LLM) agents built with LangChain
to automatically summarize and generate information about legislative policy
comments from the public.

This code was developed during a collaboration with Health Resources in Action
to look at legislative testimony delivered in oral hearings at multiple stage 
legislatures in October 2023. The AI models and methods used were selected
at that point in time and are no longer the best available.

NOTE: This repository does not include input data or outputs, but does include
all code and environment specifications necessary to reproduce this work.

## Setup

### Compute environment

A `conda` environment is used to manage the compute environment for this work.

To install the environment, use: 

```
conda env create --file langchain_comments.yml
```

If you receive errors to related to platform availability, you may need to use
the platform agonistic specification instead:
    
```
conda env create --file langchain_comments_crossplatform.yml
```

To activate the environment, use:
    
```
conda activate langchain_comments
```

Note: The script `export_conda_env.sh` is used to export the environment
specification (.yml) files.

### Credentials

You will need to create a file `SECRET_openai.txt` in the root directory
containing your OPENAI API key to be able to use `comment_summarization.py`.

## Contents

### Data extraction

The analysis code described below expects data input in a json format with
fields for the `speaker_id`, `timestamp`, comment `text`, and a `source_url`
corresponding to each comment. An example taken from a North Dakota oral
hearing looks like this:

```
[{"speaker_id": "1", "timestamp": "00:03", "text": "Live. We're live. Good morning. Welcome to Education Committee member. We have a lot of strange faces in our audience today, which is wonderful. We don't see them in here very often, but we'll start with role please. ", "source_url": "[https:...]"}, {"speaker_id": "2", "timestamp": "00:03", "text": "Chairman Heinert. Is Adam meeting right? Correct. Vice Chairman Schreiber. Back here. Representatives Dyke. Here. Hulk? Here. Heman. Here. Hoverson. Jonas? Here. Longmire. Here. Marshall. Here. Murphy? Here. Novak. Timmons. Kami. Here. Hager here. ", "source_url": "[https:...]"}
```

The provided script `document_extraction.py` can be used to extract 
transcriptions provided by the service `rev.com` in Word document 
format into a structured `json`-formatted file. The script operates
on Word document transcripts formatted as follows:

```
Speaker 1 (00:03:18):
Live. We're live. Good morning. Welcome to Education Committee member. We have a lot of strange faces in our audience today, which is wonderful. We don't see them in here very often, but we'll start with role please.

Speaker 2 (00:03:32):

Chairman Heinert. Is Adam meeting right? Correct. Vice Chairman Schreiber. Back here. Representatives Dyke. Here. Hulk? Here. Heman. Here. Hoverson. Jonas? Here. Longmire. Here. Marshall. Here. Murphy? Here. Novak. Timmons. Kami. Here. Hager here.
```

Whether created using the `document_extraction.py` script or otherwise, the
analysis script expects json data files to be available in directories
named `data/[STATE]`.

### AI agent

The file `comment_summarization.py` is the primary entrypoint to the AI agent.
The agent is built using [`langchain`](https://python.langchain.com/) and has
the following primary features,

* It uses the OpenAI API for LLM queries (the default has been updated to 
the `gpt-4-turbo-2024-04-09` model, but the user should change this to the 
latest or preferred model).
* It can expand beyond the context window of the model in order to summarize
long texts. In particular, it uses a Map-Reduce approac to summarize the full
Montana testimony transcript, which goes beyond the context window size of the
model. The GPT-3.5 model originally used for this analysis had a context window 
of 4,000 tokens. The 4k token model typically accommodates ~5 pages of input, 
whereas oral hearing transcripts often extend to 100 pages or more. The 
Map-Reduce methodology resolves this problem by first using the GPT model to 
summarize each speaker's statement individually. The GPT model can then 
summarize the summaries.
* It uses caching so that the API is not queried twice for the same prompt; it
will load previous results from the database on disk if available.

### Other files

* `utils.py` contains some reusable utilities related to logging and 
 reproducibility.
* `prompt_library.py` contains the prompt templates used in `comment_summarization.py`

