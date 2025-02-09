"""This script contains a variety of prompt templates for use in comment
summarization.
"""

from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

REDUCE_PREAMBLE = """What follows are excerpts of testimony from oral hearings about bills concerning {issue_description} and other topics delivered by many individuals before various state legislatures. 

Some are not substantive and you should ignore them; for example, ignore statements that are simply introducing the session or other speakers. Some are not related to the topic at hand and you should also ignore those; only pay attention to substantive testimony about the bills themselves and issues concerning {issue_description}.

When answering questions,
* Use bullet points
* Cite specific sources, using `Document ID`s with markdown URL notation to cite `Source URL`s
* Be sure to mention the speakers' names, states, and occupations
* When returning multiple examples, be sure to pick a diverse set from multiple states

"""

UNIQUE_REDUCE_PROMPT = PromptTemplate(
    template=REDUCE_PREAMBLE + """Write a concise summary of the {N} most unique arguments delivered within the following texts about {issue_description}. Choose half the examples from each side of the issue and compare and contrast them.


```{text}```


BULLETED LIST:""", 
    input_variables=['issue_description', 'text', 'N']
)


MOST_X_PROMPT = PromptTemplate(
    template=REDUCE_PREAMBLE + """Write a concise summary of the {N} most {adjective} arguments delivered within the texts about {issue_description}. Put the most {adjective} argument first. Don't include any examples that are not {adjective}. Choose half the examples from each side of the issue and compare and contrast them.


```{text}```


BULLETED LIST:""", 
    input_variables=['issue_description', 'text', 'N', 'adjective']
)


VALUES_PROMPT = PromptTemplate(
    template=REDUCE_PREAMBLE + """
    Identify {N} different values expressed about the issue of about {issue_description} and one or a few arguments that best exemplify each. Provide a bulleted list with the values bolded and the example speakers cited as evidence, with their names and states. Choose half the examples from each side of the issue.


```{text}```


BULLETED LIST:""", 
    input_variables=['issue_description', 'text', 'N']
)


STORYTELLING_PROMPT = PromptTemplate(
    template=REDUCE_PREAMBLE + """
    Write a concise summary of the {N} comments about {issue_description} that are presented in the form of a story. Identify the comments that most effectively use narrative and personal experience to express themselves. Group your results into examples that are A) {pro_description} and B) results that are {con_description}. Use markdown headings to seperate them. Choose examples from both sides and try to provide at least {N} total. Explain how each comment uniquely uses storytelling in their presentation. 


```{text}```


BULLETED LIST:""", 
    input_variables=['issue_description', 'text', 'N', 'pro_description', 'con_description']
)



FROM_X_PROMPT = PromptTemplate(
    template=REDUCE_PREAMBLE + """Write a concise summary of the {N} most common arguments from {speaker_type} delivered within the texts about {issue_description}. Cite some examples for each one, referencing speceific document IDs. Don't include any examples that are not from speakers who are {speaker_type}. Choose half the examples from each side of the issue and compare and contrast them.


```{text}```


BULLETED LIST:""", 
    input_variables=['issue_description', 'text', 'N', 'speaker_type']
)

EVIDENCE_PROMPT = PromptTemplate(
    template=REDUCE_PREAMBLE + """Write a concise summary of the {N} (from each side) arguments delivered within the texts that best reflect the types of data and evidence used on each side of the debate on {issue_description}. What data points are being referenced? What sources/groups are being referenced? Include both legitimate and illegitimate (misinformation) and call out which they are. Choose half the examples from each side of the issue and compare and contrast them.


```{text}```


BULLETED LIST:""", 
    input_variables=['issue_description', 'text', 'N']
)


MISINFORMATION_PROMPT = PromptTemplate(
    template=REDUCE_PREAMBLE + """Write a concise summary of the {N} (from each side) arguments delivered within the texts about {issue_description} that best reflect examples of misinformation or disinformation shared by testifiers. Explain specifically what they said that was wrong and why. Don't make up examples or stretch the meaning of "misinformation," but try to return at least {N} examples.


```{text}```


BULLETED LIST:""", 
    input_variables=['issue_description', 'text', 'N']
)


class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: list[int] = Field(
        ...,
        description="The integer Document IDs of the SPECIFIC sources which justify the answer.",
    )
