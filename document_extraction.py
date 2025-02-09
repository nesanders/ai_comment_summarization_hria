"""Extract discrete statements from a directory of DOCX files (utils.DATA_DIR) containing testimony 
transcriptions from rev.com, formatted as blocks of text like "Speaker 1 (00:00:56)".

The extracted statements including text and speaker metadata are written to json files named and
placed alongside each docx file.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

from docx import Document

from utils import DATA_DIR, setup_logging

LOGGER = setup_logging('document_extraction')
MINIMUM_CHARACTERS = 100
SPEAKER_REGEXP = r"^Speaker (\d+) \((\d{2}:\d{2})(:\d{2})?\):"

def extract_if_speaker(text: str) -> Optional[tuple[int, str]]:
    """Check if the string `text` matches a 'Speaker' declaration. If not,
    return None. If so, return the speaker number and timestamp.

    Example: "Speaker 1 (00:00:56):"
    Would return (1, '00:00:56')
    """
    matches = re.match(SPEAKER_REGEXP, text, re.MULTILINE)
    if matches:
        return matches.groups()
    return None

def process_doc(docx_path: Path) -> list[dict[str, str]]:
    """Process a docx word doc and return a list of extracted statements.
    """
    output = []
    document = Document(docx_path)
    current_speaker = ('Introduction', '00:00:00')
    current_text = ''
    current_url = ''
    for para in document.paragraphs:
        extracted_speaker = extract_if_speaker(para.text)
        if extracted_speaker is None:
            # The current paragraph is not a speaker paragraph, so add it to the
            # current text variable if it's not just a timestamp or other
            # trivial statement
            if len(para.text) > MINIMUM_CHARACTERS:
                current_text += para.text + ' '
        else:
            # The current paragraph is a speaker paragraph
            # If we have collected sufficient text, store it
            if len(current_text) >= MINIMUM_CHARACTERS:
                output.append({
                    'speaker_id': current_speaker[0],
                    'timestamp': current_speaker[1],
                    'text': current_text,
                    'source_url': current_url
                    })
            # Reset the current variables
            current_text = ''
            current_speaker = extracted_speaker
            current_url = para.hyperlinks[0].url
    return output

if __name__ == '__main__':
    # Find all files in the DATA_DIR
    all_document_paths = list(DATA_DIR.glob('**/*.docx'))
    
    for doc_path in all_document_paths:
        LOGGER.info(f'Processing {doc_path}')
        doc_json = process_doc(doc_path)
        out_path = doc_path.with_suffix('.json')
        with open(out_path, 'w') as f:
            json.dump(doc_json, f)

