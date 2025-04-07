##############################################################################
##############################################################################
#MIT License

#Copyright (c) 2024 Christophe Khalil
#Email: christophe.khalil@outlook.com / cak29@mail.aub.edu

#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:

#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.

#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.
##############################################################################
##############################################################################


# Standard library imports
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import re

# Typing imports
from typing import (
    Any,
    Dict,
    Generator,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

# Third-party imports
import openai
import tiktoken
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_anthropic import ChatAnthropic
from langchain_cohere import ChatCohere
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

################# Document Loading and IO Handling code #############################
from typing import Dict, Pattern


def clean_wiki_text(text):
    """
    Clean messy WikiExtractor output, particularly for Arabic text.
    Handles nested links, remaining wiki markup, categories, and other formatting issues.
    
    Args:
        text (str): Raw text from WikiExtractor
    Returns:
        str: Cleaned text
    """
    if not text:
        return text
    
    # Remove category tags (تصنيف) and their content
    text = re.sub(r'\[\[:?[تТ]صنيف:.*?\]\]', '', text)  # Handle both Arabic and look-alike characters
    text = re.sub(r'\[\[:?Category:.*?\]\]', '', text)   # English category tags
    text = re.sub(r'\{\{[تТ]صنيف\|.*?\}\}', '', text)   # Category templates
    
    # Function to process internal wiki links
    def replace_wiki_link(match):
        # Split by | if it exists, take the last part (display text)
        parts = match.group(1).split('|')
        return parts[-1].strip()
    
    # 1. Handle nested wiki links - work from inside out
    prev_text = None
    while '[[' in text and ']]' in text:
        prev_text = text
        text, changes = re.subn(r'\[\[([^\[\]]+?)\]\]', replace_wiki_link, text)
        if changes == 0:  # Break if no changes are made to prevent infinite loops
            break
    
    # 2. Clean up any remaining square brackets
    text = re.sub(r'\[\[|\]\]', '', text)
    
    # 3. Remove parenthetical dates and numbers
    text = re.sub(r'\(\[?\d+\]?[م]?\)', '', text)
    
    # 4. Clean up various wiki markup
    cleanup_patterns = [
        (r'#[^#\n]*?\|', ''),  # Remove section links
        (r'\{\{[^\}]+\}\}', ''), # Remove templates
        (r'<[^>]+>', ''),  # Remove HTML tags
        (r'\|[^\|\n\]]+\|', '|'),  # Clean up table markup
        (r'\'{2,}', ''),  # Remove bold/italic markup
        (r'={2,}.*?={2,}', ''),  # Remove headers
        (r'ملف:', ''),  # Remove file prefix
        (r'صورة:', ''),  # Remove image prefix
        (r'\[\[(?:File|Image|صورة|ملف):.*?\]\]', ''),  # Remove file/image links
        (r'\n\s*\n\s*\n', '\n\n'),  # Normalize multiple newlines
    ]
    
    for pattern, replacement in cleanup_patterns:
        text = re.sub(pattern, replacement, text)
    
    # 5. Fix spacing issues
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = re.sub(r'\s*\|\s*', ' ', text)  # Clean up remaining pipes
    text = re.sub(r'،\s*', '، ', text)  # Fix Arabic comma spacing
    text = re.sub(r'\s+([.،؟!])', r'\1', text)  # Fix punctuation spacing
    
    # 6. Remove common wiki template remnants
    text = re.sub(r'\{\{.*?\}\}', '', text)  # Remove any remaining templates
    text = re.sub(r'__[A-Z]+__', '', text)  # Remove magic words
    text = re.sub(r'&[a-zA-Z]+;', '', text)  # Remove HTML entities
    
    # 7. Final trimming
    text = text.strip()
    
    return text


def load_documents(directory: str) -> Generator[Document, None, None]:
    """
    Load JSON files from a directory and yield LangChain Document objects.

    Args:
        directory (str): Path to the directory containing JSON files.

    Yields:
        Document: LangChain Document object for each JSON file.
    """
    for entry in os.scandir(directory):
        if entry.is_file() and entry.name.endswith('.json'):
            try:
                with open(entry.path, 'r', encoding='utf-8') as file:
                    data = json.load(file)
                    yield Document(
                        page_content= clean_wiki_text(data['text']),
                        metadata={
                            'id': data['id'],
                            'url': data['url'],
                            'title': data['title'],
                            'split_id': None  # Initialize split_id as None
                        }
                    )
            except (json.JSONDecodeError, KeyError, IOError) as e:
                print(f"Error processing {entry.name}: {str(e)}")
def count_tokens(text: str, model: str = "o200k_base") -> int:
    """
    Count the number of tokens in a given text.
    Args:
        text (str): The input text.
        model (str): The name of the tokenizer model to use (default: "o200k_base").
    Returns:
        int: The number of tokens in the text.
    """
    try:
        encoding = tiktoken.get_encoding(model)
        return len(encoding.encode(text))
    except KeyError:
        print(f"Warning: Model '{model}' not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

def split_document(doc: Document, chunk_size: int, chunk_overlap: int,
                   min_paragraph_tokens: int) -> List[Document]:
    """
    Split a document into chunks based on token count and merge small paragraphs.
    If document doesn't need splitting, returns it with split_id = 0.
    Args:
        doc (Document): The input document.
        chunk_size (int): Maximum number of tokens per chunk.
        chunk_overlap (int): Number of overlapping tokens between chunks.
        min_paragraph_tokens (int): Minimum number of tokens to consider as a paragraph.
    Returns:
        List[Document]: List containing either the original document (with split_id=0) or its splits.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=count_tokens,
    )
    
    # Check if the document needs splitting
    if count_tokens(doc.page_content) <= chunk_size:
        return [Document(
            page_content=doc.page_content,
            metadata={**doc.metadata, 'split_id': 0}
        )]
    
    # Split the document
    splits = text_splitter.split_text(doc.page_content)
    
    # Merge small paragraphs with the previous chunk
    merged_splits = []
    current_split = splits[0]
    
    for i in range(1, len(splits)):
        current_tokens = count_tokens(current_split)
        next_split = splits[i]
        next_tokens = count_tokens(next_split)
        
        # If the next split is too small, append it to current_split
        if next_tokens < min_paragraph_tokens:
            current_split = current_split + " " + next_split
        else:
            # If current_split is too small, append it to the previous chunk
            if current_tokens < min_paragraph_tokens and merged_splits:
                merged_splits[-1] = merged_splits[-1] + " " + current_split
            else:
                merged_splits.append(current_split)
            current_split = next_split
    
    # Handle the last chunk
    if count_tokens(current_split) < min_paragraph_tokens and merged_splits:
        merged_splits[-1] = merged_splits[-1] + " " + current_split
    else:
        merged_splits.append(current_split)
    
    # Create new documents with split IDs starting from 1
    return [
        Document(
            page_content=split,
            metadata={**doc.metadata, 'split_id': i + 1}
        ) for i, split in enumerate(merged_splits)
    ]

def count_words_and_tokens(text: str, model: str = "o200k_base") -> Dict[str, int]:
    """
    Count words and tokens in the given text.

    Args:
        text (str): The input text.
        model (str): The model name for token counting (default: "o200k_base").

    Returns:
        Dict[str, int]: A dictionary containing word and token counts.
    """
    words = len(text.split())
    tokens = count_tokens(text, model)
    return {"words": words, "tokens": tokens}

def process_documents(directory: str, chunk_size: int = 1000, chunk_overlap: int = 0,
                      min_paragraph_size: int = 500) -> List[Union[Document, List[Document]]]:
    """
    Process all documents in the given directory, grouping splits together.

    Args:
        directory (str): Path to the directory containing JSON files.
        chunk_size (int): Maximum number of tokens per chunk (default: 1000).
        chunk_overlap (int): Number of overlapping tokens between chunks (default: 100).
        min_paragraph_size (int): Minimum number of tokens to consider as a paragraph (default: 200).
jvh
    Returns:
        List[Union[Document, List[Document]]]: Processed documents with splits grouped.
    """
    start_time = time.time()
    processed_docs = []

    for doc in load_documents(directory):
        split_result = split_document(doc, chunk_size, chunk_overlap, min_paragraph_size)
        if len(split_result) == 1:
            processed_docs.append(split_result[0])
        else:
            processed_docs.append(split_result)

    end_time = time.time()
    print(f"Processed {len(processed_docs)} documents in {end_time - start_time:.2f} seconds.")

    return processed_docs

def output_processed_documents(processed_docs: List[Union[Document, List[Document]]]):
    """
    Output the processed documents in a structured and informative way.

    Args:
        processed_docs (List[Union[Document, List[Document]]]): The list of processed documents.
    """
    total_docs = 0
    total_splits = 0
    in_total = 0  #total numbre at the end
    print("Processed Documents Overview:")
    print("=============================")

    for i, item in enumerate(processed_docs, 1):
        if isinstance(item, Document):
            total_docs += 1
            in_total = total_docs
            doc_id = item.metadata.get('id', 'Unknown')
            token_count = count_tokens(item.page_content)
            print(f"Document {i}: ID {doc_id}")
            print(f"  - Tokens: {token_count}")
            print(f"  - Split: No")
            print()
        elif isinstance(item, list):
            total_docs += 1
            total_splits += len(item)
            in_total += total_splits
            doc_id = item[0].metadata.get('id', 'Unknown')
            print(f"Document {i}: ID {doc_id}")
            print(f"  - Splits: {len(item)}")
            for j, split in enumerate(item, 1):
                split_id = split.metadata.get('split_id', 'Unknown')
                token_count = count_tokens(split.page_content)
                print(f"    Split {j}: ID {split_id}")
                print(f"      - Tokens: {token_count}")
            print()

    print("Summary:")
    print(f"Total Documents: {total_docs}")
    print(f"Total Splits: {total_splits}")
    print(f"Total Processed Items: {in_total}")

def flatten_documents(nonflat_documents: List[Union[Document, List[Document]]]) -> List[Document]:
    """
    Process a list of documents or lists of documents, ensuring that all items are Document objects.

    Args:
        nonflat_documents (List[Union[Document, List[Document]]]): A list of documents or lists of documents.

    Returns:
        List[Document]: A list of all the documents.
    """
    documents = []
    i = 0

    for doc in nonflat_documents:
        if isinstance(doc, Document):
            documents.append(doc)
        else:
            for split in doc:
                i += 1
                documents.append(split)
    return documents

def load_text_file_to_string(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        return "File not found."
    except Exception as e:
        return f"An error occurred: {e}"

############### LLM Adapter Classes and API Access ######################
class LLMAdapter(ABC):
    def __init__(self, model_name: str, temperature: float, api_key: str):
        self.model_name = model_name
        self.temperature = temperature
        self.api_key = api_key

    @abstractmethod
    def get_llm(self) -> Any:
        pass

class OpenAIAdapter(LLMAdapter):
    def __init__(
        self, 
        model_name: str, 
        temperature: float, 
        api_key: str,
        top_p: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        streaming: bool = False
    ):
        super().__init__(model_name, temperature, api_key)
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.streaming = streaming

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            streaming=self.streaming
        )
    
class AnthropicAdapter(LLMAdapter):
    def get_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )

class GroqAdapter(LLMAdapter):
    def get_llm(self) -> ChatGroq:
        return ChatGroq(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )

class CohereAdapter(LLMAdapter):
    def get_llm(self) -> ChatCohere:
        return ChatCohere(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key
        )

class RateLimiter:
    def __init__(self, max_calls: int, time_frame: float):
        self.max_calls = max_calls
        self.time_frame = time_frame
        self.calls = []

    def wait(self):
        current_time = time.time()
        self.calls = [call for call in self.calls if current_time - call < self.time_frame]
        if len(self.calls) >= self.max_calls:
            sleep_time = self.calls[0] + self.time_frame - current_time
            time.sleep(max(0, sleep_time))
        self.calls.append(time.time())

class LLMEngine:
    def __init__(self, adapter: LLMAdapter, prompt_template: str,
                 max_calls_per_minute: int = 20):
        self.llm = adapter.get_llm()
        self.prompt = PromptTemplate.from_template(prompt_template)
        self.rate_limiter = RateLimiter(max_calls_per_minute, 1)

        # Build the LCEL chain with proper input/output handling
        self.chain = (
            RunnablePassthrough() |
            self.prompt |
            self.llm |
            StrOutputParser()
        )

    async def aprocess_prompt(self, prompt_inputs: Dict[str, Any]) -> str:
        """Async version of process_prompt"""
        try:
            self.rate_limiter.wait()
            response = await self.chain.ainvoke(prompt_inputs)
            return response
        except Exception as e:
            error_msg = f"Error processing prompt asynchronously: {str(e)}"
            print(error_msg)
            raise  # In production, you might want to raise a custom exception

    def process_prompt(self, prompt_inputs: Dict[str, Any]) -> str:
        """Synchronous version of process_prompt"""
        try:
            self.rate_limiter.wait()
            response = self.chain.invoke(prompt_inputs)
            return response
        except Exception as e:
            error_msg = f"Error processing prompt: {str(e)}"
            print(error_msg)
            raise  # In production, you might want to raise a custom exception

#####################################################################
#TASK1 :SPECFIC CODE
#####################################################################
import os
import json
import time
from typing import List, Union, Dict
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from collections import deque
from langchain_core.prompts import PromptTemplate

@dataclass
class ProcessingStats:
    session_id: str
    start_time: datetime
    processed_count: int = 0
    total_count: int = 0
    estimated_time: float = 0.0
    first_call_time: float = 0.0

def FileSize(file: Union[str, Path], unit: str = 'B') -> float:
    """
    Calculate file size in specified unit
    Args:
        file: Path to file
        unit: Unit to return size in ('B', 'KB', 'MB', 'GB')
    Returns:
        float: File size in specified unit
    """
    size_bytes = os.path.getsize(file)
    units = {'B': 0, 'KB': 1, 'MB': 2, 'GB': 3}
    
    if unit not in units:
        raise ValueError(f"Invalid unit. Must be one of {list(units.keys())}")
        
    return size_bytes / (1024 ** units[unit])

def SaveFile(file_txt: str, filepath: Union[str, Path], encoding: str = 'utf-8') -> None:
    """
    Save text content to file ensuring directory exists
    Args:
        file_txt: Content to save
        filepath: Path to save file
        encoding: File encoding (default: utf-8)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding=encoding) as f:
        f.write(file_txt)

def alert_user():
    """Alert function using print bell"""
    for _ in range(10):
        print('\a', end='', flush=True)
        time.sleep(0.1)

def Task1PipeLine(
    indir: str, 
    outdir: str, 
    prompt: str, 
    chunksize: int, 
    min_file_threshold: float,
    llm_adapter: LLMAdapter,
    max_calls_per_minute: int = 20
) -> List[str]:
    # Initialize stats and logging
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    start_time = datetime.now()
    stats = ProcessingStats(
        session_id=session_id,
        start_time=start_time
    )
    
    # Print session start time
    print(f"\nSession started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Setup logging
    log_file = Path(outdir) / f"processing_log_{session_id}.json"
    
    # Create if doesn't exist with empty array
    if not log_file.exists():
        SaveFile("[]", log_file)
    
    def save_log_entry(entry: dict):
        """Save a log entry to the JSON log file"""
        with open(log_file, 'r', encoding='utf-8') as f:
            logs = json.load(f)
        logs.append({
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            **entry
        })
        SaveFile(json.dumps(logs, ensure_ascii=False, indent=2), log_file)
    
    prompt_template = """
    {task_prompt}
    ##START OF SUPPORTING TEXT##
    {document_content}
    ##END OF SUPPORTING TEXT##
    """
    
    llm_engine = LLMEngine(
        adapter=llm_adapter,
        prompt_template=prompt_template,
        max_calls_per_minute=max_calls_per_minute
    )

    nonflat_documents = process_documents(indir, chunk_size=chunksize,min_paragraph_size=100)
    documents = flatten_documents(nonflat_documents)
    stats.total_count = len(documents)
    results = []
    print(f"Chunks to process:{len(documents)}")
    # Use tqdm with ETA
    with tqdm(total=len(documents), desc="Processing documents", unit="doc") as pbar:
        
        for i, doc in enumerate(documents):
            doc_id = doc.metadata['id']
            split_id = doc.metadata['split_id']
            response_path = Path(outdir) / f"{doc_id}_{split_id}.txt"
            
            # Check if the output file is already present
            if response_path.exists():
                # If the file is present, skip processing and update the progress bar
                print(f"Skipping document {doc_id}_{split_id} as the output file already exists.")
                stats.processed_count += 1
                pbar.update(1)
                continue
            
            try:
                prompt_inputs = {
                    "task_prompt": prompt,
                    "document_content": doc.page_content
                }
                
                start_time = time.time()
                response = llm_engine.process_prompt(prompt_inputs)
                processing_time = time.time() - start_time
                
                if i == 0:
                    stats.first_call_time = processing_time
                    stats.estimated_time = processing_time * stats.total_count
                
                SaveFile(response, response_path)
                
                if FileSize(response_path, 'KB') < min_file_threshold:
                    alert_user()
                    error_msg = f"Critical Error: Output file size below threshold ({min_file_threshold}KB) for document {doc_id}. Stopping processing."
                    print(f"\n{error_msg}", file=sys.stderr)
                    
                    # Log the critical error
                    save_log_entry({
                        "doc_id": doc_id,
                        "split_id": split_id,
                        "error": error_msg,
                        "file_size": FileSize(response_path, 'KB'),
                        "threshold": min_file_threshold,
                        "status": "critical_error",
                        "prompt": prompt_inputs,
                        "response": response,
                        "llm_model": llm_adapter.model_name
                    })
                    
                    # Clean up the potentially incomplete response file
                    response_path.unlink(missing_ok=True)
                    #we still have to update the bar even if a file failed.
                    stats.processed_count += 1
                    pbar.update(1)
                    return results

                # Log successful processing
                save_log_entry({
                    "doc_id": doc_id,
                    "split_id": split_id,
                    "prompt": prompt_inputs,
                    "response": response,
                    "llm_model": llm_adapter.model_name,
                    "processing_time": processing_time,
                    "file_size": FileSize(response_path, 'KB'),
                    "status": "success"
                })
                
                results.append(response)
                stats.processed_count += 1
                pbar.update(1)
                
            except Exception as e:
                error_msg = f"Error processing document {doc.metadata.get('id', 'unknown')}: {str(e)}"
                print(f"\n{error_msg}", file=sys.stderr)
                
                # Log error
                save_log_entry({
                    "doc_id": doc.metadata.get('id', 'unknown'),
                    "split_id": doc.metadata.get('split_id', 'unknown'),
                    "error": str(e),
                    "status": "error"
                })
    
    return results

import argparse
from datetime import datetime
from pathlib import Path
from typing import List
import sys


def load_prompt_from_file(prompt_path: str) -> str:
    """Load prompt template from a file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"Failed to load prompt from {prompt_path}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(
        description="Process documents using Task1PipeLine with LLM integration"
    )
    
    parser.add_argument(
        "--indir",
        type=str,
        required=True,
        help="Input directory containing documents to process"
    )
    
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory for processed results"
    )
    
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to file containing the prompt template"
    )
    
    parser.add_argument(
        "--chunksize",
        type=int,
        default=1000,
        help="Size of document chunks to process (default: 1000)"
    )
    
    parser.add_argument(
        "--min-file-threshold",
        type=float,
        default=0.1,
        help="Minimum file size threshold in KB (default: 0.1)"
    )
    
    parser.add_argument(
        "--max-calls-per-minute",
        type=int,
        default=20,
        help="Maximum number of LLM API calls per minute (default: 20)"
    )
    
    args = parser.parse_args()
    
    # Verify prompt file exists
    if not Path(args.prompt_file).exists():
        print(f"Error: Prompt file not found: {args.prompt_file}", file=sys.stderr)
        sys.exit(1)
    
    # Hardcoded LLM adapter configuration
    llm_adapter = OpenAIAdapter()

    # Create output directory if it doesn't exist
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load prompt from file
        prompt = load_prompt_from_file(args.prompt_file)
        
        results = Task1PipeLine(
            indir=args.indir,
            outdir=args.outdir,
            prompt=prompt,
            chunksize=args.chunksize,
            min_file_threshold=args.min_file_threshold,
            llm_adapter=llm_adapter,
            max_calls_per_minute=args.max_calls_per_minute
        )
        print(f"\nProcessing completed. Results saved to: {args.outdir}")
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
