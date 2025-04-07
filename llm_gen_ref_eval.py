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
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
import tiktoken
import asyncio
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor


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
###########################################################
###########################################################
def load_prompt_from_file(prompt_path: str) -> str:
    """Load prompt template from a file."""
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        raise ValueError(f"Failed to load prompt from {prompt_path}: {str(e)}")
    
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
        streaming: bool = False,
        max_tokens=4096
    ):
        super().__init__(model_name, temperature, api_key)
        self.top_p = top_p
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.streaming = streaming
        self.max_tokens=max_tokens

    def get_llm(self) -> ChatOpenAI:
        return ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            top_p=self.top_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            streaming=self.streaming,
            max_tokens=self.max_tokens
        )
    
class AnthropicAdapter(LLMAdapter):
    def __init__(
        self, 
        model_name: str, 
        temperature: float, 
        api_key: str,
        streaming: bool = False,
        max_tokens=4096
    ):
        super().__init__(model_name, temperature, api_key)
        self.streaming = streaming
        self.max_tokens=max_tokens

    def get_llm(self) -> ChatAnthropic:
        return ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature,
            api_key=self.api_key,
            streaming=self.streaming,
            max_tokens=self.max_tokens
        )

class GroqAdapter(LLMAdapter):
    
    def get_llm(self) -> ChatGroq:
        return ChatGroq(
            model=self.model_name,
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
        self.chain = (
            RunnablePassthrough() |
            self.prompt |
            self.llm |
            StrOutputParser()
        )

    async def aprocess_prompt(self, prompt_inputs: Dict[str, Any]) -> str:
        try:
            self.rate_limiter.wait()
            response = await self.chain.ainvoke(prompt_inputs)
            return response
        except Exception as e:
            error_msg = f"Error processing prompt asynchronously: {str(e)}"
            print(error_msg)
            raise  # In production, you might want to raise a custom exception

    def process_prompt(self, prompt_inputs: Dict[str, Any]) -> str:
        try:
            self.rate_limiter.wait()
            response = self.chain.invoke(prompt_inputs)
            return response
        except Exception as e:
            error_msg = f"Error processing prompt: {str(e)}"
            print(error_msg)
            raise  # In production, you might want to raise a custom exception


###########################################################
###########################################################
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('processing.log'),
        logging.StreamHandler()
    ]
)

@dataclass
class BatchInfo:
    start_id: int
    end_id: int
    batch_number: int
    token_count: int

class LargeJSONProcessor:
    def __init__(
        self, 
        input_file: str,
        output_dir: str,
        batch_size: int = 8000,
        min_response_size: int = 1024,  # Minimum response size in KB
        model_name: str = "o200k_base"
    ):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.min_response_size = min_response_size * 1024  # Convert to bytes
        self.model_name = model_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.params_file = self.output_dir / 'params.json'
        self.log_file = self.output_dir / 'processing_log.json'
        
    def format_object_for_tokenization(self, obj: Dict) -> str:
        """Format JSON object according to specified rules for token counting."""
        template = (
            f"id: {obj['id']}\n"
            f"evidence: {obj['evidence']}\n"
            f"claim: {obj['claim']}\n"
            f"Entity_in_Claim: {obj['Entity_in_Claim']}\n"
            f"Co_referenced_in_Evidence: {obj['Co_referenced_in_Evidence']}\n"
            f"Co_referenced_in_Text: {obj['Co_referenced_in_Text']}\n"
        )
        return template

    def calculate_object_tokens(self, obj: Dict) -> int:
        """Calculate tokens for a single object using the specified format."""
        formatted_text = self.format_object_for_tokenization(obj)
        return count_tokens(formatted_text, self.model_name)

    def create_batches(self) -> List[BatchInfo]:
        """Create batches of objects respecting token limits and object atomicity."""
        batches = []
        current_batch = []
        current_tokens = 0
        current_batch_num = 0
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for obj in tqdm(data, desc="Creating batches"):
            obj_tokens = self.calculate_object_tokens(obj)
            
            if current_tokens + obj_tokens > self.batch_size and current_batch:
                # Save current batch
                batch_info = BatchInfo(
                    start_id=current_batch[0]['id'],
                    end_id=current_batch[-1]['id'],
                    batch_number=current_batch_num,
                    token_count=current_tokens
                )
                batches.append(batch_info)
                current_batch = []
                current_tokens = 0
                current_batch_num += 1
            
            current_batch.append(obj)
            current_tokens += obj_tokens
        
        # Add the last batch if it exists
        if current_batch:
            batch_info = BatchInfo(
                start_id=current_batch[0]['id'],
                end_id=current_batch[-1]['id'],
                batch_number=current_batch_num,
                token_count=current_tokens
            )
            batches.append(batch_info)
        
        return batches

    def save_batch_params(self, batches: List[BatchInfo]):
        """Save batch parameters to JSON file."""
        params_data = [
            {
                "batch_number": batch.batch_number,
                "start_id": batch.start_id,
                "end_id": batch.end_id,
                "token_count": batch.token_count
            }
            for batch in batches
        ]
        
        with open(self.params_file, 'w', encoding='utf-8') as f:
            json.dump(params_data, f, ensure_ascii=False, indent=2)

    def log_processing(self, batch_num: int, model: str, prompt: str, batch_content: str,response: str):
        """Log processing details to JSON file."""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "batch_number": batch_num,
            "model": model,
            "prompt": prompt,
            "batch_content": batch_content,
            "response":response
        }
        
        # Append to log file
        if self.log_file.exists():
            with open(self.log_file, 'r+', encoding='utf-8') as f:
                logs = json.load(f)
                logs.append(log_entry)
                f.seek(0)
                json.dump(logs, f, ensure_ascii=False, indent=2)
        else:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump([log_entry], f, ensure_ascii=False, indent=2)

    def process_batch(self, batch: BatchInfo, llm_engine: LLMEngine, prompt: str) -> None:
        """Process a single batch with the LLM engine."""
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            # Convert IDs to strings for consistent comparison
            start_id = str(batch.start_id)
            end_id = str(batch.end_id)
            
            # Find the start and end indices in the data array
            start_idx = next((idx for idx, obj in enumerate(data) 
                            if str(obj['id']) == start_id), None)
            end_idx = next((idx for idx, obj in enumerate(data) 
                          if str(obj['id']) == end_id), None)
            
            if start_idx is None or end_idx is None:
                raise ValueError(f"Could not find batch boundaries: start_id={start_id}, end_id={end_id}")
                
            # Extract the batch data using array slicing
            batch_data = data[start_idx:end_idx + 1]
            
            if not batch_data:
                raise ValueError(f"Empty batch data for batch {batch.batch_number}")
        
        # Format batch content
        batch_content = "\n".join(
            self.format_object_for_tokenization(obj) for obj in batch_data
        )
        
        if not batch_content.strip():
            raise ValueError(f"Empty formatted content for batch {batch.batch_number}")

        # Process with LLM
        try:
            prompt_inputs = {
                "task_prompt": prompt,
                "document_content": batch_content
            }
            
            response = llm_engine.process_prompt(prompt_inputs)
            
            # Save response
            response_path = self.output_dir / f"eval_batch_{batch.batch_number}.txt"
            SaveFile(response, response_path)
            
            engine=  llm_engine.llm.model_name,
            
            # Log processing
            self.log_processing(
                batch.batch_number,
                engine,    
                prompt,
                batch_content,
                response
            )
            
        except Exception as e:
            logging.error(f"Error processing batch {batch.batch_number}: {str(e)}")
            raise

def main():
    # Initialize processor
    processor = LargeJSONProcessor(
        input_file="",
        output_dir="",
        batch_size=, #in tokens
        min_response_size=
    )
    
    # Create batches
    batches = processor.create_batches()
    processor.save_batch_params(batches)
    
    # Load prompt
    prompt = load_prompt_from_file("")
    
    # Initialize LLM engine
    llm_adapter =
  
    prompt_template = """
    {task_prompt}
    ##START OF TEXT##
    {document_content}
    ##END OF TEXT##
    """
    llm_engine = LLMEngine(llm_adapter, prompt_template)
    
    # Process batches
    for batch in tqdm(batches, desc="Processing batches"):
        processor.process_batch(batch, llm_engine, prompt)

if __name__ == "__main__":
    main()
