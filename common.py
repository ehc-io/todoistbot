import os
import re
from litellm import litellm, completion
# litellm._turn_on_debug() # Keep commented unless debugging litellm

import logging # Ensure logging is configured if not already
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# Suppress LiteLLM INFO logs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
def clean_filename(filename):
    """
    Clean a filename by:
    1) Converting to lowercase
    2) Replacing special characters and spaces with underscores
    3) Removing multiple consecutive underscores
    """
    # Get the base name and extension separately
    base_name, extension = os.path.splitext(filename)
    
    # Step 1: Convert to lowercase
    base_name = base_name.lower()
    extension = extension.lower()
    
    # Step 2: Replace special characters and spaces with underscores
    # This regex matches any character that is not alphanumeric or underscore
    base_name = re.sub(r'[^a-z0-9_]', '_', base_name)
    
    # Step 3: Replace multiple consecutive underscores with a single underscore
    base_name = re.sub(r'_{2,}', '_', base_name)
    
    # Remove leading and trailing underscores
    base_name = base_name.strip('_')
    
    # Return the cleaned filename with extension
    return base_name + extension

def call_llm_completion(prompt: str, text_model: str, max_tokens: int = 2048) -> str:
        """Helper function to call the LLM completion API."""
        try:
            messages = [{"content": prompt, "role": "user"}]
            # Use context manager for potential temporary env var setting if needed
            # with litellm.utils.set_verbose(False): # Reduce litellm verbosity if desired
            response = completion(
                model=text_model,
                messages=messages,
                max_tokens=max_tokens, # Limit output size
                temperature=0.1, # Lower temperature for factual summary
                # Add other parameters like top_p if needed
            )
            # Accessing content correctly for LiteLLM v1+
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                summary = response.choices[0].message.content.strip()
                # Optional: Post-process summary (remove boilerplate, etc.)
                return summary
            else:
                logger.warning(f"LLM response structure invalid or content missing. Response: {response}")
                return "Summary generation failed (Invalid LLM response)."

        except Exception as e:
            # Catch specific LiteLLM errors if possible
            logger.error(f"Error calling LLM ({text_model}) for summary: {e}")
            # Log traceback for detailed debugging if needed
            # import traceback
            # logger.error(traceback.format_exc())
            return f"Error generating summary: {e}"