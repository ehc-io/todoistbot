import os
import re
from typing import List
from datetime import datetime

from litellm import completion
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
        
def generate_markdown(tasks_data: List[dict]) -> str:
    """
    Generate formatted markdown content from processed tasks data.
    """
    content = [
        f"# Todoist Capture Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        # f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]

    # Filter out None values or tasks that resulted in errors
    valid_tasks = []
    for task in tasks_data:
        if not task:
            continue
            
        # Skip tasks with LLM errors or no successfully processed URLs
        has_error = False
        if task.get('processed_urls'):
            # Check if any URL has LLM errors
            for url_data in task.get('processed_urls', []):
                if 'error' in url_data and url_data.get('error'):
                    if url_data.get('content', '').startswith('Error generating summary:') or 'Error calling LLM' in url_data.get('error', ''):
                        logger.info(f"Excluding task with LLM error from report: {task.get('original_task', {}).get('content', 'Unknown Task')}")
                        has_error = True
                        break
                # Check for content extraction errors
                if url_data.get('content', '').startswith('[Error:'):
                    has_error = True
                    break
        
        if not has_error and task.get('processed_urls'):
            valid_tasks.append(task)

    if not valid_tasks:
        content.append("No tasks with successfully scraped content were processed in this run.")
        return "\n".join(content)

    for i, task_result in enumerate(valid_tasks):
        task_content = task_result.get('original_task', {}).get('content', 'Unknown Task')
        task_description = task_result.get('original_task', {}).get('description', '')
        task_labels = task_result.get('original_task', {}).get('labels', [])

        content.extend([
            f"## {task_content}",
            ""
        ])

        # Original Description
        if task_description:
            content.extend([
                "### Original Description",
                f"> {task_description.replace(chr(10), chr(10) + '> ')}", # Blockquote description
                ""
            ])

        # Labels
        if task_labels:
            content.extend([
                "### Labels",
                ", ".join([f"`{label}`" for label in task_labels]),
                ""
            ])

        # Processed URL Content
        for url_data in task_result.get('processed_urls', []):
            url = url_data.get('url', 'N/A')
            scraped_content = url_data.get('content', '[No content extracted]')
            content_type = url_data.get('type', 'unknown')
            extraction_method = url_data.get('extraction_method', '') 
            error = url_data.get('error')
            
            # Get downloaded media information
            downloaded_video = url_data.get('downloaded_video_path', '')
            downloaded_markdown = url_data.get('downloaded_markdown_path', '')
            s3_url = url_data.get('s3_url', '')

            # content.append(f"**URL**: [{url}]({url})  ")
            content.append(f"**Type**: {content_type.capitalize()}  ")
            if extraction_method: 
                content.append(f"*Extraction Method: {extraction_method}*  ")
                content.append("  ")  # Spacer after each URL's content
            
            # Add media links in the new format
            media_links = []
            
            # Add downloaded video link if available
            if s3_url and 'mp4' in s3_url.lower():
                media_links.append(f"- [Video: 1]({s3_url})")
            elif downloaded_video:
                media_links.append(f"- [Video: 1 (Local)]({downloaded_video})")
                
            # Add downloaded markdown link if available
            if downloaded_markdown:
                media_links.append(f"- [Markdown]({downloaded_markdown})")
            
            # Add any other S3 URLs if they're not already covered
            if s3_url and not any(s3_url in link for link in media_links):
                # Determine media type from URL
                if '.jpg' in s3_url.lower() or '.jpeg' in s3_url.lower() or '.png' in s3_url.lower():
                    media_links.append(f"- [Image]({s3_url})")
                elif '.mp3' in s3_url.lower():
                    media_links.append(f"- [Audio]({s3_url})")
                else:
                    media_links.append(f"- [Media]({s3_url})")
            
            # Process the scraped content to replace media section with the new format
            if scraped_content:
                # First, identify if there are multiple media files (pattern for multiple S3 URLs)
                multi_media_pattern = re.compile(
                    r'Media: (\d+) items detected.*?\n\s*'
                    r'Uploaded to S3 \d+ media files?:.*?\n'
                    r'- (Video|Photo|Audio|Image):.*?\n\s*'
                    r'S3 Locations?:(.*?)(?=\n\n|$)',
                    re.DOTALL
                )
                
                def replace_multi_media(match):
                    count = int(match.group(1))
                    media_type = match.group(2)
                    locations_section = match.group(3)
                    
                    # Extract all URLs from the locations section
                    urls = re.findall(r'- (https://[^\s\n]+)', locations_section)
                    
                    # Generate new formatted content
                    result = f"Media: {count} items detected\n\n"
                    
                    # Create properly formatted links for each URL
                    for i, url in enumerate(urls, 1):
                        # Remove the check that limits URLs to count
                        # Display all available URLs in the result
                        result += f"- [{media_type}: {i}]({url})\n"
                    
                    return result
                
                # Try to replace multi-media sections
                processed_content = multi_media_pattern.sub(replace_multi_media, scraped_content)
                
                # If no replacement happened, try the original patterns
                if processed_content == scraped_content:
                    # Look for single media item with more forgiving pattern
                    media_section_pattern = re.compile(
                        r'Media: (\d+) items? detected.*?\n\s*'
                        r'Uploaded to S3 \d+ media files?:.*?\n'
                        r'- (Video|Photo|Audio|Image):.*?\n\s*'
                        r'S3 Locations?:.*?\n'
                        r'- (https://[^\s\n]+)',
                        re.DOTALL
                    )
                    
                    def replace_media_section(match):
                        count = match.group(1)
                        media_type = match.group(2)
                        s3_url = match.group(3)
                        # Return the new format
                        return f"Media: {count} items detected\n\n- [{media_type}: 1]({s3_url})"
                    
                    # Replace all instances of the pattern
                    processed_content = media_section_pattern.sub(replace_media_section, scraped_content)
                    
                    # Check if the replacement happened
                    if processed_content == scraped_content:
                        # Try an alternative pattern if the first one didn't match
                        alt_pattern = re.compile(
                            r'(Media: \d+ items? detected.*?'
                            r'Uploaded to S3.*?'
                            r'S3 Locations?:.*?\n'
                            r'- (https://[^\s\n]+).*?)(?:\n\n|\Z)',
                            re.DOTALL
                        )
                        
                        def alt_replace(match):
                            full_text = match.group(1)
                            s3_url = match.group(2)
                            
                            # Extract media type and count
                            media_type = "Video" if "Video:" in full_text else "Photo" if "Photo:" in full_text else "Media"
                            count_match = re.search(r'Media: (\d+)', full_text)
                            count = count_match.group(1) if count_match else "1"
                            
                            # Extract all URLs from the full text, not just the first one
                            all_urls = re.findall(r'- (https://[^\s\n]+)', full_text)
                            
                            # Format with all extracted URLs
                            result = f"Media: {count} items detected\n\n"
                            for i, url in enumerate(all_urls, 1):
                                result += f"- [{media_type}: {i}]({url})\n"
                                
                            return result
                        
                        processed_content = alt_pattern.sub(alt_replace, scraped_content)
                
                # Final check for any unprocessed URLs in the content after our regexes
                # This catches any trailing URLs that might have been missed
                url_pattern = re.compile(r'^- (https://[^\s\n]+)', re.MULTILINE)
                
                def format_remaining_url(match):
                    url = match.group(1)
                    # Determine media type from URL
                    if '.mp4' in url.lower():
                        media_type = "Video"
                    elif '.jpg' in url.lower() or '.jpeg' in url.lower() or '.png' in url.lower():
                        media_type = "Photo"
                    elif '.mp3' in url.lower():
                        media_type = "Audio"
                    else:
                        media_type = "Media"
                    
                    return f"- [{media_type}]({url})"
                
                # Replace any remaining bare URLs
                processed_content = url_pattern.sub(format_remaining_url, processed_content)
                
                # Format Twitter URLs in the "URLs in tweet:" section
                twitter_urls_pattern = re.compile(r'URLs in tweet:\n((?:\s*- .*?\n)+)', re.MULTILINE)
                
                def format_twitter_urls(match):
                    urls_section = match.group(1)
                    lines = urls_section.split('\n')
                    formatted_lines = []
                    
                    for line in lines:
                        if not line.strip():
                            continue
                        
                        # Match Twitter username pattern (e.g., /OpenAI, /llama_index)
                        username_match = re.match(r'\s*- (/\w+)$', line)
                        if username_match:
                            username = username_match.group(1)
                            # Remove the leading slash for the link text
                            display_name = username.lstrip('/')
                            formatted_lines.append(f" - [{display_name}](https://x.com{username})")
                        else:
                            # Keep other URLs as they are
                            formatted_lines.append(line)
                    
                    return "URLs in tweet:\n" + "\n".join(formatted_lines) + "\n"
                
                # Apply the Twitter URL formatting
                processed_content = twitter_urls_pattern.sub(format_twitter_urls, processed_content)
                
                scraped_content = processed_content

            # Add custom media links to content if any exist and not already in the content
            if media_links and (s3_url not in scraped_content if s3_url else True):
                # content.append("Media:")
                content.extend(media_links)
                content.append("")  # Extra spacing

            if error:
                content.append(f"**Error:** {error}")

            content.append(scraped_content)
            content.append("") # Spacer after each URL's content

        content.extend([
            "---", 
            ""
        ])

    return "\n".join(content)
