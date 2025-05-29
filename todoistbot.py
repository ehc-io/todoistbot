#!/usr/bin/env python3
import os
import re
import sys
import argparse
import time
from typing import List, Optional, Dict, Any
from todoist_api_python.api import TodoistAPI
from datetime import datetime
from pathlib import Path # Use pathlib for paths

# Import the URL database manager
from db_manager import URLDatabase

# Import helper functions
from common import generate_markdown

# --- Import the refactored scraper ---
from url_scraper import URLScraper
# --- Import the YouTube extractor directly ---
from yt_extractor import YouTubeDataFetcher, enhance_youtube_processing
# ---

import logging # Ensure logging is used

def setup_logging(verbose=False):
    """
    Set up logging with appropriate filters and levels.
    This centralizes all logging configuration for the application.
    
    Args:
        verbose: Whether to enable DEBUG level logging
    """
    # Configure basic logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # --- Custom Log Filters ---
    # Filter to ignore TeX math warnings
    class TexMathFilter(logging.Filter):
        def filter(self, record):
            # Skip any log messages about TeX math conversion
            message = record.getMessage()
            if "Could not convert TeX math" in message or "rendering as TeX" in message:
                return False
            return True
    
    # Filter for Monaco Editor errors (Google Colab related)
    class MonacoEditorFilter(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            
            # These patterns indicate Monaco editor/VS Code related errors
            monaco_patterns = [
                "monaco_editor", "vs/editor", "vs/css", "editor.main",
                "gstatic.com/colaboratory", "colab", "monaco",
                "modules that depend on it", "loading failed",
                "could not find", "vs/base/browser/ui", 
                "Critical browser error", "actionbar", "browser/ui"
            ]
            
            # If this is a warning/error log and contains monaco patterns, filter it out
            if record.levelno >= logging.WARNING:
                if any(pattern in message.lower() for pattern in monaco_patterns):
                    return False
                
                # Check for lists of VS Code modules (common in dependency lists)
                if message.startswith("[vs/") or ("]" in message and "vs/" in message):
                    return False
                
                # Filter long stack traces from browser errors
                if "at Object." in message and "js:" in message:
                    return False
                
                # Filter messages about dependencies
                if "Here are the modules that depend on it:" in message:
                    return False
                    
            return True
    
    # Filter for HTTP API requests
    class HTTPRequestFilter(logging.Filter):
        def filter(self, record):
            message = record.getMessage()
            
            # Filter out successful HTTP requests
            if "HTTP Request:" in message:
                # Only keep error responses (4xx, 5xx)
                if "HTTP/1.1 200" in message or "HTTP/1.1 201" in message or "HTTP/1.1 204" in message:
                    return False
                    
            # Filter common API endpoints
            if "api/show" in message or "api/generate" in message:
                # Always filter successful responses
                if "HTTP/1.1 200" in message:
                    return False
                
            return True
    
    # --- Apply Filters ---
    # Create filter instances
    tex_filter = TexMathFilter()
    monaco_filter = MonacoEditorFilter()
    http_filter = HTTPRequestFilter()
    
    # Set log level based on verbosity
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.getLogger().setLevel(log_level)
    
    # Apply filters to all existing loggers
    for logger_name in logging.Logger.manager.loggerDict:
        log_instance = logging.getLogger(logger_name)
        log_instance.addFilter(tex_filter)
        log_instance.addFilter(monaco_filter)
        log_instance.addFilter(http_filter)
    
    # Apply filters to root logger
    root_logger = logging.getLogger()
    root_logger.addFilter(tex_filter)
    root_logger.addFilter(monaco_filter)
    root_logger.addFilter(http_filter)
    
    # --- Configure Specific Loggers ---
    # Set higher levels for noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pypandoc").setLevel(logging.INFO)
    logging.getLogger("playwright").setLevel(logging.WARNING)  # Reduce Playwright noise
    
    # Browser console logging (extremely noisy)
    browser_console_logger = logging.getLogger("playwright.browser.console")
    browser_console_logger.setLevel(logging.ERROR)  # Only show errors, not warnings
    
    # Return the main application logger
    return logging.getLogger("main")

# Initialize logging with default settings
logger = setup_logging()

class TaskStats:
    def __init__(self):
        self.total_tasks_considered = 0 # Tasks initially fetched
        self.tasks_to_process = 0       # Tasks attempting processing (after label/limit filter)
        self.successful_tasks = 0       # Tasks processed yielding content
        self.failed_tasks = 0           # Tasks attempted but failed (error or no content)
        self.skipped_tasks = 0          # Tasks skipped due to 'not-scrapeable' label
        self.notebook_tasks_skipped = 0 # Tasks skipped due to containing notebook URLs
        self.tasks_remaining_in_capture = 0 # Tasks left in Capture project after run
        self.skipped_urls = 0           # URLs skipped due to duplication
        self.processed_urls = 0         # New URLs processed

    def print_summary(self):
        print("\n=== Task Processing Summary ===")
        print(f"Tasks initially found in 'Capture': {self.total_tasks_considered}")
        print(f"Tasks attempted processing:        {self.tasks_to_process}")
        print(f"Tasks skipped ('not-scrapeable'):  {self.skipped_tasks}")
        print(f"Tasks skipped (notebook URLs):     {self.notebook_tasks_skipped}")
        print(f"Successfully processed (got content): {self.successful_tasks}")
        print(f"Failed processing (error/no content): {self.failed_tasks}")
        print(f"Tasks remaining in 'Capture':      {self.tasks_remaining_in_capture}")
        print(f"URLs skipped (already processed): {self.skipped_urls}")
        print(f"New URLs processed: {self.processed_urls}")
        print("==============================")

class URLExtractor:
    """Uses the static methods from URLScraper"""
    @staticmethod
    def extract_relevant_urls(text: str) -> list[str]:
        """Extracts and cleans URLs relevant for scraping."""
        if not text: return []
        # Use the improved extraction and cleaning from URLScraper
        raw_urls = URLScraper.extract_urls(text)
        # Optional: Add more filtering here if needed (e.g., ignore certain domains)
        return [url for url in raw_urls if URLScraper.is_valid_url(url)]
        
    @staticmethod
    def is_notebook_url(url: str) -> bool:
        """Checks if URL is a Jupyter notebook or Google Colab."""
        # Use URLScraper method 
        return URLScraper.is_notebook_url(url)

def get_api_key() -> str:
    api_key = os.getenv('TODOIST_API_KEY')
    if not api_key:
        raise ValueError("TODOIST_API_KEY environment variable not set")
    return api_key


def custom_scrape_url(
    url: str,
    text_model: str = "ollama/llama3.2:3b",
    vision_model: str = "ollama/llava:7b",
    save_media_locally: bool = False,
    media_output_dir: str = "downloads",
    use_search_fallback: bool = True,
    s3_upload: bool = False,
    s3_bucket: str = None
) -> Optional[Dict[str, Any]]:
    """
    Wrapper around URLScraper.scrape_url that skips YouTube processing.
    This prevents double-processing of YouTube URLs.
    """
    # Clean the URL first to handle extra parameters 
    cleaned_url = URLScraper.clean_url(url)
    
    # Skip if it's a YouTube URL (we handle those separately)
    if URLScraper.is_youtube_url(cleaned_url):
        return {
            'url': url,
            'type': 'youtube',
            'error': 'YouTube URLs are handled by our enhanced processor',
            'content': '[Error: YouTube processing attempted through general URL scraper]'
        }
    
    # For non-YouTube URLs, use the original scraper
    return URLScraper.scrape_url(
        url,
        text_model=text_model,
        vision_model=vision_model,
        save_media_locally=save_media_locally,
        media_output_dir=media_output_dir,
        use_search_fallback=use_search_fallback,
        s3_upload=s3_upload,
        s3_bucket=s3_bucket
    )

def process_single_task(api: TodoistAPI, task, args, url_db=None) -> Optional[dict]:
    """
    Processes a single Todoist task: extracts URLs, scrapes them, and returns results.
    Handles adding 'not-scrapeable' label and optionally closing the task.
    
    Args:
        api: TodoistAPI client
        task: Todoist task object
        args: Command line arguments
        url_db: URLDatabase instance for deduplication
        
    Returns:
        Dictionary with processing results or None
    """
    NOT_SCRAPEABLE_LABEL = "not-scrapeable"
    task_id = task.id
    task_content = task.content or ""
    task_description = task.description or ""
    task_labels = task.labels or []

    logger.info(f"→ Processing task {task_id}: {task_content[:50]}...")

    if NOT_SCRAPEABLE_LABEL in task_labels:
        logger.info(f"  Skipping task {task_id}: Already has '{NOT_SCRAPEABLE_LABEL}' label.")
        return {'status': 'skipped', 'task_id': task_id}

    # Extract URLs from both content and description
    urls_from_content = URLExtractor.extract_relevant_urls(task_content)
    urls_from_description = URLExtractor.extract_relevant_urls(task_description)
    # Combine and deduplicate while preserving order roughly
    all_urls = list(dict.fromkeys(urls_from_content + urls_from_description))

    # Check if any URL is a Jupyter notebook or Colab link (temporary debug mechanism)
    has_notebook_url = False
    for url in all_urls:
        if URLExtractor.is_notebook_url(url):
            logger.info(f"  Task contains a Jupyter notebook or Colab URL: {url}")
            has_notebook_url = True
            break
            
    if has_notebook_url:
        logger.info(f"  Marking task {task_id} as '{NOT_SCRAPEABLE_LABEL}' (contains notebook URL)")
        updated_labels = task_labels + [NOT_SCRAPEABLE_LABEL]
        api.update_task(task_id=task_id, labels=updated_labels)
        args.stats.notebook_tasks_skipped += 1
        return {'status': 'skipped', 'task_id': task_id}

    if not all_urls:
        logger.info(f"  No valid URLs found in task {task_id}.")
        # Decide if no URLs means failure or just nothing to do. Let's mark as failure to add label.
        # return {'status': 'no_urls', 'task_id': task_id} # Option: Treat as success with no action
        mark_as_failed = True # Treat no URLs as a failure case for labeling
        processed_url_results = []
    else:
        logger.info(f"  Found URLs: {', '.join(all_urls)}")
        processed_url_results = []
        any_scrape_successful = False
        any_url_processed = False  # Track if any URL was actually processed (not just skipped)
        skipped_urls = 0
        processed_urls = 0
        
        # Track already processed YouTube video IDs to avoid duplicates
        processed_youtube_ids = set()

        for url in all_urls:
            logger.debug(f"  Scraping URL: {url}")
            
            # Check if URL has been processed before
            if url_db and url_db.url_exists(url):
                logger.info(f"  Skipping already processed URL: {url}")
                skipped_urls += 1
                continue  # Skip this URL, move to next one
                
            # Mark that we're processing at least one URL
            any_url_processed = True
            processed_urls += 1
            
            # Handle YouTube URLs with enhanced processing
            if URLScraper.is_youtube_url(url):
                # Check if this video ID has already been processed
                from yt_extractor import YouTubeDataFetcher
                video_id = YouTubeDataFetcher.extract_video_id(url)
                
                if video_id in processed_youtube_ids:
                    logger.info(f"  Skipping duplicate YouTube URL: {url} (already processed video ID: {video_id})")
                    continue
                
                logger.info(f"  Processing YouTube URL: {url}")
                youtube_result = enhance_youtube_processing(
                    url, 
                    text_model=args.text_model,
                    download_youtube_video=args.save_media_locally,
                    youtube_output_dir=args.output_dir,
                    s3_upload=args.s3_upload,
                    s3_bucket=args.s3_bucket
                )
                
                # Remember this video ID to avoid reprocessing
                if video_id:
                    processed_youtube_ids.add(video_id)
                  
                processed_url_results.append(youtube_result)
                if youtube_result and youtube_result.get('content') and not youtube_result.get('error'):
                    any_scrape_successful = True
                    if youtube_result.get('downloaded_video_path'):
                        logger.info(f"  ✓ Successfully processed and downloaded video for: {url}")
                    
                    if youtube_result.get('s3_url'):
                        logger.info(f"  ✓ Successfully uploaded video to S3: {youtube_result.get('s3_url')}")
                    else:
                        logger.info(f"  ✓ Successfully processed YouTube URL: {url}")
                    
                    # Add successfully processed URL to the database
                    if url_db:
                        url_db.add_url(url, task_id=task_id, success=True)
                else:
                    logger.warning(f"  ⚠ Failed to process YouTube URL: {url} (Error: {youtube_result.get('error', 'Unknown error')})")
                    # Track failed URL in database
                    if url_db:
                        url_db.add_url(url, task_id=task_id, success=False)
                
            # Skip YouTube handling in URL scraper by using a custom processing wrapper
            else:
                # For non-YouTube URLs, use the existing scrape_url method
                try:
                    # Use custom_scrape_url to handle non-YouTube URLs
                    scrape_result = custom_scrape_url(
                        url,
                        text_model=args.text_model,
                        vision_model=args.vision_model,
                        save_media_locally=args.save_media_locally,
                        media_output_dir=args.output_dir,
                        use_search_fallback=not args.no_search_fallback,
                        s3_upload=args.s3_upload,
                        s3_bucket=args.s3_bucket
                    )

                    if scrape_result and scrape_result.get('content') and not scrape_result.get('error'):
                        # Additional check for error messages in content
                        content = scrape_result.get('content', '')
                        if content.startswith('[Error:') or content.startswith('Error generating summary:') or 'Error calling LLM' in content:
                            processed_url_results.append(scrape_result) # Keep result to show error in report
                            logger.warning(f"  ⚠ LLM error in content for: {url} (Error message in content)")
                            # Track failed URL in database
                            if url_db:
                                url_db.add_url(url, task_id=task_id, success=False)
                        else:
                            processed_url_results.append(scrape_result)
                            any_scrape_successful = True
                            # Log downloaded media if any
                            if scrape_result.get('downloaded_media_paths'):
                                logger.info(f"  ✓ Successfully downloaded {len(scrape_result['downloaded_media_paths'])} media files for: {url}")
                            # Log downloaded markdown if any
                            if scrape_result.get('downloaded_markdown_path'):
                                logger.info(f"  ✓ Successfully saved markdown file for: {url} at {scrape_result['downloaded_markdown_path']}")
                            # Log S3 upload if any
                            if scrape_result.get('s3_url'):
                                logger.info(f"  ✓ Successfully uploaded to S3: {scrape_result['s3_url']}")
                            else:
                                logger.info(f"  ✓ Successfully scraped: {url}")
                            
                            # Add successfully processed URL to the database
                            if url_db:
                                url_db.add_url(url, task_id=task_id, success=True)
                    elif scrape_result: # Scrape attempted, but failed or no content
                        processed_url_results.append(scrape_result) # Keep result to show error in report
                        logger.warning(f"  ⚠ Failed to get valid content for: {url} (Error: {scrape_result.get('error', 'No content')})")
                        # Track failed URL in database
                        if url_db:
                            url_db.add_url(url, task_id=task_id, success=False)
                    else: # Scraper returned None (should be rare)
                        processed_url_results.append({'url': url, 'error': 'Scraper returned None', 'content': '[Error: Scraper failed unexpectedly]'})
                        logger.error(f"  ✕ Scraper returned None for URL: {url}")
                        # Track failed URL in database
                        if url_db:
                            url_db.add_url(url, task_id=task_id, success=False)
                except Exception as e:
                    processed_url_results.append({'url': url, 'error': f'Error during scraping: {str(e)}', 'content': f'[Error: {str(e)}]'})
                    logger.error(f"  ✕ Exception while scraping URL {url}: {e}")
                    # Track failed URL in database
                    if url_db:
                        url_db.add_url(url, task_id=task_id, success=False)

        # Update stats
        if url_db:
            args.stats.skipped_urls += skipped_urls
            args.stats.processed_urls += processed_urls

        # Determine if task should be marked as failed:
        # 1. If no URLs were processed (all were skipped due to duplication) AND there were skipped URLs,
        #    consider task successful (since all content was already processed in previous runs)
        # 2. If some URLs were processed but none were successful, mark as failed
        # 3. If some URLs were processed and at least one was successful, mark as success
        if not any_url_processed and skipped_urls > 0:
            # All URLs were skipped due to previous processing - consider this a success
            mark_as_failed = False
            # Set flag to indicate we had successful processing (from previous runs)
            any_scrape_successful = True
            logger.info(f"  All URLs in task {task_id} were already processed in previous runs.")
        else:
            # Some URLs were processed in this run - base success on their results
            mark_as_failed = not any_scrape_successful

    # --- Handle Task Update/Closure ---
    final_status = 'failed' if mark_as_failed else 'success'

    # Check for LLM errors in the processed content
    has_llm_error = False
    for url_result in processed_url_results:
        if 'error' in url_result and url_result.get('error'):
            # Check if it's an LLM error in the content
            if url_result.get('content', '').startswith('Error generating summary:') or 'Error calling LLM' in str(url_result.get('error', '')):
                has_llm_error = True
                logger.error(f"  ✕ LLM processing error detected for URL: {url_result.get('url')}")
                mark_as_failed = True
                final_status = 'failed'
                break
        
    # Mark content extraction failures as failed
    if any(result.get('content', '').startswith('[Error:') for result in processed_url_results):
        mark_as_failed = True
        final_status = 'failed'

    try:
        if mark_as_failed:
            if NOT_SCRAPEABLE_LABEL not in task_labels:
                logger.info(f"  Marking task {task_id} as '{NOT_SCRAPEABLE_LABEL}'.")
                updated_labels = task_labels + [NOT_SCRAPEABLE_LABEL]
                api.update_task(task_id=task_id, labels=updated_labels)
        elif not args.no_close: # Success and closing is enabled
            logger.info(f"  Marking task {task_id} as complete.")
            api.close_task(task_id=task_id)
            final_status = 'closed'
        else: # Success but closing disabled
            logger.info(f"  Task {task_id} processed successfully, not closing (--no-close).")

    except Exception as e:
        logger.error(f"  ✕ Failed to update/close task {task_id}: {e}")
        # Don't change final_status based on update failure alone

    return {
        'status': final_status,
        'task_id': task_id,
        'original_task': { # Include original task details for the report
            'content': task_content,
            'description': task_description,
            'labels': task_labels
            },
        'processed_urls': processed_url_results
    }

def main():
    parser = argparse.ArgumentParser(description='Process Todoist tasks: scrape URLs, summarize, and report.')
    # Existing args
    parser.add_argument('--no-close', action='store_true', help='Do not mark tasks as closed after processing')
    parser.add_argument('--max-tasks', type=int, default=None, help='Maximum number of tasks to process (default: all)')
    parser.add_argument('--text-model', type=str, default='ollama/llama3.2:3b', help='Text model for summarization (e.g., ollama/llama3.2:3b, openai/gpt-3.5-turbo)') # Updated default
    parser.add_argument('--vision-model', type=str, default='ollama/llava:13b', help='Vision model (usage reduced, kept for potential future use)') # Updated default
    parser.add_argument('--screen', action='store_true', help='Display markdown output to screen instead of saving to file')

    # URL deduplication arguments
    parser.add_argument('--no-deduplication', action='store_true', 
                       help='Disable URL deduplication (URL deduplication is enabled by default)')
    parser.add_argument('--db-path', type=str, default='db/processed_urls.db', 
                       help='Path to the SQLite database for URL deduplication (default: db/processed_urls.db)')

    # Unified media handling options
    parser.add_argument('--save-media-locally', action='store_true', help='Download media (videos/images) from successfully processed Twitter/X and YouTube links')
    parser.add_argument('--output-dir', type=str, default='downloads', help='Output directory for all media downloads')
    parser.add_argument('--s3-upload', action='store_true', help='Upload media to S3 (works with or without local download)')
    parser.add_argument('--s3-bucket', type=str, default='2025-captured-notes', help='S3 bucket name for media uploads')

    # Search fallback control
    parser.add_argument('--no-search-fallback', action='store_true', help='Disable Google Search fallback for failed extractions')

    # Verbosity control
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose DEBUG logging')

    args = parser.parse_args()
    
    # Configure logging with verbosity from command line arguments
    global logger
    logger = setup_logging(verbose=args.verbose)

    logger.info("Starting Todoist processing...")
    logger.debug(f"Arguments: {args}")

    # --- Ensure Media Directory Exists (for Twitter and YouTube downloads) ---
    if args.save_media_locally:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Media download enabled. Output directory: {args.output_dir}")
        
    # --- Check S3 settings ---
    if args.s3_upload:
        try:
            import boto3
            logger.info(f"S3 upload enabled. Bucket: {args.s3_bucket}")
        except ImportError:
            logger.error("Error: boto3 not installed. Install with: pip install boto3")
            return 1

    # --- Initialize URL database ---
    url_db = None
    if not args.no_deduplication:
        try:
            url_db = URLDatabase(args.db_path)
            logger.info(f"URL deduplication enabled. Database: {args.db_path}")
            
            # Get initial stats
            stats = url_db.get_stats()
            logger.info(f"URL database contains {stats['total']} URLs ({stats['successful']} successful, {stats['failed']} failed)")
        except Exception as e:
            logger.error(f"Failed to initialize URL database: {e}")
            logger.warning("URL deduplication will be disabled")
            url_db = None
    else:
        logger.info("URL deduplication disabled with --no-deduplication flag")

    try:
        api = TodoistAPI(get_api_key())
        stats = TaskStats()
        # Add stats to args for easier access in functions
        args.stats = stats

        # --- Fetch Tasks from "Capture" Project ---
        logger.info("Fetching projects to find 'Capture' project ID...")
        projects_paginator = api.get_projects()
        
        # Debug the paginator before conversion
        logger.debug(f"Projects paginator type: {type(projects_paginator)}")
        logger.debug(f"Projects paginator: {projects_paginator}")
        
        # Convert ResultsPaginator to list
        try:
            projects_raw = list(projects_paginator)
            logger.debug(f"Successfully converted paginator to list. Length: {len(projects_raw)}")
            
            # Handle nested list structure if present
            if len(projects_raw) == 1 and isinstance(projects_raw[0], list):
                projects = projects_raw[0]  # Flatten the nested list
                logger.debug(f"Flattened nested list. New length: {len(projects)}")
            else:
                projects = projects_raw
                
        except Exception as e:
            logger.error(f"Error converting paginator to list: {e}")
            # Try alternative approach
            projects = []
            try:
                for project in projects_paginator:
                    projects.append(project)
                logger.debug(f"Alternative iteration worked. Length: {len(projects)}")
            except Exception as e2:
                logger.error(f"Alternative iteration also failed: {e2}")
                raise
        
        # Debug: inspect the structure of projects to understand the format
        logger.debug(f"Projects type: {type(projects)}")
        logger.debug(f"Projects length: {len(projects)}")
        if projects:
            logger.debug(f"First project type: {type(projects[0])}")
            logger.debug(f"First project content: {projects[0]}")
        
        # Handle different possible formats from the API
        capture_project_ids = []
        try:
            # Try the expected format (objects with .name and .id attributes)
            capture_project_ids = [p.id for p in projects if hasattr(p, 'name') and p.name.lower() == 'capture']
        except AttributeError:
            # If that fails, try dictionary format
            try:
                capture_project_ids = [p['id'] for p in projects if isinstance(p, dict) and p.get('name', '').lower() == 'capture']
            except (KeyError, TypeError):
                # If both fail, log the structure and raise a more helpful error
                logger.error(f"Unexpected projects format. Type: {type(projects)}")
                if projects:
                    logger.error(f"First project structure: {projects[0]}")
                raise ValueError("Unable to parse projects from Todoist API. Please check the API response format.")

        if not capture_project_ids:
            logger.error("Fatal: Project named 'Capture' not found in Todoist.")
            # List available projects for debugging
            try:
                if projects and hasattr(projects[0], 'name'):
                    available_projects = [p.name for p in projects]
                else:
                    available_projects = [p.get('name', 'Unknown') for p in projects if isinstance(p, dict)]
                logger.error(f"Available projects: {available_projects}")
            except (IndexError, AttributeError):
                logger.error("Could not list available projects due to format issues.")
            return 1 # Exit if essential project is missing

        capture_project_id = capture_project_ids[0]
        if len(capture_project_ids) > 1:
             logger.warning("Multiple 'Capture' projects found. Using the first one found.")

        logger.info(f"Fetching tasks from project ID: {capture_project_id}")
        # Filter tasks by project ID directly if API supports it, otherwise filter after fetching all
        # get_tasks() fetches all active tasks, filter locally.
        tasks_paginator = api.get_tasks()
        
        # Debug the paginator before conversion
        logger.debug(f"Tasks paginator type: {type(tasks_paginator)}")
        
        # Convert ResultsPaginator to list (same approach as projects)
        try:
            tasks_raw = list(tasks_paginator)
            logger.debug(f"Successfully converted tasks paginator to list. Length: {len(tasks_raw)}")
            
            # Handle nested list structure for tasks (same issue as projects)
            if len(tasks_raw) == 1 and isinstance(tasks_raw[0], list):
                all_active_tasks = tasks_raw[0]  # Flatten the nested list
                logger.debug(f"Flattened nested tasks list. Length: {len(all_active_tasks)}")
            else:
                all_active_tasks = tasks_raw
                logger.debug(f"Tasks not nested, using direct list. Length: {len(all_active_tasks)}")
                
        except Exception as e:
            logger.error(f"Error converting tasks paginator to list: {e}")
            # Try alternative approach
            all_active_tasks = []
            try:
                for task in tasks_paginator:
                    all_active_tasks.append(task)
                logger.debug(f"Alternative tasks iteration worked. Length: {len(all_active_tasks)}")
            except Exception as e2:
                logger.error(f"Alternative tasks iteration also failed: {e2}")
                raise
            
        tasks_in_capture = [t for t in all_active_tasks if t.project_id == capture_project_id]
        stats.total_tasks_considered = len(tasks_in_capture)
        logger.info(f"Found {stats.total_tasks_considered} tasks in 'Capture' project.")

        # Sort tasks (e.g., by creation date, oldest first) - requires date parsing
        try:
            tasks_in_capture.sort(key=lambda t: datetime.fromisoformat(t.created_at.replace('Z', '+00:00')))
            logger.info("Sorted tasks by creation date (oldest first).")
        except Exception as sort_e:
            logger.warning(f"Could not sort tasks by date ({sort_e}), processing in default order.")


        # Apply max_tasks limit
        tasks_to_process_list = tasks_in_capture[:args.max_tasks] if args.max_tasks else tasks_in_capture
        stats.tasks_to_process = len(tasks_to_process_list)

        if not tasks_to_process_list:
            logger.info("No tasks to process in 'Capture' project (or limit is 0).")
            # Fetch remaining tasks count for summary
            final_tasks_paginator = api.get_tasks()
            
            # Apply same flattening logic as before
            try:
                final_tasks_raw = list(final_tasks_paginator)
                if len(final_tasks_raw) == 1 and isinstance(final_tasks_raw[0], list):
                    final_all_tasks = final_tasks_raw[0]  # Flatten the nested list
                else:
                    final_all_tasks = final_tasks_raw
            except Exception as e:
                logger.warning(f"Could not fetch final task count: {e}")
                final_all_tasks = []
            
            final_tasks_in_capture = [t for t in final_all_tasks if t.project_id == capture_project_id]
            stats.tasks_remaining_in_capture = len(final_tasks_in_capture)
            stats.print_summary()
            return 0

        logger.info(f"Attempting to process {stats.tasks_to_process} tasks.")

        # --- Process Tasks ---
        processed_tasks_data = []
        for task in tasks_to_process_list:
             result = process_single_task(api, task, args, url_db)
             if result:
                  # Track stats based on result status
                  if result['status'] == 'success' or result['status'] == 'closed':
                       # Additional check: Don't count as success if there are LLM errors
                       has_llm_error = False
                       for url_data in result.get('processed_urls', []):
                           if url_data.get('content', '').startswith('Error generating summary:') or 'Error calling LLM' in str(url_data.get('error', '')):
                               logger.info(f"Re-classifying task as failed due to LLM error: {task.id}")
                               has_llm_error = True
                               stats.failed_tasks += 1
                               break
                           
                       if not has_llm_error:
                           stats.successful_tasks += 1
                           processed_tasks_data.append(result) # Only include successful tasks in report
                  elif result['status'] == 'failed':
                       stats.failed_tasks += 1
                       # Don't include failed tasks in report
                  elif result['status'] == 'skipped':
                       stats.skipped_tasks += 1
             else:
                  # Should not happen if process_single_task always returns a dict
                  logger.error(f"Processing task {task.id} returned None unexpectedly.")
                  stats.failed_tasks +=1


        # --- Generate Report ---
        if not processed_tasks_data:
            logger.info("No tasks were successfully processed in this run.")
        else:
            logger.info("Generating markdown report...")
            markdown_content = generate_markdown(processed_tasks_data)

            if args.screen:
                print("\n" + "="*80 + "\nMarkdown Report:\n" + "="*80 + "\n")
                print(markdown_content)
                print("\n" + "="*80)
                logger.info("Output displayed to screen.")
            else:
                # Determine save directory (Environment variable or default)
                save_dir_env = os.getenv('CAPTURED_NOTES_FOLDER')
                if save_dir_env and Path(save_dir_env).is_dir():
                     save_dir = Path(save_dir_env)
                else:
                     if save_dir_env: logger.warning(f"CAPTURED_NOTES_FOLDER ('{save_dir_env}') not found or invalid, using script directory.")
                     save_dir = Path(__file__).parent # Default to script's directory

                save_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists

                filename = f"{int(time.time())}-capture-report.md"
                file_path = save_dir / filename

                try:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(markdown_content)
                    logger.info(f"✓ Report saved successfully to: {file_path}")
                except IOError as e:
                    logger.error(f"✕ Failed to save report to {file_path}: {e}")
                    # Print to screen as fallback?
                    print("\n" + "="*80 + "\nError saving file! Displaying report on screen:\n" + "="*80 + "\n")
                    print(markdown_content)
                    # print("\n" + "="*80)


        # --- Final Summary ---
        # Fetch final count of tasks remaining in Capture
        final_tasks_paginator = api.get_tasks()
        
        # Apply same flattening logic as before
        try:
            final_tasks_raw = list(final_tasks_paginator)
            if len(final_tasks_raw) == 1 and isinstance(final_tasks_raw[0], list):
                final_all_tasks = final_tasks_raw[0]  # Flatten the nested list
            else:
                final_all_tasks = final_tasks_raw
        except Exception as e:
            logger.warning(f"Could not fetch final task count: {e}")
            final_all_tasks = []
            
        final_tasks_in_capture = [t for t in final_all_tasks if t.project_id == capture_project_id]
        stats.tasks_remaining_in_capture = len(final_tasks_in_capture)
        stats.print_summary()
        
        # Display URL database statistics if enabled
        if url_db:
            db_stats = url_db.get_stats()
            print("\n=== URL Database Statistics ===")
            print(f"Total URLs tracked: {db_stats['total']}")
            print(f"Successfully processed URLs: {db_stats['successful']}")
            print(f"Failed URLs: {db_stats['failed']}")
            print("==============================")
            
            # Close database connection
            url_db.close()
            
        logger.info("Processing finished.")
        return 0

    except ValueError as e: # Catch specific errors like missing API key
         logger.error(f"Configuration error: {e}")
         return 1
    except Exception as e:
        logger.error(f"An unexpected fatal error occurred: {e}", exc_info=True) # Log traceback for fatal errors
        # Attempt to print stats even on fatal error if stats object exists
        try: stats.print_summary()
        except NameError: pass
        return 1


if __name__ == "__main__":
    sys.exit(main())