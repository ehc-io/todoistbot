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

# --- Import the refactored scraper ---
from url_scraper import URLScraper
# --- Import the YouTube extractor directly ---
from yt_extractor import YouTubeDataFetcher, get_transcript_summary, enhance_youtube_processing
# ---

import logging # Ensure logging is used

# Configure logging for main script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("main")

# Create a filter to ignore TeX math warnings
class TexMathFilter(logging.Filter):
    def filter(self, record):
        # Skip any log messages about TeX math conversion
        if "Could not convert TeX math" in record.getMessage():
            return False
        if "rendering as TeX" in record.getMessage():
            return False
        return True

# Apply the filter to all loggers
for logger_name in logging.Logger.manager.loggerDict:
    logging_instance = logging.getLogger(logger_name)
    logging_instance.addFilter(TexMathFilter())

# Also apply filter to the root logger
logging.getLogger().addFilter(TexMathFilter())

class TaskStats:
    def __init__(self):
        self.total_tasks_considered = 0 # Tasks initially fetched
        self.tasks_to_process = 0       # Tasks attempting processing (after label/limit filter)
        self.successful_tasks = 0       # Tasks processed yielding content
        self.failed_tasks = 0           # Tasks attempted but failed (error or no content)
        self.skipped_tasks = 0          # Tasks skipped due to 'not-scrapeable' label
        self.tasks_remaining_in_capture = 0 # Tasks left in Capture project after run

    def print_summary(self):
        print("\n=== Task Processing Summary ===")
        print(f"Tasks initially found in 'Capture': {self.total_tasks_considered}")
        print(f"Tasks attempted processing:        {self.tasks_to_process}")
        print(f"Tasks skipped ('not-scrapeable'):  {self.skipped_tasks}")
        print(f"Successfully processed (got content): {self.successful_tasks}")
        print(f"Failed processing (error/no content): {self.failed_tasks}")
        print(f"Tasks remaining in 'Capture':      {self.tasks_remaining_in_capture}")
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

def get_api_key() -> str:
    api_key = os.getenv('TODOIST_API_KEY')
    if not api_key:
        raise ValueError("TODOIST_API_KEY environment variable not set")
    return api_key

def generate_markdown(tasks_data: List[dict]) -> str:
    """
    Generate formatted markdown content from processed tasks data.
    """
    content = [
        "# Todoist Capture Report",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "---",
        "",
    ]

    # Filter out None values or tasks that resulted in errors without content
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
            
            # Add downloaded video information if available
            downloaded_video = url_data.get('downloaded_video_path', '')
            downloaded_markdown = url_data.get('downloaded_markdown_path', '')
            s3_url = url_data.get('s3_url', '')

            content.append(f"**URL**: [{url}]({url})  ")
            content.append(f"**Type**: {content_type.capitalize()}  ")
            if extraction_method: 
                content.append(f"*Extraction Method: {extraction_method}*  ")
                content.append("  ")  # Spacer after each URL's content
                # content.append("---")  # Separator for each URL
            
            # Add downloaded video info if available
            if downloaded_video:
                content.append(f"**Local Folder**: {downloaded_video}  ")
                
            # Add downloaded markdown info if available
            if downloaded_markdown:
                content.append(f"**Markdown File**: {downloaded_markdown}  ")
            
            # Add S3 URL if available
            if s3_url:
                # Extract week folder from s3_url (format: week_XX)
                week_folder = "Unknown Week"
                match = re.search(r'week_(\d+)', s3_url)
                if match:
                    week_folder = f"Week {match.group(1)}"
                
                content.append(f"**S3 Location**: [{s3_url}]({s3_url})  ")
                # content.append(f"**S3 Folder Structure**: {week_folder}  ")
                
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

def process_single_task(api: TodoistAPI, task, args) -> Optional[dict]:
    """
    Processes a single Todoist task: extracts URLs, scrapes them, and returns results.
    Handles adding 'not-scrapeable' label and optionally closing the task.
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
        
        # Track already processed YouTube video IDs to avoid duplicates
        processed_youtube_ids = set()

        for url in all_urls:
            logger.debug(f"  Scraping URL: {url}")
            
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
                else:
                    logger.warning(f"  ⚠ Failed to process YouTube URL: {url} (Error: {youtube_result.get('error', 'Unknown error')})")
                
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
                    elif scrape_result: # Scrape attempted, but failed or no content
                        processed_url_results.append(scrape_result) # Keep result to show error in report
                        logger.warning(f"  ⚠ Failed to get valid content for: {url} (Error: {scrape_result.get('error', 'No content')})")
                    else: # Scraper returned None (should be rare)
                        processed_url_results.append({'url': url, 'error': 'Scraper returned None', 'content': '[Error: Scraper failed unexpectedly]'})
                        logger.error(f"  ✕ Scraper returned None for URL: {url}")
                except Exception as e:
                    processed_url_results.append({'url': url, 'error': f'Error during scraping: {str(e)}', 'content': f'[Error: {str(e)}]'})
                    logger.error(f"  ✕ Exception while scraping URL {url}: {e}")

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

    # --- Configure Logging Level ---
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.getLogger().setLevel(log_level) # Set root logger level
    # Adjust levels for noisy libraries if needed
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("pypandoc").setLevel(logging.INFO)
    logging.getLogger("playwright").setLevel(logging.INFO) # Playwright can be very verbose on DEBUG

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

    try:
        api = TodoistAPI(get_api_key())
        stats = TaskStats()

        # --- Fetch Tasks from "Capture" Project ---
        logger.info("Fetching projects to find 'Capture' project ID...")
        projects = api.get_projects()
        capture_project_ids = [p.id for p in projects if p.name.lower() == 'capture']

        if not capture_project_ids:
            logger.error("Fatal: Project named 'Capture' not found in Todoist.")
            return 1 # Exit if essential project is missing

        capture_project_id = capture_project_ids[0]
        if len(capture_project_ids) > 1:
             logger.warning("Multiple 'Capture' projects found. Using the first one found.")

        logger.info(f"Fetching tasks from project ID: {capture_project_id}")
        # Filter tasks by project ID directly if API supports it, otherwise filter after fetching all
        # get_tasks() fetches all active tasks, filter locally.
        all_active_tasks = api.get_tasks()
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
            final_tasks_in_capture = [t for t in api.get_tasks() if t.project_id == capture_project_id]
            stats.tasks_remaining_in_capture = len(final_tasks_in_capture)
            stats.print_summary()
            return 0

        logger.info(f"Attempting to process {stats.tasks_to_process} tasks.")

        # --- Process Tasks ---
        processed_tasks_data = []
        for task in tasks_to_process_list:
             result = process_single_task(api, task, args)
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
        final_tasks_in_capture = [t for t in api.get_tasks() if t.project_id == capture_project_id]
        stats.tasks_remaining_in_capture = len(final_tasks_in_capture)
        stats.print_summary()
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