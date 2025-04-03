#!/usr/bin/env python3
import os
import re
import argparse
import time
from typing import List
from todoist_api_python.api import TodoistAPI
from datetime import datetime
from url_scraper import URLScraper

class TaskStats:
    def __init__(self):
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.remaining_tasks = 0

    def print_summary(self):
        print("\n=== Task Processing Summary ===")
        print(f"Total tasks processed: {self.total_tasks}")
        print(f"Successfully processed: {self.successful_tasks}")
        print(f"Failed to process: {self.failed_tasks}")
        print(f"Tasks remaining: {self.remaining_tasks}")
        print("============================")
        
class URLExtractor:
    """Helper class to extract and clean URLs from text content."""
    
    @staticmethod
    def extract_markdown_urls(text: str) -> list[str]:
        """
        Extract URLs from text, handling both plain URLs and Markdown-formatted links.
        Returns a list of clean URLs.
        """
        if not text:
            return []
        
        # Pattern for Markdown links [text](url)
        markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
        # Pattern for raw URLs
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        
        urls = []
        
        # Extract URLs from Markdown links
        for match in re.finditer(markdown_pattern, text):
            urls.append(match.group(2))
        
        # Extract raw URLs that aren't part of Markdown links
        # First, remove Markdown links to avoid double-counting
        text_without_markdown = re.sub(markdown_pattern, '', text)
        urls.extend(re.findall(url_pattern, text_without_markdown))
        
        # Clean and normalize URLs
        cleaned_urls = []
        for url in urls:
            # Remove any trailing punctuation or brackets
            url = re.sub(r'[\]\)]+$', '', url)
            # Remove any leading brackets
            url = re.sub(r'^[\[\(]+', '', url)
            cleaned_urls.append(url)
        
        return list(set(cleaned_urls))  # Remove duplicates

def get_api_key() -> str:
    api_key = os.getenv('TODOIST_API_KEY')
    if not api_key:
        raise ValueError("TODOIST_API_KEY environment variable not set")
    return api_key

def clean_task_name(content: str) -> str:
    """Convert task content to a clean name format."""
    # Remove special characters and replace spaces with underscores
    clean = re.sub(r'[^\w\s-]', '', content)
    return clean.strip().replace(' ', '_').lower()[:30]

def format_section(title: str, content: str | None) -> str:
    """Format a section with title if content exists."""
    if not content:
        return ""
    return f"### {title}\n{content}\n\n"

def generate_markdown(tasks_data: List[dict]) -> str:
    """
    Generate formatted markdown content from tasks data with clear separation and task names.
    
    Args:
        tasks_data: List of task dictionaries containing task information
        
    Returns:
        Formatted markdown string
    """
    # Header section
    content = [
        "# Todoist Tasks Report",
        "",
        f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "--"
        "",
    ]
    
    # Filter out None values
    valid_tasks = [task for task in tasks_data if task is not None]
    
    # Generate task sections
    for i, task in enumerate(valid_tasks):
        # Use consistent anchor IDs that match the TOC
        anchor_id = f"task-{i+1}"
        
        # Task header with anchor
        content.extend([
            f"## {task['content']}",
            ""
        ])
        
        # Description section
        if task['description']:
            content.extend([
                "### Description",
                f"{task['description']}",
                ""
            ])
        
        # Labels section
        if task['labels']:
            content.extend([
                "### Labels",
                ", ".join([f"`{label}`" for label in task['labels']]),
                ""
            ])
        
        # URL content section - directly include content without headers
        if task['urls_content']:
            for url_data in task['urls_content']:
                if url_data['content']:
                    content_preview = url_data['content'][:2000]
                    if len(url_data['content']) > 2000:
                        content_preview += '...'
                    
                    content.extend([
                        content_preview,
                        "",
                        ""
                    ])
        
        # Task separator
        content.extend([
            "<div style='page-break-after: always;'></div>",
            "---",
            ""
        ])
    
    return "\n".join(content)
    
def _process_task(api: TodoistAPI, task, mark_closed: bool, text_model: str, vision_model: str) -> dict:
    """
    Process a single task and its URLs.
    Returns None if task has not-scrapeable label or if no valid content was scraped.
    In case of failure, adds 'not-scrapeable' label to the task.
    """
    NOT_SCRAPEABLE_LABEL = "not-scrapeable"
    
    # Skip processing if task already has not-scrapeable label
    if NOT_SCRAPEABLE_LABEL in task.labels:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Skipping task {task.id}: has not-scrapeable label")
        return None
    
    task_data = {
        'content': task.content,
        'description': task.description,
        'labels': task.labels,
        'urls_content': []
    }

    urls = URLExtractor.extract_markdown_urls(task.content)
    if task.description:
        urls.extend(URLExtractor.extract_markdown_urls(task.description))

    urls = list(dict.fromkeys(url for url in urls if URLScraper.is_valid_url(url)))
    
    successful_scrapes = False
    for url in urls:
        try:
            scraped_content = URLScraper.scrape_url(url, text_model=text_model, vision_model=vision_model)
            if scraped_content and scraped_content.get('content'):
                task_data['urls_content'].append({
                    'url': url,
                    'content': scraped_content['content']
                })
                successful_scrapes = True
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ No content retrieved from URL: {url}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error scraping URL {url}: {str(e)}")

    if not successful_scrapes:
        try:
            # Add not-scrapeable label to the task
            updated_labels = list(task.labels)
            if NOT_SCRAPEABLE_LABEL not in updated_labels:
                updated_labels.append(NOT_SCRAPEABLE_LABEL)
                api.update_task(task_id=task.id, labels=updated_labels)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ↪ Task {task.id} marked as not-scrapeable")
            return None
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Failed to update task {task.id} with not-scrapeable label: {str(e)}")
            return None
    
    # Only close the task if we successfully scraped content and marking as closed is requested
    if successful_scrapes and mark_closed:
        try:
            api.close_task(task_id=task.id)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Task {task.id} marked as completed")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Failed to close task {task.id}: {str(e)}")

    return task_data

def process_task(api: TodoistAPI, task, mark_closed: bool, text_model: str, vision_model: str) -> dict:
    """
    Process a single task and its URLs.
    Returns None if task has not-scrapeable label or if no valid content was scraped.
    In case of failure, adds 'not-scrapeable' label to the task.
    """
    NOT_SCRAPEABLE_LABEL = "not-scrapeable"
    
    # Skip processing if task already has not-scrapeable label
    if NOT_SCRAPEABLE_LABEL in task.labels:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Skipping task {task.id}: has not-scrapeable label")
        return None
    
    task_data = {
        'content': task.content,
        'description': task.description,
        'labels': task.labels,
        'urls_content': []
    }

    urls = URLExtractor.extract_markdown_urls(task.content)
    if task.description:
        urls.extend(URLExtractor.extract_markdown_urls(task.description))

    urls = list(dict.fromkeys(url for url in urls if URLScraper.is_valid_url(url)))
    
    successful_scrapes = False
    for url in urls:
        try:
            # Pass both text_model and vision_model to scrape_url
            scraped_content = URLScraper.scrape_url(
                url, 
                text_model=text_model, 
                vision_model=vision_model
            )
            
            if scraped_content and scraped_content.get('content'):
                task_data['urls_content'].append({
                    'url': url,
                    'content': scraped_content['content']
                })
                successful_scrapes = True
            else:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ No content retrieved from URL: {url}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error scraping URL {url}: {str(e)}")

    if not successful_scrapes:
        try:
            # Add not-scrapeable label to the task
            updated_labels = list(task.labels)
            if NOT_SCRAPEABLE_LABEL not in updated_labels:
                updated_labels.append(NOT_SCRAPEABLE_LABEL)
                api.update_task(task_id=task.id, labels=updated_labels)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ↪ Task {task.id} marked as not-scrapeable")
            return None
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Failed to update task {task.id} with not-scrapeable label: {str(e)}")
            return None
    
    # Only close the task if we successfully scraped content and marking as closed is requested
    if successful_scrapes and mark_closed:
        try:
            api.close_task(task_id=task.id)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Task {task.id} marked as completed")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Failed to close task {task.id}: {str(e)}")

    return task_data


def main():
    parser = argparse.ArgumentParser(description='Process Todoist tasks and extract URL contents')
    parser.add_argument('--no-close', action='store_true', help='Do not mark tasks as closed after processing')
    parser.add_argument('--max-tasks', type=int, help='Maximum number of tasks to process')
    parser.add_argument('--text-model', type=str, default='ollama/llama3.2:3b', 
                       help='Text model to use for content summarization (default: llama3.2:3b)')
    parser.add_argument('--vision-model', type=str, default='ollama/llava:7b',
                       help='Vision model to use for image analysis (default: ollama/llava)')
    parser.add_argument('--screen', action='store_true', help='Display output to screen instead of saving to file')
    args = parser.parse_args()

    api = TodoistAPI(get_api_key())
    stats = TaskStats()
    
    try:
        all_tasks = api.get_tasks()
        
        # Get all projects to find the Capture project ID
        projects = api.get_projects()
        capture_project_ids = [p.id for p in projects if p.name.lower() == 'capture']
        
        if not capture_project_ids:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Warning: No project named 'Capture' found")
            stats.print_summary()
            return
            
        # Filter to only include tasks in the Capture project
        tasks = [t for t in all_tasks if t.project_id in capture_project_ids]
        
        # Get total remaining tasks after filtering for the capture tag
        stats.remaining_tasks = len(tasks)
        
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
            stats.remaining_tasks = max(0, stats.remaining_tasks - args.max_tasks)

        if not tasks:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ No tasks with #capture tag found")
            stats.print_summary()
            return

        tasks_data = []
        for task in tasks:
            stats.total_tasks += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] → Processing task: {task.content}")
            task_data = process_task(api, task, not args.no_close, args.text_model, args.vision_model)
            if task_data:  # Only append if we got valid content
                tasks_data.append(task_data)
                stats.successful_tasks += 1
            else:
                stats.failed_tasks += 1

        if not tasks_data:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ No tasks contained valid content after processing")
            stats.print_summary()
            return

        # Generate markdown content
        markdown_content = generate_markdown(tasks_data)
    
        # If --screen option is used, print to screen instead of saving to file
        if args.screen:
            print("\n" + "="*80 + "\n")
            print(markdown_content)
            print("\n" + "="*80)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Output displayed to screen")
        else:
            # Save to local folder
            filename = f"{int(time.time())}-captured-notes.md"
            
            # Get save directory from environment variable or use current directory
            import os
            save_dir = os.environ.get('CAPTURED_NOTES_FOLDER', os.path.dirname(os.path.abspath(__file__)))
            
            # Create directory if it doesn't exist
            try:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                    print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ Created directory: {save_dir}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Could not create directory {save_dir}: {str(e)}")
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ Using current directory instead")
                save_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Save the file
            file_path = os.path.join(save_dir, filename)
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Report saved to: {file_path}")
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Failed to save file: {str(e)}")
                stats.print_summary()
                exit(1)

        # Print final statistics
        stats.print_summary()

    except Exception as error:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Fatal error: {str(error)}")
        stats.print_summary()
        exit(1)

if __name__ == "__main__":
    main()
