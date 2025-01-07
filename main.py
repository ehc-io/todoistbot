#!/usr/bin/env python3
import os
import re
import argparse
import time
import json
from typing import List, Dict, Set
from todoist_api_python.api import TodoistAPI
from datetime import datetime
from url_scraper import URLScraper
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
import io

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

def upload_to_drive(content: str, filename: str) -> str:
    """Upload content to Google Drive using service account."""
    # Use broader scope for testing
    SCOPES = ['https://www.googleapis.com/auth/drive']
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    credentials_path = os.environ.get("GOOGLE_DRIVE_KEY")

    if not folder_id:
        raise ValueError("GOOGLE_DRIVE_FOLDER_ID environment variable not found.")
    if not credentials_path:
        raise ValueError("GOOGLE_DRIVE_KEY environment variable not found.")
    
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=SCOPES)
    
    service = build('drive', 'v3', credentials=credentials)

    try:
        file_metadata = {
            'name': filename,
            'parents': [folder_id]
        }
        
        content_bytes = io.BytesIO(content.encode('utf-8'))
        media = MediaIoBaseUpload(
            content_bytes,
            mimetype='text/plain',
            resumable=True
        )
        
        file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, webViewLink',
            supportsAllDrives=True  # Add support for shared drives
        ).execute()
        
        return file.get('webViewLink')
        
    except Exception as e:
        print(f"\nError uploading to Drive: {str(e)}")
        raise

def generate_markdown(tasks_data: List[dict]) -> str:
    """Generate markdown content from tasks data."""
    content = "# Todoist Tasks Report\n\n"
    content += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # Filter out None values (tasks with no content)
    valid_tasks = [task for task in tasks_data if task is not None]
    
    for task_data in valid_tasks:
        content += f"## {task_data['content']}\n\n"
        
        if task_data['description']:
            content += f"Description: {task_data['description']}\n\n"
        
        if task_data['labels']:
            content += "Labels: " + ", ".join(task_data['labels']) + "\n\n"
        
        if task_data['urls_content']:
            for url_data in task_data['urls_content']:
                content_text = url_data['content']
                if content_text:
                    content_preview = content_text[:2000] + ('...' if len(content_text) > 2000 else '')
                    content += f"```\n{content_preview}\n```\n\n"
        
        content += "---\n\n"
    
    return content

def get_bypassed_project_ids(api: TodoistAPI, bypass_projects: str = None) -> Set[str]:
    """
    Get the project IDs that should be bypassed during processing.
    By default, returns the Inbox project ID.
    If bypass_projects is provided, returns the IDs of specified projects.
    
    Args:
        api: TodoistAPI instance
        bypass_projects: Comma-separated string of project names to bypass
    
    Returns:
        Set of project IDs to bypass
    """
    try:
        projects = api.get_projects()
        
        # If no specific projects are provided, bypass Inbox by default
        if not bypass_projects:
            return {p.id for p in projects if p.is_inbox_project}
            
        # Convert project names to lowercase for case-insensitive matching
        bypass_names = {name.strip().lower() for name in bypass_projects.split(',')}
        bypass_ids = set()
        
        for project in projects:
            if project.name.lower() in bypass_names:
                bypass_ids.add(project.id)
                bypass_names.remove(project.name.lower())
        
        # Warn about any project names that weren't found
        if bypass_names:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Warning: Projects not found: {', '.join(bypass_names)}")
            
        return bypass_ids
        
    except Exception as error:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ⚠ Error getting projects: {str(error)}")
        return set()
    
def process_task(api: TodoistAPI, task, mark_closed: bool) -> dict:
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
            scraped_content = URLScraper.scrape_url(url)
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
    parser.add_argument('--bypass-projects', help='Comma-separated list of project names to bypass. If not specified, bypasses Inbox project by default')
    args = parser.parse_args()

    api = TodoistAPI(get_api_key())
    stats = TaskStats()
    
    try:
        bypass_project_ids = get_bypassed_project_ids(api, args.bypass_projects)
        
        all_tasks = api.get_tasks()
        # Get total remaining tasks before filtering
        stats.remaining_tasks = len([t for t in all_tasks if t.project_id not in bypass_project_ids])
        
        # Filter out tasks from bypass projects and tasks with not-scrapeable label
        tasks = [t for t in all_tasks if t.project_id not in bypass_project_ids]
        
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
            stats.remaining_tasks = max(0, stats.remaining_tasks - args.max_tasks)

        if not tasks:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ℹ No tasks to process after filtering")
            stats.print_summary()
            return

        tasks_data = []
        for task in tasks:
            stats.total_tasks += 1
            print(f"[{datetime.now().strftime('%H:%M:%S')}] → Processing task: {task.content}")
            task_data = process_task(api, task, not args.no_close)
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
    
        # Upload to Google Drive
        filename = f"{int(time.time())}-captured-notes.md"
        try:
            drive_link = upload_to_drive(markdown_content, filename)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Report uploaded to Google Drive: {drive_link}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Failed to upload to Google Drive: {str(e)}")
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