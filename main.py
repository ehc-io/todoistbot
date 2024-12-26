#!/usr/bin/env python3
import os
import argparse
import time
from typing import List, Dict
from todoist_api_python.api import TodoistAPI
from datetime import datetime
from url_scraper import URLScraper

def get_api_key() -> str:
    api_key = os.getenv('TODOIST_API_KEY')
    if not api_key:
        raise ValueError("TODOIST_API_KEY environment variable not set")
    return api_key

def process_task(api: TodoistAPI, task, mark_closed: bool) -> dict:
    """Process a single task and its URLs."""
    task_data = {
        'content': task.content,
        'description': task.description,
        'labels': task.labels,
        'urls_content': []
    }
    
    # Extract URLs from both content and description using static methods
    urls = URLScraper.extract_urls(task.content)
    if task.description:
        urls.extend(URLScraper.extract_urls(task.description))
    
    # Remove duplicates and validate URLs
    urls = list(set(url for url in urls if URLScraper.is_valid_url(url)))
    
    # Process each valid URL
    for url in urls:
        scraped_content = URLScraper.scrape_url(url)
        if scraped_content:
            task_data['urls_content'].append({
                'url': url,
                'content': scraped_content
            })
    
    if urls and mark_closed and task_data['urls_content']:
        try:
            api.close_task(task_id=task.id)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Task {task.id} marked as completed")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Failed to close task {task.id}: {str(e)}")
    
    return task_data

def write_markdown(tasks_data: List[dict], output_file: str):
    """Write tasks data to markdown file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Todoist Tasks Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for task_data in tasks_data:
            f.write(f"## {task_data['content']}\n\n")
            
            if task_data['description']:
                f.write(f"Description: {task_data['description']}\n\n")
            
            if task_data['labels']:
                f.write("Labels: " + ", ".join(task_data['labels']) + "\n\n")
            
            if task_data['urls_content']:
                f.write("### URL Contents\n\n")
                for url_data in task_data['urls_content']:
                    f.write(f"#### {url_data['url']}\n\n")
                    content = url_data['content'].get('content', '')
                    if content:
                        content_preview = content[:2000] + ('...' if len(content) > 2000 else '')
                        f.write(f"```\n{content_preview}\n```\n\n")
            
            f.write("---\n\n")

def main():
    parser = argparse.ArgumentParser(description='Process Todoist tasks and extract URL contents')
    parser.add_argument('--project-id', help='Specific project ID to process')
    parser.add_argument('--no-close', action='store_true', help='Do not mark tasks as closed after processing')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--max-tasks', type=int, help='Maximum number of tasks to process')
    
    args = parser.parse_args()
    
    api = TodoistAPI(get_api_key())
    
    try:
        tasks = api.get_tasks()
        
        if args.project_id:
            tasks = [t for t in tasks if t.project_id == args.project_id]
        
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
        
        tasks_data = []
        for task in tasks:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] → Processing task: {task.content}")
            task_data = process_task(api, task, not args.no_close)
            tasks_data.append(task_data)
        
        output_file = args.output if args.output else f"{int(time.time())}.md"
        write_markdown(tasks_data, output_file)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Report written to: {output_file}")
        
    except Exception as error:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ✕ Fatal error: {str(error)}")
        exit(1)

if __name__ == "__main__":
    main()