#!/usr/bin/env python3

import os
import argparse
import time
import requests
from typing import Optional, List
from todoist_api_python.api import TodoistAPI
from datetime import datetime

def get_api_key() -> str:
    """Get Todoist API key from environment variables."""
    api_key = os.getenv('TODOIST_API_KEY')
    if not api_key:
        raise ValueError("TODOIST_API_KEY environment variable not set")
    return api_key

def scrape_url(url: str) -> Optional[str]:
    """
    Scrape content from a URL.
    This is a basic implementation that can be replaced with a more sophisticated one.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except Exception as e:
        print(f"Error scraping URL {url}: {str(e)}")
        return None

def process_task(api: TodoistAPI, task, mark_closed: bool) -> dict:
    """Process a single task and return its data."""
    task_data = {
        'content': task.content,
        'description': task.description,
        'labels': task.labels,
        'url_content': None
    }
    
    if task.url:
        print(f"Scraping content from URL: {task.url}")
        task_data['url_content'] = scrape_url(task.url)
        
        if task_data['url_content'] and mark_closed:
            try:
                api.close_task(task_id=task.id)
                print(f"Task {task.id} marked as completed")
            except Exception as e:
                print(f"Error closing task {task.id}: {str(e)}")
    
    return task_data

def write_markdown(tasks_data: List[dict], output_file: str):
    """Write tasks data to a markdown file."""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# Todoist Tasks Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for task_data in tasks_data:
            f.write(f"## {task_data['content']}\n\n")
            
            if task_data['description']:
                f.write(f"Description: {task_data['description']}\n\n")
            
            if task_data['labels']:
                f.write("Labels: " + ", ".join(task_data['labels']) + "\n\n")
            
            if task_data['url_content']:
                f.write("### URL Content\n\n")
                f.write(f"```\n{task_data['url_content'][:1000]}...\n```\n\n")
            
            f.write("---\n\n")

def main():
    parser = argparse.ArgumentParser(description='Process Todoist tasks and extract URL contents')
    parser.add_argument('--project-id', help='Specific project ID to process')
    parser.add_argument('--no-close', action='store_true', help='Do not mark tasks as closed after processing')
    parser.add_argument('--output', help='Output file path')
    parser.add_argument('--max-tasks', type=int, help='Maximum number of tasks to process')
    
    args = parser.parse_args()
    
    # Initialize API
    api = TodoistAPI(get_api_key())
    
    try:
        # Get tasks
        tasks = api.get_tasks()
        
        # Filter by project if specified
        if args.project_id:
            tasks = [t for t in tasks if t.project_id == args.project_id]
        
        # Limit number of tasks if specified
        if args.max_tasks:
            tasks = tasks[:args.max_tasks]
        
        # Process tasks
        tasks_data = []
        for task in tasks:
            print(f"Processing task: {task.content}")
            task_data = process_task(api, task, not args.no_close)
            tasks_data.append(task_data)
        
        # Determine output file
        output_file = args.output if args.output else f"{int(time.time())}.md"
        
        # Write results
        write_markdown(tasks_data, output_file)
        print(f"Report written to: {output_file}")
        
    except Exception as error:
        print(f"Error: {str(error)}")
        exit(1)

if __name__ == "__main__":
    main()