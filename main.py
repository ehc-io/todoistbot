#!/usr/bin/env python3
import os
import argparse
import time
import re
import requests
from typing import Optional, List, Dict, Any
from todoist_api_python.api import TodoistAPI
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
from urllib.parse import urlparse, unquote

def get_api_key() -> str:
    api_key = os.getenv('TODOIST_API_KEY')
    if not api_key:
        raise ValueError("TODOIST_API_KEY environment variable not set")
    return api_key

def clean_url(url: str) -> str:
    """Clean and normalize URL."""
    # Remove trailing parentheses and other common artifacts
    url = re.sub(r'[)\]]$', '', url)
    # Remove hash fragments unless they're meaningful
    if '#' in url and not any(x in url for x in ['#page=', '#section=']):
        url = url.split('#')[0]
    return url.strip()

def is_valid_url(url: str) -> bool:
    """Validate URL and check if it's scrapeable."""
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            return False
        # Skip PDFs and other problematic formats
        if url.lower().endswith(('.pdf', '.jpg', '.png', '.gif')):
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Skipping non-HTML URL: {url}")
            return False
        return True
    except:
        return False

def scrape_url(url: str, selectors: Dict[str, str] = None, wait_for: str = None) -> Optional[Dict[str, Any]]:
    """Scrape content from URL with improved error handling."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] → Starting to scrape URL: {url}")
    
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                viewport={'width': 1280, 'height': 800},
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            )
            page = context.new_page()
            
            try:
                page.goto(url, wait_until='domcontentloaded', timeout=30000)
            except PlaywrightTimeout:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Timeout loading page: {url}")
                return None
                
            result = {}
            
            try:
                if selectors:
                    for name, selector in selectors.items():
                        elements = page.query_selector_all(selector)
                        texts = []
                        for el in elements:
                            text = el.text_content()
                            if text and text.strip():
                                texts.append(text.strip())
                        result[name] = texts
                else:
                    # First try to get main content
                    main_content = page.evaluate("""() => {
                        const selectors = [
                            'main article',
                            'main',
                            'article',
                            '[role="main"]',
                            '.content',
                            '.main',
                            '#content',
                            '#main',
                            'body'
                        ];
                        
                        for (const selector of selectors) {
                            const element = document.querySelector(selector);
                            if (element) {
                                return element.textContent
                                    .replace(/\\s+/g, ' ')
                                    .trim();
                            }
                        }
                        return '';
                    }""")
                    
                    if main_content:
                        result['content'] = main_content
                    
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error extracting content: {str(e)}")
                return None
            
            browser.close()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Successfully scraped URL: {url}")
            return result
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error scraping {url}:")
        print(f"├── Error type: {type(e).__name__}")
        print(f"└── Details: {str(e)}")
        return None

def extract_urls(text: str) -> List[str]:
    """Extract and clean URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return [clean_url(url) for url in urls]

def process_task(api: TodoistAPI, task, mark_closed: bool) -> dict:
    """Process a single task and its URLs."""
    task_data = {
        'content': task.content,
        'description': task.description,
        'labels': task.labels,
        'urls_content': []
    }
    
    # Extract URLs from both content and description
    urls = extract_urls(task.content)
    if task.description:
        urls.extend(extract_urls(task.description))
    
    # Remove duplicates and validate URLs
    urls = list(set(url for url in urls if is_valid_url(url)))
    
    # Process each valid URL
    for url in urls:
        scraped_content = scrape_url(url)
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
                        # Truncate long content and add ellipsis
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