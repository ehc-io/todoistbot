#!/usr/bin/env python3
import re
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse
from datetime import datetime
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout

class URLScraper:
    @staticmethod
    def clean_url(url: str) -> str:
        """Clean and normalize URL."""
        # Remove trailing parentheses and other common artifacts
        url = re.sub(r'[)\]]$', '', url)
        # Remove hash fragments unless they're meaningful
        if '#' in url and not any(x in url for x in ['#page=', '#section=']):
            url = url.split('#')[0]
        return url.strip()

    @staticmethod
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

    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract and clean URLs from text."""
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls = re.findall(url_pattern, text)
        return [URLScraper.clean_url(url) for url in urls]

    @staticmethod
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