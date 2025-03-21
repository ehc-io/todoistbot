#!/usr/bin/env python3
"""
Twitter Scraper - A tool to extract content from Twitter/X.com

This script loads a tweet URL using Playwright and extracts various types of data.
It supports loading Twitter session information for authenticated access.
"""

import os
import json
import argparse
from datetime import datetime
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright


def setup_argparse():
    """Set up command-line argument parsing."""
    parser = argparse.ArgumentParser(description="Extract content from Twitter/X.com tweets")
    parser.add_argument("url", help="URL of the tweet to scrape")
    parser.add_argument(
        "--session-file", 
        default="session-data/session.json", 
        help="Path to the Twitter session data file (default: session-data/session.json)"
    )
    return parser.parse_args()

def extract_tweet_images(page):
    """
    Extract tweet images by finding anchor tags with '/photo/' in their href.
    
    Args:
        page: The Playwright page object with loaded tweet
    
    Returns:
        list: List of image URLs
    """
    image_urls = []
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Looking for images using photo links")
    
    try:
        # Use JavaScript to extract photo links from the tweet
        photo_links = page.evaluate('''() => {
            // Get the tweet article element
            const tweet = document.querySelector('article[data-testid="tweet"]');
            
            if (!tweet) return [];
            
            // Find all anchor tags that include '/photo/' in their href attribute
            const photoLinks = tweet.querySelectorAll('a[href*="/photo/"]');
            
            // Map the results to extract the href values
            const imageUrls = Array.from(photoLinks).map(link => link.href);
            
            // Remove duplicates
            const uniqueImageUrls = [...new Set(imageUrls)];
            
            return uniqueImageUrls;
        }''')
        
        for url in photo_links:
            if url and url not in image_urls:
                # Clean URL by removing the HTML entity encodings if any
                clean_url = url.replace('&amp;', '&')
                image_urls.append(clean_url)
                
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Found {len(image_urls)} image links via photo links")
            
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error extracting photo links: {str(e)}")
    
    return image_urls

def extract_tweet_gifs(page):
    """
    Extracts GIFs from a tweet using Playwright by finding <video> elements
    labeled as GIF content (e.g. video[aria-label^='GIF']).
    
    Args:
        page (playwright.sync_api.Page): The Playwright page instance.

    Returns:
        list: A list of GIF URLs extracted from the tweet.
    """
    gif_selectors = "video[aria-label^='GIF']"
    gif_elements = page.query_selector_all(gif_selectors)
    gif_urls = []

    for gif in gif_elements:
        src = gif.get_attribute("src")
        if src:
            gif_urls.append(src)

    return gif_urls


def scrape_tweet(url, session_file="session-data/session.json"):
    """
    Scrape a tweet using Playwright with optional session information.
    
    Args:
        url (str): URL of the tweet to scrape
        session_file (str): Path to the Twitter session data file
    
    Returns:
        dict: Tweet data including text and media URLs
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] → Starting to scrape Twitter URL: {url}")
    
    tweet_data = {
        "url": url,
        "text": "",
        "username": "",
        "display_name": "",
        "images": [],
        "timestamp": ""
    }
    
    with sync_playwright() as p:
        # Launch browser with specific arguments
        browser = p.chromium.launch(
            headless=True,
            args=[
                '--disable-gpu',
                '--disable-dev-shm-usage',
                '--disable-setuid-sandbox',
                '--no-sandbox',
            ]
        )
        
        # Configure context with resource handling and session data if applicable
        context_options = {
            'viewport': {'width': 1280, 'height': 800},
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'device_scale_factor': 1,
            'is_mobile': False,
            'has_touch': False
        }
        
        # Load session data for Twitter/X.com if available
        if os.path.exists(session_file):
            try:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Loading Twitter/X.com session data from {session_file}")
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                context_options['storage_state'] = session_data
            except Exception as e:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Error loading session data: {str(e)}")
        else:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Session file not found: {session_file}")
        
        context = browser.new_context(**context_options)
        page = context.new_page()
        
        # For tweet images, we shouldn't block image resources
        page.route("**/*", lambda route: route.abort() 
            if route.request.resource_type in ['font', 'media'] 
            else route.continue_())
        
        # Configure page for optimal loading
        page.set_extra_http_headers({"Accept-Language": "en-US,en;q=0.9"})
        
        try:
            # Use a longer timeout for Twitter/X.com pages
            page.goto(url, wait_until='domcontentloaded', timeout=30000)
            
            # Wait a bit longer for Twitter/X.com to load properly with the authenticated session
            page.wait_for_timeout(5000)
            
            # Check if we're logged in by looking for specific elements
            is_logged_in = page.query_selector('a[aria-label="Profile"]') is not None
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Twitter/X.com login status: {'Logged in' if is_logged_in else 'Not logged in'}")
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Twitter Loaded")
            
            # Extract tweet text
            tweet_text_element = page.query_selector('[data-testid="tweetText"]')
            if tweet_text_element:
                tweet_data["text"] = tweet_text_element.inner_text()
            
            # Extract username and display name
            username_element = page.query_selector('div[dir="ltr"] span:has-text("@")')
            if username_element:
                tweet_data["username"] = username_element.inner_text().strip()
            
            display_name_element = page.query_selector('div[data-testid="User-Name"] div[dir="ltr"] span span')
            if display_name_element:
                tweet_data["display_name"] = display_name_element.inner_text().strip()
            
            # Extract images
            tweet_data["images"] = extract_tweet_images(page)

            # Extract gifs using
            tweet_data["gifs"] = extract_tweet_gifs(page)
            
            # Extract timestamp
            time_element = page.query_selector('time')
            if time_element:
                tweet_data["timestamp"] = time_element.get_attribute('datetime')
            
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ✓ Successfully scraped tweet")
            
            # Print the extracted data in a user-friendly format
            print("\n----- TWEET DATA -----")
            print(f"Username: {tweet_data['username']}")
            print(f"Display Name: {tweet_data['display_name']}")
            print(f"Tweet Text: {tweet_data['text']}")
            print(f"Timestamp: {tweet_data['timestamp']}")
            print("\nImages:")
            for img_url in tweet_data["images"]:
                print(f"- {img_url}")
            print("---------------------\n")
            print("GIFs:")
            for gif_url in tweet_data["gifs"]:
                print(f"- {gif_url}")
            print("---------------------\n")
            
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Error loading or parsing the page: {str(e)}")
        
        # Clean up
        browser.close()
        
    return tweet_data

if __name__ == "__main__":
    args = setup_argparse()
    tweet_data = scrape_tweet(args.url, args.session_file)