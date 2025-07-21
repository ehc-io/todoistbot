#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Tuple, Set

import requests
from playwright.sync_api import sync_playwright

# --- Basic Configuration ---
logger = logging.getLogger(__name__)

# Session data will be stored in a sub-directory next to the script
SCRIPT_DIR = Path(__file__).parent
SESSION_DIR = SCRIPT_DIR / "session-data"
SESSION_FILE = SESSION_DIR / "session.json"
NODE_SCRIPT_PATH = SCRIPT_DIR / "refresh_twitter_session.js"


# --- Core Logic Classes ---

class TwitterSessionManager:
    """
    Manages the Twitter/X authentication session, refreshing it when necessary
    by calling an external Node.js script.
    """
    @staticmethod
    def ensure_valid_session() -> bool:
        logger.info("Ensuring a valid Twitter/X session...")
        if TwitterSessionManager._is_session_valid_playwright():
            logger.info("Current session is valid.")
            return True
        else:
            logger.warning("Session is invalid or missing. Attempting refresh.")
            if TwitterSessionManager._run_refresh_script():
                logger.info("Session refreshed successfully.")
                return True
            else:
                logger.error("Session refresh script failed.")
                return False

    @staticmethod
    def _is_session_valid_playwright() -> bool:
        if not SESSION_FILE.exists():
            logger.info("Session file does not exist.")
            return False
        logger.info("Verifying session validity using Playwright...")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(storage_state=str(SESSION_FILE))
                page = context.new_page()
                page.goto("https://x.com/home", timeout=20000)
                compose_button_selector = 'a[data-testid="SideNav_NewTweet_Button"]'
                try:
                    page.wait_for_selector(compose_button_selector, timeout=10000)
                    is_logged_in = True
                except Exception:
                    is_logged_in = False
                browser.close()
                return is_logged_in
        except Exception as e:
            logger.error(f"An error occurred during Playwright session validation: {e}")
            return False

    @staticmethod
    def _run_refresh_script() -> bool:
        if not NODE_SCRIPT_PATH.exists():
            logger.error(f"Refresh script not found at {NODE_SCRIPT_PATH}")
            return False
        if 'X_USERNAME' not in os.environ or 'X_PASSWORD' not in os.environ:
            logger.error("X_USERNAME and X_PASSWORD environment variables must be set.")
            return False
        SESSION_DIR.mkdir(exist_ok=True)
        try:
            logger.info("Attempting to refresh session via Node.js script...")
            if not (SCRIPT_DIR / "node_modules").exists():
                logger.info("Node modules not found. Running 'npm install'...")
                subprocess.run(["npm", "install"], cwd=SCRIPT_DIR, check=True, capture_output=True, text=True)
            subprocess.run(["node", str(NODE_SCRIPT_PATH)], capture_output=True, text=True, check=True, cwd=SCRIPT_DIR)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            logger.error(f"Failed to run refresh script: {e}")
            return False


class ThreadScraper:
    """
    Analyzes a tweet's conversation thread to count consecutive tweets
    from the original author.
    """

    def __init__(
        self,
        tweet_url: str,
        save_media_locally: bool = False,
        s3_upload: bool = False,
        s3_bucket: str = "2025-captured-notes",
        media_output_dir: str = "downloads"
    ):
        self.tweet_url = tweet_url
        self.tweet_id = self._extract_tweet_id_from_url(tweet_url)
        if not self.tweet_id:
            raise ValueError(f"Could not extract Tweet ID from URL: {self.tweet_url}")
        self.auth_tokens: Optional[Tuple[str, str, str]] = None
        self.scraped_tweets: List[Dict[str, Any]] = []
        self.save_media_locally = save_media_locally
        self.s3_upload = s3_upload
        self.s3_bucket = s3_bucket
        self.media_output_dir = media_output_dir

    @staticmethod
    def _extract_tweet_id_from_url(url: str) -> Optional[str]:
        match = re.search(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d+)", url)
        return match.group(1) if match else None

    def _extract_auth_tokens(self) -> None:
        """Extracts authentication tokens required for API calls from the session file."""
        logger.info("Extracting authentication tokens...")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(storage_state=str(SESSION_FILE))
            cookies = context.cookies()
            auth_token = next((c["value"] for c in cookies if c["name"] == "auth_token"), None)
            csrf_token = next((c["value"] for c in cookies if c["name"] == "ct0"), None)
            bearer_token = None
            def handle_request(request):
                nonlocal bearer_token
                auth_header = request.headers.get('authorization')
                if auth_header and auth_header.startswith('Bearer '):
                    bearer_token = auth_header.replace('Bearer ', '')
            page = context.new_page()
            page.on('request', handle_request)
            try:
                page.goto("https://x.com/home", wait_until='domcontentloaded', timeout=15000)
                page.wait_for_timeout(5000)
            except Exception:
                logger.warning("Could not navigate to home to capture bearer token, but may already have it.")
            browser.close()
            if not all([auth_token, csrf_token, bearer_token]):
                raise RuntimeError("Failed to extract all required authentication tokens.")
            self.auth_tokens = (auth_token, csrf_token, bearer_token)

    def scrape_thread(self) -> List[Dict[str, Any]]:
        """
        Fetches the tweet conversation from the API and scrapes all tweets in a
        continuous thread from the original author.
        
        Returns:
            A list of dictionaries, where each dictionary represents a tweet 
            in the thread, in chronological order.
        """
        if not self.auth_tokens:
            self._extract_auth_tokens()

        auth_token, csrf_token, bearer_token = self.auth_tokens
        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Csrf-Token": csrf_token,
            "Cookie": f"auth_token={auth_token}; ct0={csrf_token}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        }
        api_url = f"https://x.com/i/api/graphql/zJvfJs3gSbrVhC0MKjt_OQ/TweetDetail"
        
        all_tweets: List[Dict[str, Any]] = []
        seen_tweet_ids: Set[str] = set()
        cursor: Optional[str] = None
        original_author_id: Optional[str] = None
        is_first_request = True

        while True:
            try:
                # ====================================================================
                # START: FIX - Use the full, correct variables and features payload
                # ====================================================================
                variables = {
                    "focalTweetId": self.tweet_id,
                    "with_rux_injections": False,
                    "includePromotedContent": False,
                    "withCommunity": False,
                    "withQuickPromoteEligibilityTweetFields": False,
                    "withBirdwatchNotes": False,
                    "withVoice": False,
                    "withV2Timeline": True
                }
                if cursor:
                    variables["cursor"] = cursor

                params = {
                    "variables": json.dumps(variables),
                    "features": json.dumps({
                        "rweb_tipjar_consumption_enabled": False,
                        "responsive_web_graphql_exclude_directive_enabled": False,
                        "verified_phone_label_enabled": False,
                        "creator_subscriptions_tweet_preview_api_enabled": False,
                        "responsive_web_graphql_timeline_navigation_enabled": False,
                        "responsive_web_graphql_skip_user_profile_image_extensions_enabled": False,
                        "communities_web_enable_tweet_community_results_fetch": False,
                        "c9s_tweet_anatomy_moderator_badge_enabled": False,
                        "articles_preview_enabled": True,
                        "tweetypie_unmention_optimization_enabled": False,
                        "responsive_web_edit_tweet_api_enabled": False,
                        "graphql_is_translatable_rweb_tweet_is_translatable_enabled": False,
                        "view_counts_everywhere_api_enabled": False,
                        "longform_notetweets_consumption_enabled": False,
                        "responsive_web_twitter_article_tweet_consumption_enabled": False,
                        "tweet_awards_web_tipping_enabled": False,
                        "creator_subscriptions_quote_tweet_preview_enabled": False,
                        "freedom_of_speech_not_reach_fetch_enabled": False,
                        "standardized_nudges_misinfo": False,
                        "tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled": False,
                        "tweet_with_visibility_results_prefer_gql_media_interstitial_enabled": False,
                        "rweb_video_timestamps_enabled": False,
                        "longform_notetweets_rich_text_read_enabled": False,
                        "longform_notetweets_inline_media_enabled": False,
                        "responsive_web_enhance_cards_enabled": False
                    })
                }
                # ====================================================================
                # END: FIX
                # ====================================================================
                
                response = requests.get(api_url, params=params, headers=headers, timeout=15)
                response.raise_for_status()
                api_data = response.json()

                instructions = api_data['data']['threaded_conversation_with_injections_v2']['instructions']
                entries = next((inst['entries'] for inst in instructions if inst.get('type') == 'TimelineAddEntries'), [])
                
                if not entries:
                    break

                if is_first_request:
                    for entry in entries:
                        if entry['entryId'].startswith(f"tweet-{self.tweet_id}"):
                            original_author_id = entry['content']['itemContent']['tweet_results']['result']['core']['user_results']['result']['rest_id']
                            break
                    if not original_author_id:
                        raise ValueError("Could not identify the original author of the provided tweet.")
                    logger.info(f"Identified thread author ID: {original_author_id}")
                    is_first_request = False

                new_tweets_found_in_page = False
                thread_interrupted = False
                
                page_tweets = []
                for entry in entries:
                    if entry['entryId'].startswith('conversationthread-'):
                        for item in entry.get('content', {}).get('items', []):
                            tweet_data = self._get_tweet_data(item)
                            if tweet_data:
                                page_tweets.append(tweet_data)
                    elif entry['entryId'].startswith('tweet-'):
                        tweet_data = self._get_tweet_data(entry)
                        if tweet_data:
                            page_tweets.append(tweet_data)

                for tweet in page_tweets:
                    tweet_id = tweet['rest_id']
                    author_id = tweet['core']['user_results']['result']['rest_id']

                    if author_id == original_author_id and tweet_id not in seen_tweet_ids:
                        # Process media if requested
                        if self.save_media_locally or self.s3_upload:
                            media_results = _process_and_upload_media(
                                tweet,
                                self.save_media_locally,
                                self.s3_upload,
                                self.s3_bucket,
                                self.media_output_dir
                            )
                            tweet['media_s3_urls'] = media_results.get('s3_urls', [])

                        all_tweets.append(tweet)
                        seen_tweet_ids.add(tweet_id)
                        new_tweets_found_in_page = True
                    elif author_id != original_author_id:
                        thread_interrupted = True
                        break
                
                if thread_interrupted or not new_tweets_found_in_page:
                    break
                
                # Find the next cursor
                new_cursor = None
                for entry in entries:
                    if entry['entryId'].startswith('cursor-bottom-'):
                        new_cursor = entry.get('content', {}).get('value')
                        break
                
                cursor = new_cursor
                if not cursor:
                    break

            except requests.RequestException as e:
                logger.error(f"API request failed: {e}")
                if e.response: logger.error(f"Response Body: {e.response.text}")
                break
            except (KeyError, IndexError, ValueError) as e:
                logger.error(f"Failed to parse API response. Structure may have changed. Error: {e}")
                logger.debug(f"Full API Response Data: {json.dumps(api_data, indent=2)}")
                break
        
        self.scraped_tweets = sorted(all_tweets, key=lambda x: int(x['rest_id']))
        return self.scraped_tweets

    def _get_tweet_data(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extracts tweet data from a timeline item if it exists."""
        try:
            tweet_result = item.get('content', {}).get('itemContent', {}).get('tweet_results', {}).get('result')
            if not tweet_result:
                tweet_result = item.get('item', {}).get('itemContent', {}).get('tweet_results', {}).get('result')
            
            if tweet_result and 'rest_id' in tweet_result:
                 return tweet_result
        except (KeyError, TypeError):
            pass
        return None

    def _process_item(self, item: Dict[str, Any], author_id: str, processed_tweets: List[Dict[str, Any]], seen_ids: Set[str]):
        """
        Helper function to check if a timeline item is a tweet by the thread author
        and add its content to the processed_tweets list if it is.
        """
        try:
            tweet_result = item.get('content', {}).get('itemContent', {}).get('tweet_results', {}).get('result')
            if not tweet_result:
                tweet_result = item.get('item', {}).get('itemContent', {}).get('tweet_results', {}).get('result')

            if tweet_result:
                tweet_author_id = tweet_result['core']['user_results']['result']['rest_id']
                tweet_id = tweet_result['rest_id']
                
                if tweet_author_id == author_id and tweet_id not in seen_ids:
                    logger.info(f"Scraping tweet: {tweet_id}")
                    processed_tweets.append(tweet_result)
                    seen_ids.add(tweet_id)
        except (KeyError, TypeError):
            pass


def _process_and_upload_media(
    tweet: Dict[str, Any],
    save_media_locally: bool,
    s3_upload: bool,
    s3_bucket: str,
    media_output_dir: str
) -> Dict[str, List[str]]:
    """
    Downloads media from a tweet and optionally uploads it to S3.
    Returns a dictionary with local paths and S3 URLs.
    """
    media_results = {"local_paths": [], "s3_urls": []}
    media_files = tweet.get('legacy', {}).get('extended_entities', {}).get('media', [])
    
    if not media_files:
        return media_results

    if s3_upload:
        try:
            import boto3
            s3_client = boto3.client('s3')
        except ImportError:
            logger.error("boto3 is not installed. Cannot upload to S3.")
            return media_results
    else:
        s3_client = None

    if save_media_locally:
        Path(media_output_dir).mkdir(parents=True, exist_ok=True)

    for media_item in media_files:
        media_url = media_item.get('media_url_https')
        if media_item.get('type') == 'video':
            variants = media_item.get('video_info', {}).get('variants', [])
            highest_bitrate_variant = max(variants, key=lambda v: v.get('bitrate', 0), default=None)
            if highest_bitrate_variant:
                media_url = highest_bitrate_variant['url']

        if not media_url:
            continue

        try:
            response = requests.get(media_url, stream=True, timeout=20)
            response.raise_for_status()
            
            # Use a temporary file to avoid saving corrupted files
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                tmp_file_path = tmp_file.name

            file_extension = Path(urlparse(media_url).path).suffix.split('?')[0] or '.jpg'
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"tweet_{tweet['rest_id']}_{media_item['id_str']}_{timestamp}{file_extension}"
            
            local_path = None
            if save_media_locally:
                local_path = Path(media_output_dir) / filename
                os.rename(tmp_file_path, local_path)
                media_results["local_paths"].append(str(local_path))
                logger.info(f"Downloaded media to {local_path}")
            
            if s3_upload and s3_client:
                # Use week-based folder structure
                week_of_year = datetime.now().isocalendar()[1]
                folder_name = f"week_{week_of_year:02d}"
                object_name = f"{folder_name}/{filename}"

                upload_path = local_path if local_path else tmp_file_path
                s3_client.upload_file(str(upload_path), s3_bucket, object_name)
                s3_url = f"https://{s3_bucket}.s3.amazonaws.com/{object_name}"
                media_results["s3_urls"].append(s3_url)
                logger.info(f"Uploaded media to S3: {s3_url}")

            # Clean up temp file if not moved
            if not save_media_locally and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

        except requests.RequestException as e:
            logger.error(f"Failed to download media {media_url}: {e}")
        except Exception as e:
            logger.error(f"An error occurred during media processing for {media_url}: {e}")
            if 'tmp_file_path' in locals() and os.path.exists(tmp_file_path):
                os.remove(tmp_file_path)

    return media_results


def format_tweet_thread_output(
    tweets: List[Dict[str, Any]], 
    original_url: str,
) -> str:
    """Formats the scraped tweet data into a structured markdown string."""
    if not tweets:
        return "No tweets were found to format."

    first_tweet = tweets[0]
    author_info = first_tweet['core']['user_results']['result']
    author_name = author_info['legacy']['name']
    author_handle = author_info['legacy']['screen_name']
    tweet_id = first_tweet['rest_id']
    
    # Format the header
    output = [
        f"## [{author_name} (@{author_handle}) on X]({original_url})",
        "",
        "**Type**: Twitter",
        "*Extraction Method: playwright_bs4*",
        "",
        f"User: @{author_handle} ({author_name})",
        f"Tweet ID: {tweet_id}",
    ]

    # Try to get the post date
    try:
        created_at_str = first_tweet['legacy']['created_at']
        dt_object = datetime.strptime(created_at_str, '%a %b %d %H:%M:%S %z %Y')
        output.append(f"Posted on: {dt_object.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    except (KeyError, ValueError):
        output.append("Posted on: [Date not available]")

    output.append("")

    # Format each tweet in the thread
    for i, tweet in enumerate(tweets, 1):
        output.append(f"--- tweet #{i} ---")
        
        tweet_text = tweet.get('legacy', {}).get('full_text', '')
        if 'note_tweet' in tweet:
             tweet_text = tweet['note_tweet']['note_tweet_results']['result']['text']

        # Clean up text for display by removing the t.co link at the end
        cleaned_text = re.sub(r" https://t.co/\w+$", "", tweet_text).strip()
        output.append(cleaned_text)
        output.append("")

        media_files = tweet.get('legacy', {}).get('extended_entities', {}).get('media', [])
        s3_urls = tweet.get('media_s3_urls', [])

        if s3_urls:
            output.append(f"Media: {len(s3_urls)} items detected")
            for i, url in enumerate(s3_urls, 1):
                media_type = "Video" if ".mp4" in url else "Photo"
                output.append(f"- [{media_type}: {i}]({url})")
        elif media_files:
            output.append(f"Media: {len(media_files)} items detected")
            for j, media in enumerate(media_files, 1):
                media_type = media.get('type')
                if media_type == 'video':
                    video_url = media.get('video_info', {}).get('variants', [{}])[0].get('url')
                    output.append(f"- [Video: {j}]({video_url})")
                elif media_type == 'photo':
                    photo_url = media.get('media_url_https')
                    output.append(f"- [Photo: {j}]({photo_url})")
        else:
            output.append("[Note: No media files found to process]")

        output.append("")

    output.append("--- end of tweet thread ---")
    return "\n".join(output)


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Scrape all tweets in a thread by the original author.",
        epilog="Requires Node.js and environment variables X_USERNAME and X_PASSWORD for initial login."
    )
    parser.add_argument(
        "-u", "--url",
        required=True,
        help="The full URL of any tweet within the thread."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging purposes."
    )
    args = parser.parse_args()

    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    try:
        if not TwitterSessionManager.ensure_valid_session():
            raise RuntimeError("Could not establish a valid session.")

        scraper = ThreadScraper(args.url) # This will need to be updated to pass S3 args
        tweets = scraper.scrape_thread()

        if tweets:
            formatted_output = format_tweet_thread_output(tweets, args.url)
            print(formatted_output)
        else:
            print("\n❌ Could not scrape any tweets. The URL might be invalid, the tweet deleted, or an API error occurred.")
            if args.verbose:
                print("Check the logs above for specific error details.")
            else:
                print("Run with the --verbose flag for more details.")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=args.verbose)
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    main()