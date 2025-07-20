#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from playwright.sync_api import sync_playwright

# --- Basic Configuration ---
# We'll configure the logging level in main() based on user arguments.
logger = logging.getLogger(__name__)

# Session data will be stored in a sub-directory next to the script
SCRIPT_DIR = Path(__file__).parent
SESSION_DIR = SCRIPT_DIR / "session-data"
SESSION_FILE = SESSION_DIR / "session.json"
NODE_SCRIPT_PATH = SCRIPT_DIR / "refresh_twitter_session.js"


# --- Helper Functions ---

def clean_filename(filename: str) -> str:
    """
    Cleans a filename by converting to lowercase, replacing special characters
    and spaces with underscores, and removing consecutive underscores.
    """
    base_name, extension = os.path.splitext(filename)
    base_name = base_name.lower()
    extension = extension.lower()
    base_name = re.sub(r'[^a-z0-9_]', '_', base_name)
    base_name = re.sub(r'_{2,}', '_', base_name)
    base_name = base_name.strip('_')
    return base_name + extension


# --- Core Logic Classes ---

class TwitterSessionManager:
    """
    Manages the Twitter/X authentication session, refreshing it when necessary
    by calling an external Node.js script.
    """

    @staticmethod
    def ensure_valid_session() -> bool:
        """
        Ensures a valid session exists. Checks the current session and
        refreshes if it's invalid or missing.
        """
        logger.info("Ensuring a valid Twitter/X session...")
        if TwitterSessionManager._is_session_valid_playwright():
            logger.info("Current session is valid.")
            return True
        else:
            logger.warning("Session is invalid or missing. Attempting refresh.")
            if TwitterSessionManager._run_refresh_script():
                logger.info("Session refreshed. Re-validating...")
                if TwitterSessionManager._is_session_valid_playwright():
                    logger.info("Refreshed session confirmed as valid.")
                    return True
                else:
                    logger.error("Session was refreshed but still fails validation.")
                    return False
            else:
                logger.error("Session refresh script failed.")
                return False

    @staticmethod
    def _is_session_valid_playwright() -> bool:
        """
        Checks if the session stored in the file is currently valid using Playwright.
        """
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
        """
        Executes the Node.js script to refresh the Twitter session.
        """
        if not NODE_SCRIPT_PATH.exists():
            logger.error(f"Refresh script not found at {NODE_SCRIPT_PATH}")
            logger.error("Please ensure 'refresh_twitter_session.js' and 'package.json' are in the same directory as this script.")
            return False

        if 'X_USERNAME' not in os.environ or 'X_PASSWORD' not in os.environ:
            logger.error("X_USERNAME and X_PASSWORD environment variables must be set to refresh the session.")
            return False
            
        SESSION_DIR.mkdir(exist_ok=True)

        try:
            logger.info("Attempting to refresh session via Node.js script...")
            if not (SCRIPT_DIR / "node_modules").exists():
                logger.info("Node modules not found. Running 'npm install'...")
                subprocess.run(
                    ["npm", "install"],
                    cwd=SCRIPT_DIR,
                    check=True,
                    capture_output=True, text=True
                )

            subprocess.run(
                ["node", str(NODE_SCRIPT_PATH)],
                capture_output=True,
                text=True,
                check=True,
                cwd=SCRIPT_DIR
            )
            logger.info("Session data refreshed successfully by Node.js script.")
            return True
        except FileNotFoundError:
            logger.error("Error: 'node' or 'npm' command not found. Please ensure Node.js is installed and in your PATH.")
            return False
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to refresh session. Node script exited with an error.")
            logger.error(f"Stdout: {e.stdout}")
            logger.error(f"Stderr: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"An unexpected error occurred while running the refresh script: {e}")
            return False


class TweetMediaExtractor:
    """
    Extracts tweet information and media URLs using a combination of
    Playwright-based scraping and direct API calls.
    """

    def __init__(self, tweet_url: str):
        self.tweet_url = tweet_url
        self.tweet_id = self._extract_tweet_id_from_url(tweet_url)
        if not self.tweet_id:
            raise ValueError(f"Could not extract Tweet ID from URL: {tweet_url}")

        self.auth_tokens: Optional[Tuple[str, str, str]] = None
        self.tweet_details: Dict[str, Any] = {}

    @staticmethod
    def _extract_tweet_id_from_url(url: str) -> Optional[str]:
        match = re.search(r"(?:twitter\.com|x\.com)/[^/]+/status/(\d+)", url)
        return match.group(1) if match else None
        
    def _extract_auth_tokens(self) -> None:
        logger.info("Extracting authentication tokens from session...")
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
                logger.warning("Could not navigate to home to capture bearer token, but may have it from context.")
            
            browser.close()

            if not all([auth_token, csrf_token, bearer_token]):
                missing = [name for name, val in [("auth_token", auth_token), ("csrf_token", csrf_token), ("bearer_token", bearer_token)] if not val]
                raise RuntimeError(f"Failed to extract required tokens: {', '.join(missing)}")
            
            self.auth_tokens = (auth_token, csrf_token, bearer_token)
            logger.info("Successfully extracted all authentication tokens.")

    def _get_initial_details(self) -> None:
        logger.info("Getting initial tweet details from page...")
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(storage_state=str(SESSION_FILE))
            page = context.new_page()
            try:
                page.goto(self.tweet_url, timeout=30000)
                page.wait_for_selector('article[data-testid="tweet"]', state='visible', timeout=15000)
                
                user_handle = page.evaluate('() => document.querySelector(\'[data-testid="User-Name"] div[dir="ltr"]\').textContent.trim().replace("@", "")')
                timestamp_str = page.evaluate('() => document.querySelector("time").getAttribute("datetime")')
                
                dt_object = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                
                self.tweet_details = {
                    'user_handle': user_handle,
                    'timestamp': int(dt_object.timestamp())
                }
            finally:
                browser.close()

    def get_media_items(self) -> List[Dict[str, Any]]:
        if not self.auth_tokens:
            self._extract_auth_tokens()
        if not self.tweet_details:
            self._get_initial_details()

        auth_token, csrf_token, bearer_token = self.auth_tokens
        
        logger.info(f"Fetching API data for tweet ID: {self.tweet_id}")
        api_url = f"https://x.com/i/api/graphql/zJvfJs3gSbrVhC0MKjt_OQ/TweetDetail"
        
        params = {
            "variables": json.dumps({
                "focalTweetId": self.tweet_id,
                "with_rux_injections": False,
                "includePromotedContent": False,
                "withCommunity": False,
                "withQuickPromoteEligibilityTweetFields": False,
                "withBirdwatchNotes": False,
                "withVoice": False,
                "withV2Timeline": True
            }),
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

        headers = {
            "Authorization": f"Bearer {bearer_token}",
            "X-Csrf-Token": csrf_token,
            "Cookie": f"auth_token={auth_token}; ct0={csrf_token}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
        }

        try:
            response = requests.get(api_url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            api_data = response.json()
            
            # ====================================================================
            # START: FIX - Robustly find the correct 'entries' list
            # ====================================================================
            instructions = api_data['data']['threaded_conversation_with_injections_v2']['instructions']
            
            entries = []
            for instruction in instructions:
                if instruction.get('type') == 'TimelineAddEntries':
                    entries = instruction.get('entries', [])
                    break
            # ====================================================================
            # END: FIX
            # ====================================================================

            if not entries:
                logger.error("Could not find 'TimelineAddEntries' instruction in the API response.")
                return []

            tweet_entry = next((e for e in entries if e['entryId'].startswith(f"tweet-{self.tweet_id}")), None)
            
            if not tweet_entry:
                logger.error("Could not find the target tweet in the API response entries.")
                return []
            
            media_list = tweet_entry['content']['itemContent']['tweet_results']['result'].get('legacy', {}).get('extended_entities', {}).get('media', [])
            
            media_items = []
            for index, media_data in enumerate(media_list):
                media_type = media_data.get('type')
                item = {'type': media_type, 'index': index, 'url': None, 'extension': None}
                
                if media_type == 'photo':
                    item['url'] = media_data.get('media_url_https')
                    item['extension'] = Path(item['url']).suffix[1:] if item['url'] else 'jpg'
                elif media_type in ('video', 'animated_gif'):
                    variants = media_data.get('video_info', {}).get('variants', [])
                    mp4_variants = [v for v in variants if v.get('content_type') == 'video/mp4']
                    if mp4_variants:
                        best_variant = max(mp4_variants, key=lambda v: v.get('bitrate', 0))
                        item['url'] = best_variant['url']
                        item['extension'] = 'mp4'
                
                if item['url']:
                    media_items.append(item)
            
            logger.info(f"Found {len(media_items)} media items via API.")
            return media_items

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            if e.response is not None:
                logger.error(f"Response Body: {e.response.text}")
        except (KeyError, IndexError) as e:
            logger.error(f"Failed to parse API response JSON. Structure may have changed. Error: {e}")
            logger.debug(f"Full API Response Data: {json.dumps(api_data, indent=2)}")
        return []


class MediaDownloader:
    """Handles the downloading and saving of media files."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.download_count = 0

    def download(self, media_items: List[Dict[str, Any]], tweet_details: Dict[str, Any], tweet_id: str):
        if not media_items:
            logger.warning("No media items provided to download.")
            return

        timestamp = tweet_details.get('timestamp')
        user_handle = tweet_details.get('user_handle', 'unknown_user')
        
        logger.info(f"Starting download of {len(media_items)} media items...")

        for item in media_items:
            media_url = item.get('url')
            if not media_url:
                continue

            filename_base = f"{timestamp}-tweet-{user_handle}-{tweet_id}-asset-{item['index'] + 1}.{item['extension']}"
            output_path = self.output_dir / clean_filename(filename_base)
            
            try:
                logger.info(f"Downloading {item['type']} to {output_path}...")
                with requests.get(media_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(output_path, 'wb') as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                logger.info(f"✓ Successfully downloaded {output_path.name}")
                self.download_count += 1
            except requests.RequestException as e:
                logger.error(f"✕ Failed to download {media_url}: {e}")
            except Exception as e:
                 logger.error(f"✕ An unexpected error occurred during download of {media_url}: {e}")


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Download all media (images/videos) from a single Tweet/X URL.",
        epilog="Requires Node.js and environment variables X_USERNAME and X_PASSWORD for the initial login."
    )
    parser.add_argument(
        "-u", "--url",
        required=True,
        help="The full URL of the tweet (e.g., 'https://x.com/user/status/12345')."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="./tweet_media",
        help="The local directory to save downloaded media files (default: ./tweet_media)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging for debugging purposes."
    )
    args = parser.parse_args()

    # --- FIX: Configure logging level based on verbosity ---
    log_level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

    try:
        if not TwitterSessionManager.ensure_valid_session():
            raise RuntimeError("Could not establish a valid session.")

        extractor = TweetMediaExtractor(args.url)
        media_items_to_download = extractor.get_media_items()
        
        if media_items_to_download:
            downloader = MediaDownloader(Path(args.output_dir).expanduser())
            downloader.download(media_items_to_download, extractor.tweet_details, extractor.tweet_id)
            
            if downloader.download_count > 0:
                print(f"\n✅ Success: Downloaded {downloader.download_count} media file(s) to '{args.output_dir}'.")
            else:
                print(f"\n❌ Failure: Could not download any media files. Check logs for errors.")
        else:
             print("\n❌ Failure: No media items were found for the given URL. The tweet may contain no images/videos or there was an API error.")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=args.verbose)
        print(f"\n❌ An unexpected error occurred: {e}")
        print("Run with --verbose flag for more details.")


if __name__ == "__main__":
    main()