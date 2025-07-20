#!/usr/bin/env python3
import argparse
import json
import logging
import os
import re
import subprocess
from pathlib import Path
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


class ThreadCounter:
    """
    Analyzes a tweet's conversation thread to count consecutive tweets
    from the original author.
    """

    def __init__(self, tweet_url: str):
        self.tweet_url = tweet_url
        self.tweet_id = self._extract_tweet_id_from_url(tweet_url)
        if not self.tweet_id:
            raise ValueError(f"Could not extract Tweet ID from URL: {tweet_url}")
        self.auth_tokens: Optional[Tuple[str, str, str]] = None

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

    def fetch_and_count(self) -> int:
        """
        Fetches the tweet conversation from the API and counts the tweets in the thread.
        
        Returns:
            The total number of tweets in the thread by the original author.
        """
        if not self.auth_tokens:
            self._extract_auth_tokens()

        auth_token, csrf_token, bearer_token = self.auth_tokens
        
        logger.info(f"Fetching API data for tweet ID: {self.tweet_id}")
        api_url = f"https://x.com/i/api/graphql/zJvfJs3gSbrVhC0MKjt_OQ/TweetDetail"
        
        # ====================================================================
        # START: FIX - Use the full, correct variables and features payload
        # ====================================================================
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
        # ====================================================================
        # END: FIX
        # ====================================================================
        
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

            instructions = api_data['data']['threaded_conversation_with_injections_v2']['instructions']
            entries = next((inst['entries'] for inst in instructions if inst.get('type') == 'TimelineAddEntries'), [])
            
            if not entries:
                raise ValueError("Could not find tweet entries in API response.")

            original_author_id = None
            for entry in entries:
                if entry['entryId'].startswith(f"tweet-{self.tweet_id}"):
                    original_author_id = entry['content']['itemContent']['tweet_results']['result']['core']['user_results']['result']['rest_id']
                    break
            
            if not original_author_id:
                raise ValueError("Could not identify the original author of the provided tweet.")

            logger.info(f"Identified thread author ID: {original_author_id}")
            counted_tweet_ids: Set[str] = set()

            for entry in entries:
                if entry['entryId'].startswith('conversationthread-'):
                    for item in entry.get('content', {}).get('items', []):
                        self._process_item(item, original_author_id, counted_tweet_ids)
                elif entry['entryId'].startswith('tweet-'):
                    self._process_item(entry, original_author_id, counted_tweet_ids)

            return len(counted_tweet_ids)

        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            if e.response: logger.error(f"Response Body: {e.response.text}")
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"Failed to parse API response. Structure may have changed. Error: {e}")
            logger.debug(f"Full API Response Data: {json.dumps(api_data, indent=2)}")
        
        return 0

    def _process_item(self, item: Dict[str, Any], author_id: str, counted_ids: Set[str]):
        """
        Helper function to check if a timeline item is a tweet by the thread author
        and add its ID to the counted set if it is.
        """
        try:
            tweet_result = item.get('content', {}).get('itemContent', {}).get('tweet_results', {}).get('result')
            if not tweet_result:
                tweet_result = item.get('item', {}).get('itemContent', {}).get('tweet_results', {}).get('result')

            if tweet_result:
                tweet_author_id = tweet_result['core']['user_results']['result']['rest_id']
                tweet_id = tweet_result['rest_id']
                
                if tweet_author_id == author_id:
                    if tweet_id not in counted_ids:
                        logger.info(f"Found thread tweet: {tweet_id}")
                        counted_ids.add(tweet_id)
        except (KeyError, TypeError):
            pass


# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(
        description="Count the number of tweets in a thread by the original author.",
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

        counter = ThreadCounter(args.url)
        count = counter.fetch_and_count()

        if count > 0:
            print(f"\n✅ This thread contains {count} tweet(s) from the original author.")
        else:
            print("\n❌ Could not count the tweets. The URL might be invalid, the tweet deleted, or an API error occurred.")
            if args.verbose:
                print("Check the logs above for specific error details.")
            else:
                print("Run with the --verbose flag for more details.")

    except Exception as e:
        logger.critical(f"An unexpected error occurred: {e}", exc_info=args.verbose)
        print(f"\n❌ An error occurred: {e}")


if __name__ == "__main__":
    main()