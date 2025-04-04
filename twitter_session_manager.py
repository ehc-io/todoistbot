# twitter_session_manager.py
import subprocess
import json
import os
import logging
from pathlib import Path
from playwright.sync_api import sync_playwright, Error as PlaywrightError

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TwitterSessionManager:
    """Manages Twitter authentication session validation and refresh."""

    def __init__(self, session_dir="session-data", session_file="session.json"):
        self.session_dir = Path(session_dir)
        self.session_file = session_file
        self.session_path = self.session_dir / self.session_file
        self._ensure_session_dir_exists()

    def _ensure_session_dir_exists(self):
        """Ensure the session directory exists."""
        if not self.session_dir.exists():
            logger.info(f"Creating session data directory: {self.session_dir}")
            self.session_dir.mkdir(parents=True, exist_ok=True)

    def get_session_path(self) -> Path:
        """Return the full path to the session file."""
        return self.session_path

    def _run_refresh_script(self) -> bool:
        """
        Executes the Node.js script to refresh the Twitter session.
        Returns True if successful, False otherwise.
        """
        script_path = Path(__file__).parent / "refresh_twitter_session.js"
        if not script_path.exists():
            logger.error(f"Refresh script not found at {script_path}")
            return False

        if 'X_USERNAME' not in os.environ or 'X_PASSWORD' not in os.environ:
            logger.error("X_USERNAME and X_PASSWORD environment variables must be set to refresh session.")
            return False

        try:
            logger.info("Attempting to refresh user session via Node.js script...")
            # Ensure the script is executable (useful in some environments)
            os.chmod(script_path, 0o755)

            result = subprocess.run(
                ["node", str(script_path)],
                capture_output=True,
                text=True,
                check=False, # Don't throw error on non-zero exit code, handle manually
                cwd=Path(__file__).parent # Run script from its directory
            )

            logger.debug(f"Refresh script stdout: {result.stdout}")
            logger.debug(f"Refresh script stderr: {result.stderr}")

            if result.returncode != 0:
                logger.error(f"Failed to refresh session. Node script exited with code {result.returncode}.")
                logger.error(f"Stderr: {result.stderr}")
                return False

            if not self.session_path.exists():
                 logger.error(f"Session refresh script ran but session file '{self.session_path}' was not created.")
                 return False

            logger.info("Session data refreshed successfully by Node.js script.")
            return True
        except FileNotFoundError:
             logger.error("Error: 'node' command not found. Please ensure Node.js is installed and in your PATH.")
             return False
        except Exception as e:
            logger.error(f"Error running refresh script: {e}")
            return False

    def _is_session_valid_playwright(self) -> bool:
        """Checks if the session stored in the file is currently valid using Playwright."""
        if not self.session_path.exists():
            logger.info("Session file does not exist.")
            return False

        logger.info("Verifying session validity using Playwright...")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                try:
                    context = browser.new_context(storage_state=str(self.session_path))
                    page = context.new_page()
                    try:
                         # Navigate to a page that requires login
                        page.goto("https://x.com/home", timeout=20000, wait_until='domcontentloaded')
                        page.wait_for_timeout(3000) # Allow redirect/rendering

                        # Check for a reliable indicator of being logged in
                        # Option 1: Check title (can be brittle)
                        # is_logged_in = "Home / X" in page.title() or "/ X" in page.title()

                        # Option 2: Check for a unique element only present when logged in
                        # Example: Profile link in the sidebar
                        profile_link_selector = 'a[data-testid="AppTabBar_Profile_Link"]'
                        compose_button_selector = 'a[data-testid="SideNav_NewTweet_Button"]' # More stable?

                        is_logged_in = page.query_selector(compose_button_selector) is not None

                        if is_logged_in:
                            logger.info("Playwright check: Session appears valid.")
                        else:
                            logger.warning(f"Playwright check: Session appears invalid (login indicator not found on x.com/home). Title: {page.title()}")

                        page.close()
                        context.close()
                        return is_logged_in

                    except PlaywrightError as e:
                         logger.error(f"Playwright error during session validation check: {e}")
                         # Attempt to close gracefully
                         try: page.close()
                         except: pass
                         try: context.close()
                         except: pass
                         return False # Treat playwright errors as invalid session
                finally:
                    browser.close()

        except Exception as e:
            logger.error(f"Unexpected error during Playwright session validation: {e}")
            return False # Treat other errors as invalid

    def ensure_valid_session(self) -> bool:
        """
        Ensures a valid session exists. Checks current session, refreshes if invalid or missing.
        Returns True if a valid session is confirmed or established, False otherwise.
        """
        logger.info("Ensuring valid Twitter/X session...")
        if self._is_session_valid_playwright():
            logger.info("Current session is valid.")
            return True
        else:
            logger.warning("Current session is invalid or missing. Attempting refresh.")
            if self._run_refresh_script():
                logger.info("Session refreshed. Re-validating...")
                # Re-validate after refresh for confirmation
                if self._is_session_valid_playwright():
                     logger.info("Refreshed session confirmed as valid.")
                     return True
                else:
                     logger.error("Session was refreshed, but still fails validation check.")
                     return False
            else:
                logger.error("Session refresh failed.")
                return False
