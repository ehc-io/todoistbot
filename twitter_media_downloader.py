# twitter_media_downloader.py
import requests
import logging
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class TwitterMediaDownloader:
    """Handles downloading media assets from Twitter/X."""

    def __init__(self, output_dir: str = "./downloads"):
        self.output_dir = Path(output_dir)
        self._ensure_output_dir_exists()

    def _ensure_output_dir_exists(self):
        """Creates the output directory if it doesn't exist."""
        if not self.output_dir.exists():
            logger.info(f"Creating media output directory: {self.output_dir}")
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def download_media_items(self,
                              media_items: List[Dict[str, Any]],
                              tweet_details: Dict[str, Any],
                              tweet_id: str) -> List[Dict[str, Any]]:
        """
        Downloads media files specified in media_items list.

        Args:
            media_items: List of dicts from TwitterAPIClient.extract_media_urls_from_api_data.
                         Each dict should have 'url', 'type', 'index', 'extension'.
            tweet_details: Dict containing 'timestamp_ms' and 'user_handle' from TwitterContentExtractor.
            tweet_id: The ID of the tweet.

        Returns:
            List of dicts, each containing info about a successfully downloaded file
            (e.g., {'path': str(output_path), 'type': media_type, 'index': index}).
        """
        downloaded_files = []
        if not media_items:
            logger.info("No media items provided to download.")
            return downloaded_files

        timestamp = tweet_details.get('timestamp_ms', int(Path.cwd().stat().st_ctime * 1000)) # Use folder creation time as fallback
        user_handle = tweet_details.get('user_handle', 'unknown_user')

        # Sanitize user_handle for filename
        user_handle_safe = "".join(c if c.isalnum() or c in ['_','-'] else '_' for c in user_handle)


        logger.info(f"Starting download of {len(media_items)} media items for tweet {tweet_id}...")

        for item in media_items:
            media_url = item.get('url')
            media_type = item.get('type', 'unknown')
            index = item.get('index', 0)
            extension = item.get('extension', 'bin') # Default extension

            if not media_url:
                logger.warning(f"Skipping media item index {index}: URL is missing.")
                continue

            # Construct filename: timestamp_ms-user_handle-tweet_id-asset-index.ext
            filename = f"{timestamp}-{user_handle_safe}-{tweet_id}-asset{index + 1}.{extension}"
            output_path = self.output_dir / filename

            try:
                logger.info(f"Downloading {media_type} (index {index}) to {output_path}...")
                # Use streaming download with progress bar
                with requests.get(media_url, stream=True, timeout=60) as r: # Increased timeout for potentially large files
                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))

                    with open(output_path, 'wb') as f, \
                         tqdm(total=total_size, unit='B', unit_scale=True, desc=f"{media_type} {index+1}", leave=False) as pbar:
                        for chunk in r.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                if output_path.exists() and output_path.stat().st_size > 0:
                     downloaded_files.append({
                         'path': str(output_path.resolve()), # Store absolute path
                         'type': media_type,
                         'index': index,
                         'filename': filename
                     })
                     logger.info(f"Successfully downloaded {output_path.name}")
                else:
                     logger.error(f"Download completed but file is empty or missing: {output_path.name}")
                     if output_path.exists(): output_path.unlink() # Clean up empty file

            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {media_type} from {media_url}: {e}")
                # Clean up potentially partial file
                if output_path.exists():
                    try: output_path.unlink()
                    except OSError: pass
            except Exception as e:
                logger.error(f"An unexpected error occurred during download of {media_url}: {e}")
                if output_path.exists():
                     try: output_path.unlink()
                     except OSError: pass


        logger.info(f"Finished download process. Successfully downloaded {len(downloaded_files)} files.")
        return downloaded_files