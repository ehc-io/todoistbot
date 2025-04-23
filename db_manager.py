import sqlite3
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class URLDatabase:
    """Manages a SQLite database for tracking processed URLs to avoid duplication."""
    
    def __init__(self, db_path="db/processed_urls.db"):
        """Initialize the database connection and create table if it doesn't exist."""
        self.db_path = db_path
        self.conn = None
        self.cursor = None
        self._initialize_db()
    
    def _initialize_db(self):
        """Set up the database connection and create tables if they don't exist."""
        try:
            # Create directory for database if it doesn't exist
            db_dir = os.path.dirname(self.db_path)
            if db_dir and not os.path.exists(db_dir):
                os.makedirs(db_dir)
                
            # Connect to database
            self.conn = sqlite3.connect(self.db_path)
            self.cursor = self.conn.cursor()
            
            # Create processed_urls table if it doesn't exist
            self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS processed_urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE NOT NULL,
                task_id TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                success INTEGER DEFAULT 1
            )
            ''')
            
            # Create index on URL column for faster lookups
            self.cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_url ON processed_urls(url)
            ''')
            
            self.conn.commit()
            logger.info(f"Database initialized at {self.db_path}")
            
        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            if self.conn:
                self.conn.close()
            raise
    
    def url_exists(self, url):
        """Check if a URL has been successfully processed before.
        
        Args:
            url (str): The URL to check
            
        Returns:
            bool: True if the URL exists in the database and was processed successfully, False otherwise
        """
        if not self.cursor:
            self._initialize_db()
            
        try:
            self.cursor.execute("SELECT success FROM processed_urls WHERE url = ?", (url,))
            result = self.cursor.fetchone()
            # Return True only if URL exists and was processed successfully
            return result is not None and result[0] == 1
        except sqlite3.Error as e:
            logger.error(f"Error checking URL existence: {e}")
            return False
    
    def add_url(self, url, task_id=None, success=True):
        """Add a URL to the database after processing.
        
        Args:
            url (str): The URL that was processed
            task_id (str, optional): The ID of the Todoist task
            success (bool, optional): Whether processing was successful
            
        Returns:
            bool: True if added successfully, False otherwise
        """
        if not self.cursor:
            self._initialize_db()
            
        try:
            # Convert boolean success to integer (1 or 0)
            success_int = 1 if success else 0
            
            # Check if URL already exists
            self.cursor.execute("SELECT id FROM processed_urls WHERE url = ?", (url,))
            existing_row = self.cursor.fetchone()
            
            if existing_row:
                # Update existing record
                self.cursor.execute(
                    "UPDATE processed_urls SET task_id = ?, processed_at = ?, success = ? WHERE url = ?",
                    (task_id, datetime.now().isoformat(), success_int, url)
                )
            else:
                # Insert new record
                self.cursor.execute(
                    "INSERT INTO processed_urls (url, task_id, processed_at, success) VALUES (?, ?, ?, ?)",
                    (url, task_id, datetime.now().isoformat(), success_int)
                )
                
            self.conn.commit()
            return True
        except sqlite3.Error as e:
            logger.error(f"Error adding URL to database: {e}")
            return False
    
    def get_stats(self):
        """Get statistics about processed URLs.
        
        Returns:
            dict: Statistics about processed URLs
        """
        if not self.cursor:
            self._initialize_db()
            
        try:
            # Total URLs
            self.cursor.execute("SELECT COUNT(*) FROM processed_urls")
            total = self.cursor.fetchone()[0]
            
            # Successful URLs
            self.cursor.execute("SELECT COUNT(*) FROM processed_urls WHERE success = 1")
            successful = self.cursor.fetchone()[0]
            
            # Failed URLs
            self.cursor.execute("SELECT COUNT(*) FROM processed_urls WHERE success = 0")
            failed = self.cursor.fetchone()[0]
            
            return {
                "total": total,
                "successful": successful,
                "failed": failed
            }
        except sqlite3.Error as e:
            logger.error(f"Error getting URL statistics: {e}")
            return {"total": 0, "successful": 0, "failed": 0}
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            self.cursor = None
    
    def __del__(self):
        """Ensure connection is closed when object is destroyed."""
        self.close() 