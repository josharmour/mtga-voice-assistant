import logging
import os
import time
from typing import Callable

class LogFollower:
    """Follows the Arena Player.log file and yields new lines as they're added."""
    def __init__(self, log_path: str):
        self.log_path = log_path
        self.file = None
        self.inode = None
        self.offset = 0
        self.first_open = True  # Track if this is first time opening

    def follow(self, callback: Callable[[str], None]):
        """Follow the log file indefinitely, calling the callback for each new line."""
        print(f"[DEBUG] LogFollower.follow() started! Watching: {self.log_path}")
        logging.info(f"LogFollower.follow() started! Watching: {self.log_path}")
        while True:
            try:
                current_inode = None
                try:
                    current_inode = os.stat(self.log_path).st_ino
                except FileNotFoundError:
                    logging.warning(f"Log file not found at {self.log_path}. Waiting...")
                    time.sleep(5)
                    continue # Skip to next iteration if file not found
                except Exception as e:
                    logging.error(f"Error getting inode for {self.log_path}: {e}")
                    time.sleep(1)
                    continue

                logging.debug(f"LogFollower: current_inode={current_inode}, self.inode={self.inode}")
                if self.inode is None or self.inode != current_inode:
                    print(f"[DEBUG] Opening log file (inode changed or first open)")
                    if self.file:
                        self.file.close()
                    self.file = open(self.log_path, 'r', encoding='utf-8', errors='replace')
                    self.inode = current_inode
                    print(f"[DEBUG] File opened successfully, inode={self.inode}")

                    # On first open, seek to end to ignore old matches
                    # On log rotation, start from beginning of new file
                    if self.first_open:
                        # First time opening - go to end
                        self.file.seek(0, 2)  # Seek to end of file
                        self.offset = self.file.tell()
                        self.first_open = False
                        print(f"[DEBUG] First open - seeked to end, offset={self.offset}")
                        logging.info("Log file opened - starting from end (ignoring old matches).")
                    else:
                        # Log rotation - start from beginning of new file
                        self.offset = 0
                        logging.info("Log file rotated - starting from beginning of new file.")

                self.file.seek(self.offset)
                line_count = 0
                while True:
                    line = self.file.readline()
                    if not line:
                        break
                    line_count += 1
                    self.offset = self.file.tell()
                    stripped_line = line.strip()

                    # Debug: show what we're reading
                    if "Draft" in stripped_line or "BotDraft" in stripped_line or "<==" in stripped_line or "==>" in stripped_line:
                        print(f"[DEBUG] Draft-related line: {stripped_line[:150]}")

                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        logging.debug(f"Read line: {stripped_line[:100]}...") # Log first 100 chars to avoid spam
                    callback(stripped_line)
                if line_count > 0:
                    print(f"[DEBUG] Processed {line_count} lines")
                time.sleep(0.05)
            except FileNotFoundError:
                logging.warning(f"Log file not found at {self.log_path}. Waiting...")
                time.sleep(5)
            except Exception as e:
                logging.error(f"Error following log file: {e}")
                time.sleep(1)

    def close(self):
        if self.file:
            self.file.close()
