import logging
import os
from datetime import datetime
from typing import Optional


class PerMessageFileHandler(logging.Handler):
    """Handler that writes request/response pairs to the same file.

    Useful for logging LLM requests/responses where each interaction
    (request + response) should be stored in a single file for analysis.

    Behavior:
    - Requests create a new file and are written immediately
    - Responses are appended to the matching request file
    - Matching is done by session_id if available, otherwise by sequential order

    Args:
        log_dir: Directory where individual log files will be written
        filename_template: Template for filename. Can use {timestamp}, {counter}, {session_id}
        file_extension: Extension for log files (default: .txt)
    """

    def __init__(
        self,
        log_dir: str,
        filename_template: str = "prompt_{timestamp}_{counter}",
        file_extension: str = ".txt",
    ):
        super().__init__()
        self.log_dir = log_dir
        self.filename_template = filename_template
        self.file_extension = file_extension
        self.counter = 0
        # Track the most recent request file to append responses
        # For async/concurrent scenarios, use session_id-based tracking
        self.pending_requests = {}  # {session_id or counter: filepath}
        self.last_request_file = None  # Fallback for sequential requests

        # Ensure log directory exists
        os.makedirs(log_dir, exist_ok=True)

    def emit(self, record: logging.LogRecord):
        """Write log record to file. Requests create new files, responses append to request file."""
        try:
            msg = self.format(record)
            is_request = ">>> LLM Request >>>" in msg
            is_response = "<<< LLM Response <<<" in msg

            session_id = getattr(record, "session_id", None)

            if is_request:
                # Create new file for request
                self.counter += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[
                    :-3
                ]  # milliseconds

                # Build filename from template
                filename = self.filename_template.format(
                    timestamp=timestamp,
                    counter=self.counter,
                    session_id=session_id or "",
                )
                # Add session_id to filename if present, then add extension
                if session_id:
                    filename = f"{filename}_{session_id}"
                filename = f"{filename}{self.file_extension}"

                filepath = os.path.join(self.log_dir, filename)

                # Write request to file
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(msg)

                # Track this file for the matching response
                if session_id:
                    self.pending_requests[f"session_{session_id}"] = filepath
                else:
                    # Use counter as fallback key
                    key = f"counter_{self.counter}"
                    self.pending_requests[key] = filepath
                self.last_request_file = filepath

            elif is_response:
                # Find matching request file and append response
                filepath = None

                # Try to find by session_id first
                if session_id:
                    filepath = self.pending_requests.get(f"session_{session_id}")
                    if filepath:
                        del self.pending_requests[f"session_{session_id}"]

                # Fallback to last request file (for sequential requests without session_id)
                if (
                    not filepath
                    and self.last_request_file
                    and os.path.exists(self.last_request_file)
                ):
                    filepath = self.last_request_file
                    # Clean up from pending if it's there
                    for key, path in list(self.pending_requests.items()):
                        if path == filepath:
                            del self.pending_requests[key]
                            break
                    self.last_request_file = None

                if filepath and os.path.exists(filepath):
                    # Append response to request file
                    with open(filepath, "a", encoding="utf-8") as f:
                        f.write("\n\n")
                        f.write(msg)
                else:
                    # No matching request found, create new file with both
                    self.counter += 1
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    filename = self.filename_template.format(
                        timestamp=timestamp,
                        counter=self.counter,
                        session_id=session_id or "",
                    )
                    if session_id:
                        filename = f"{filename}_{session_id}"
                    filename = f"{filename}{self.file_extension}"
                    filepath = os.path.join(self.log_dir, filename)
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(f"[Response only - no matching request found]\n\n")
                        f.write(msg)
            else:
                # Unknown message type, create new file
                self.counter += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                filename = self.filename_template.format(
                    timestamp=timestamp,
                    counter=self.counter,
                    session_id=session_id or "",
                )
                if session_id:
                    filename = f"{filename}_{session_id}"
                filename = f"{filename}{self.file_extension}"
                filepath = os.path.join(self.log_dir, filename)
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(msg)

        except Exception:
            # Don't let logging errors break the application
            self.handleError(record)
