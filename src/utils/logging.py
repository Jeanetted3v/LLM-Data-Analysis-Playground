import json
import logging
import logging.config
import os
import sys

import yaml
from pythonjsonlogger import jsonlogger

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
logger = logging.getLogger(__name__)


def setup_logging(
    logging_config_path="config/logging.yaml", default_level=logging.INFO
):
    """Set up configuration for logging utilities.

    Args:
    ----------
    logging_config_path : str, optional
        Path to YAML file containing configuration for Python logger,
        by default "./config/logging_config.yaml"
    default_level : logging object, optional, by default logging.INFO
    """
    try:
        with open(logging_config_path, "rt", encoding="utf-8") as file:
            log_config = yaml.safe_load(file.read())

        # Only add encoding to RotatingFileHandler
        if 'handlers' in log_config:
            for handler in log_config['handlers'].values():
                if (handler['class'] == 'logging.handlers.RotatingFileHandler'
                        and 'encoding' not in handler):
                    handler['encoding'] = 'utf-8'
                    
        logging.config.dictConfig(log_config)
    except Exception as error:
        logging.basicConfig(
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            level=default_level,
        )
        logger.info(error)
        logger.info(
            "Logging config file is not found. Basic config is being used."
        )


class PrettyJSONFormatter(jsonlogger.JsonFormatter):
    """A custom JSON formatter that prettifies JSON log records.

    This formatter extends the `jsonlogger.JsonFormatter` to format log
    records as pretty-printed JSON strings. It attempts to parse any
    string values within the log record as JSON.

    Methods:
    ----------
    __init__(*args, **kwargs): Initializes the PrettyJSONFormatter instance.
    format(record): Formats the log record as a pretty-printed JSON string.
    """

    def __init__(self, *args, **kwargs):
        """Initializes the PrettyJSONFormatter instance.

        Args:
        ----------
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)

    def format(self, record):
        """Format a log record by parsing string values as JSON.

        This method first formats the record using superclass's format method,
        then attempts to parse any string values in the resulting JSON record
        as JSON objects. If a string value cannot be parsed as JSON, it is left
        unchanged.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log record as a JSON string, with keys sorted
            and indented for readability.
        """
        json_record = super().format(record)
        parsed_record = json.loads(json_record)

        # Attempt to parse any string value as JSON
        for key, value in parsed_record.items():
            if isinstance(value, str):
                try:
                    parsed_record[key] = json.loads(value)
                except json.JSONDecodeError:
                    pass  # If it's not JSON, leave it as is
        return json.dumps(parsed_record, indent=2, sort_keys=True)


class UnicodeJsonFormatter(jsonlogger.JsonFormatter):
    """JSON formatter with proper Unicode handling."""
    
    def format(self, record):
        """Formats LogRecord with Unicode support."""
        # Get the base formatting
        json_record = super().format(record)
        # Parse and re-dump with Unicode support
        parsed_record = json.loads(json_record)
        return json.dumps(parsed_record, ensure_ascii=False)