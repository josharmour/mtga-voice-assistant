import requests
import re
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# URL for the official Magic: The Gathering Comprehensive Rules text file
RULES_URL = "https://media.wizards.com/2025/downloads/MagicCompRules%2020250919.txt"
OUTPUT_PATH = "data/MagicCompRules.txt"

def download_rules_file(url: str) -> str:
    """
    Downloads the content of a text file from a URL.

    Args:
        url: The URL to download the file from.

    Returns:
        The content of the file as a string, or None if the download fails.
    """
    logger.info(f"Downloading rules from: {url}")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        # The official rules file is encoded in 'cp1252'
        response.encoding = 'cp1252'
        logger.info("Download successful.")
        return response.text
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download rules file: {e}")
        return None

def clean_rules_text(text: str) -> str:
    """
    Cleans the raw comprehensive rules text by extracting only the lines that
    represent actual rules.

    Args:
        text: The raw text content of the rules file.

    Returns:
        The cleaned text containing only the rules.
    """
    logger.info("Cleaning rules text...")
    
    rule_pattern = re.compile(r'^\d+\.\d+[a-z]*\.\s+.+$')
    cleaned_lines = []
    
    for line in text.splitlines():
        if rule_pattern.match(line):
            cleaned_lines.append(line)
            
    if not cleaned_lines:
        logger.error("No rules found in the text file. The format may have changed.")
        return ""
        
    cleaned_text = "\n".join(cleaned_lines).strip()
    logger.info(f"Extracted {len(cleaned_lines)} rules.")
    return cleaned_text

def save_rules_to_file(text: str, path: str):
    """
    Saves the given text to a file.

    Args:
        text: The text to save.
        path: The path to save the file to.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    logger.info(f"Cleaned rules saved to: {output_path}")

def main():
    """
    Main function to download, clean, and save the MTG comprehensive rules.
    """
    raw_rules = download_rules_file(RULES_URL)
    if raw_rules:
        cleaned_rules = clean_rules_text(raw_rules)
        if cleaned_rules:
            save_rules_to_file(cleaned_rules, OUTPUT_PATH)
            logger.info("Rules processing complete.")
        else:
            logger.error("Failed to process rules: No content after cleaning.")
    else:
        logger.error("Failed to download rules. Aborting.")

if __name__ == "__main__":
    main()