import re

def extract_links_from_string(results: str) -> list:
    link_pattern = r'link:\s*(https?://[^\s,]+)'
    links = re.findall(link_pattern, results)
    
    return links