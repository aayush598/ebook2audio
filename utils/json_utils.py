"""
JSON extraction utilities — moved from the original monolithic script.
No behavior or logic changed.
"""

import re


def extract_json(text: str) -> str:
    """Robust JSON extraction that detects List vs Dict priority"""

    # Remove code blocks
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    text = text.strip()
    
    # Find indices of both start brackets
    idx_dict = text.find('{')
    idx_list = text.find('[')
    
    # Determine if we should look for a Dict or a List based on which comes first
    is_dict = False
    
    if idx_dict != -1 and idx_list != -1:
        if idx_dict < idx_list:
            is_dict = True
    elif idx_dict != -1:
        is_dict = True
    # else: Only list exists or neither exists
    
    if is_dict:
        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
    else:
        start = text.find('[')
        end = text.rfind(']')
        if start != -1 and end != -1 and end > start:
            return text[start:end+1]
        
    return text
