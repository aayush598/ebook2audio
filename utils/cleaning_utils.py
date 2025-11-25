"""
Text cleaning utilities for TTS compatibility.
Extracted from the original script without modifications.
"""

import re


def deep_clean_for_tts(text: str) -> str:
    """Deep cleaning for TTS compatibility"""
    text = re.sub(r'\*+', '', text)
    text = re.sub(r'#+', '', text)
    text = re.sub(r'_+', '', text)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'(?i)(panel|scene|दृश्य|पैनल)\s*\d+', '', text)
    text = re.sub(r'(?i)(visual|caption|narrator|कथावाचक):', '', text)
    
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"
        u"\U0001F300-\U0001F5FF"
        u"\U0001F680-\U0001F6FF"
        u"\U0001F1E0-\U0001F1FF"
        u"\U00002500-\U00002BEF"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
        
    text = emoji_pattern.sub(r'', text)
    text = re.sub(r'[=\-_]{3,}', '', text)
    text = re.sub(r'[•·∙‣⁃]', '', text)
    text = re.sub(r'([A-Z][A-Za-z]+):\s*', r'\1 ने कहा - ', text)
    text = re.sub(r'["""\'\'`]', '', text)
    text = re.sub(r'\s+([।,])', r'\1', text)
    text = re.sub(r'([।,])\s*', r'\1 ', text)
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\t+', ' ', text)
    
    return text.strip()
