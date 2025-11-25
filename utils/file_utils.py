"""
File handling utilities extracted from the original monolithic script.
No changes to core functionality or logic.
"""

import os


def save_chapter_ending(chapter_num: int, content: str, context_dir: str):
    """Save last few paragraphs for next chapter's context"""
    
    # Get last 500-800 words
    words = content.split()
    ending_words = words[-800:] if len(words) > 800 else words[-500:]
    ending = ' '.join(ending_words)
    
    filepath = os.path.join(context_dir, f"ch{chapter_num:03d}_ending.txt")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(ending)


def read_previous_ending(chapter_num: int, context_dir: str):
    """Read last chapter's ending if exists"""
    
    prev_ch = chapter_num - 1
    prev_file = os.path.join(context_dir, f"ch{prev_ch:03d}_ending.txt")
    
    if os.path.exists(prev_file):
        with open(prev_file, 'r', encoding='utf-8') as f:
            return f.read()
    
    return ""
