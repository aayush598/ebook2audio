"""
PDF to Detailed Hinglish Audiobook Converter
Uses Agno framework with Gemini model to extract and create detailed summaries,
then converts them to high-quality audio using Kokoro TTS with proper Hindi/English text.
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple
import PyPDF2
from agno.agent import Agent
from agno.models.google import Gemini
from misaki.espeak import EspeakG2P
from kokoro_onnx import Kokoro
from huggingface_hub import snapshot_download
import soundfile as sf
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Configuration
KOKORO_REPO_ID = "leonelhs/kokoro-thewh1teagle"
OUTPUT_DIR = "audiobook_output"
TEMP_DIR = "temp_chapters"
MIN_SUBTOPIC_DURATION = 5  # Minimum minutes per subtopic
TARGET_WORDS_PER_MINUTE = 150  # Speaking rate

VOICES = {
    'female_alpha': 'hf_alpha',
    'female_beta': 'hf_beta',
    'male_omega': 'hm_omega',
    'male_psi': 'hm_psi'
}


class PDFToAudiobookConverter:
    """Convert PDF books to detailed Hinglish audio with proper Hindi/English text"""
    
    def __init__(self, gemini_api_key: str, voice: str = 'hf_alpha', speed: float = 1.0):
        """
        Initialize the converter
        
        Args:
            gemini_api_key: Google Gemini API key
            voice: Voice profile for TTS (default: 'hf_alpha')
            speed: Speech speed multiplier (default: 1.0)
        """
        self.voice = voice
        self.speed = speed
        
        # Initialize Agno Agent with Gemini for detailed analysis
        self.agent = Agent(
            name="Detailed Book Analyzer",
            model=Gemini(id="gemini-2.0-flash-lite", api_key=gemini_api_key),
            instructions="""You are an expert educational content creator for Indian audiences.
            
            Your task is to:
            1. Analyze book chapters in-depth and break them into logical subtopics
            2. Create detailed, comprehensive explanations for each subtopic
            3. Write in a mix of Hindi (Devanagari script: हिंदी) and English
            4. Use Hindi for common conversational words and explanations
            5. Use English for technical terms, names, and specific concepts
            6. Make content detailed enough for 5-10 minutes of audio per subtopic
            7. Include examples, explanations, and context
            8. Write in a conversational, engaging style
            
            CRITICAL FORMATTING RULES:
            - Write Hindi text in Devanagari script (हिंदी में लिखें)
            - Write English text in Latin script (English words in English)
            - Do NOT transliterate Hindi to Roman (avoid "aap", "hain" etc.)
            - Use proper Hindi: आप, हैं, का, की, को, से, etc.
            - Mix naturally: "यह chapter बहुत important है"
            
            Example good output:
            "इस chapter में हम देखेंगे कि artificial intelligence कैसे काम करती है। 
            सबसे पहले, हमें यह समझना होगा कि machine learning क्या है।"
            
            Example bad output (DO NOT DO THIS):
            "Is chapter mein hum dekhenge ki artificial intelligence kaise kaam karti hai."
            """,
            markdown=False,
        )
        
        # Initialize TTS components
        self._init_tts()
        
        # Create output directories
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        Path(TEMP_DIR).mkdir(exist_ok=True)
    
    def _init_tts(self):
        """Initialize Kokoro TTS system"""
        print("Downloading Kokoro TTS model...")
        snapshot = snapshot_download(repo_id=KOKORO_REPO_ID)
        
        # Initialize G2P for Hindi
        self.g2p = EspeakG2P(language="hi")
        
        # Initialize Kokoro
        model_path = os.path.join(snapshot, "kokoro-v1.0.onnx")
        voices_path = os.path.join(snapshot, "voices-v1.0.bin")
        self.kokoro = Kokoro(model_path, voices_path)
        print("TTS system initialized!")
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract all text from PDF"""
        print(f"Extracting text from {pdf_path}...")
        text = ""
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            total_pages = len(pdf_reader.pages)
            
            for i, page in enumerate(pdf_reader.pages):
                print(f"Processing page {i+1}/{total_pages}")
                text += page.extract_text() + "\n\n"
        
        return text
    
    def split_into_chapters(self, text: str) -> List[Dict[str, str]]:
        """Split book text into chapters using Gemini"""
        print("Analyzing book structure and splitting into chapters...")
        
        chapter_prompt = f"""Analyze this book text and identify all chapters.
        
        Return ONLY a JSON array with this structure:
        [
            {{"chapter_num": 1, "title": "Chapter Title", "start_marker": "text that marks start"}},
            ...
        ]
        
        Book text (first 30000 characters):
        {text[:30000]}
        """
        
        response = self.agent.run(chapter_prompt)
        
        try:
            chapter_markers = json.loads(response.content)
            chapters = self._extract_chapter_content(text, chapter_markers)
            print(f"Identified {len(chapters)} chapters")
            return chapters
        except json.JSONDecodeError:
            print("Using fallback chapter detection...")
            return self._fallback_chapter_split(text)
    
    def _extract_chapter_content(self, text: str, markers: List[Dict]) -> List[Dict]:
        """Extract full chapter content based on markers"""
        chapters = []
        text_lower = text.lower()
        
        for i, marker in enumerate(markers):
            start_marker = marker.get('start_marker', '').lower()
            
            # Find chapter start
            start_idx = text_lower.find(start_marker) if start_marker else 0
            
            # Find chapter end (start of next chapter or end of book)
            if i < len(markers) - 1:
                next_marker = markers[i + 1].get('start_marker', '').lower()
                end_idx = text_lower.find(next_marker, start_idx + 100) if next_marker else len(text)
            else:
                end_idx = len(text)
            
            chapter_content = text[start_idx:end_idx].strip()
            
            if chapter_content:
                chapters.append({
                    "chapter_num": marker['chapter_num'],
                    "title": marker['title'],
                    "content": chapter_content
                })
        
        return chapters
    
    def _fallback_chapter_split(self, text: str) -> List[Dict[str, str]]:
        """Fallback method to split chapters"""
        patterns = [
            r'\n\s*Chapter\s+(\d+)[:\s]+([^\n]+)',
            r'\n\s*CHAPTER\s+(\d+)[:\s]+([^\n]+)',
            r'\n\s*अध्याय\s+(\d+)[:\s]+([^\n]+)',
        ]
        
        chapters = []
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            if len(matches) > 1:
                for i, match in enumerate(matches):
                    start = match.start()
                    end = matches[i + 1].start() if i < len(matches) - 1 else len(text)
                    
                    chapters.append({
                        "chapter_num": int(match.group(1)),
                        "title": match.group(2).strip(),
                        "content": text[start:end].strip()
                    })
                break
        
        if not chapters:
            # Split by size
            chunk_size = 5000
            for i in range(0, len(text), chunk_size):
                chapters.append({
                    "chapter_num": i // chunk_size + 1,
                    "title": f"Part {i // chunk_size + 1}",
                    "content": text[i:i+chunk_size]
                })
        
        return chapters
    
    def break_into_subtopics(self, chapter_content: str, chapter_title: str) -> List[Dict[str, str]]:
        """
        Break chapter into detailed subtopics using Gemini
        
        Args:
            chapter_content: Full chapter text
            chapter_title: Chapter title
            
        Returns:
            List of subtopic dictionaries with title and detailed content
        """
        print(f"Breaking '{chapter_title}' into detailed subtopics...")
        
        prompt = f"""Chapter: {chapter_title}

Content:
{chapter_content[:15000]}

Break this chapter into 4-8 logical subtopics. For EACH subtopic:
1. Give it a clear title
2. Write a DETAILED explanation (minimum 750-1000 words per subtopic)
3. Include examples, context, and thorough explanations
4. Write in Hindi (Devanagari: हिंदी) mixed with English
5. Use Hindi for conversational parts, English for technical terms
6. Make it engaging and educational

Return ONLY a JSON array:
[
    {{
        "subtopic_num": 1,
        "title": "Subtopic Title",
        "content": "बहुत detailed explanation यहाँ लिखें... (minimum 750 words)"
    }},
    ...
]

REMEMBER: Hindi in Devanagari (हिंदी), English in English. NO Roman transliteration!"""
        
        response = self.agent.run(prompt)
        
        try:
            subtopics = json.loads(response.content)
            
            # Validate subtopic length
            for subtopic in subtopics:
                word_count = len(subtopic['content'].split())
                if word_count < 500:
                    # Expand short subtopics
                    subtopic['content'] = self._expand_subtopic(
                        subtopic['title'], 
                        subtopic['content'],
                        chapter_title
                    )
            
            return subtopics
        except json.JSONDecodeError:
            print("Error parsing subtopics, using fallback...")
            return self._create_detailed_summary(chapter_content, chapter_title)
    
    def _expand_subtopic(self, subtopic_title: str, content: str, chapter_title: str) -> str:
        """Expand a subtopic to meet minimum length requirements"""
        print(f"Expanding subtopic: {subtopic_title}")
        
        expansion_prompt = f"""Chapter: {chapter_title}
Subtopic: {subtopic_title}

Current content:
{content}

This content is too brief. Expand it to at least 800-1000 words by:
1. Adding detailed explanations
2. Including relevant examples
3. Providing context and background
4. Explaining concepts thoroughly
5. Adding practical applications

Write in Hindi (हिंदी Devanagari) mixed with English.
Hindi for explanations, English for technical terms.

Return ONLY the expanded content (no JSON, just the text):"""
        
        response = self.agent.run(expansion_prompt)
        return response.content
    
    def _create_detailed_summary(self, content: str, chapter_title: str) -> List[Dict[str, str]]:
        """Fallback: Create one detailed summary if subtopic extraction fails"""
        print("Creating detailed summary as fallback...")
        
        prompt = f"""Chapter: {chapter_title}

Content:
{content[:10000]}

Create a VERY DETAILED summary of this chapter (minimum 1500 words).
Write in Hindi (Devanagari: हिंदी) mixed with English.
Be comprehensive and educational.

Return only the detailed summary text:"""
        
        response = self.agent.run(prompt)
        
        return [{
            "subtopic_num": 1,
            "title": chapter_title,
            "content": response.content
        }]
    
    def text_to_speech(self, text: str, output_path: str):
        print(f"Generating audio: {output_path}")
        print(f"Text length: {len(text)} characters, {len(text.split())} words")

        # Convert full text → phonemes
        phonemes, _ = self.g2p(text)

        # PHONEME LIMIT FIX
        MAX_PHONEMES = 480   # keep below Kokoro limit 510
        chunks = [phonemes[i:i + MAX_PHONEMES] for i in range(0, len(phonemes), MAX_PHONEMES)]

        all_samples = []
        
        for idx, chunk in enumerate(chunks):
            print(f"Processing chunk {idx+1}/{len(chunks)} (size: {len(chunk)} phonemes)")

            samples, sample_rate = self.kokoro.create(
                chunk,
                self.voice,
                self.speed,
                is_phonemes=True
            )

            all_samples.append(samples)

        # Concatenate audio chunks
        full_audio = np.concatenate(all_samples)

        sf.write(output_path, full_audio, sample_rate)

        duration_minutes = len(full_audio) / sample_rate / 60
        print(f"Audio saved: {output_path} (Duration: {duration_minutes:.2f} minutes)")

    def convert_book(self, pdf_path: str, book_name: str = None):
        """
        Complete pipeline: PDF -> Chapters -> Subtopics -> Detailed Audio
        
        Args:
            pdf_path: Path to PDF file
            book_name: Name for output files (optional)
        """
        if book_name is None:
            book_name = Path(pdf_path).stem
        
        print(f"\n{'='*60}")
        print(f"Converting: {book_name}")
        print(f"{'='*60}\n")
        
        # Step 1: Extract text
        full_text = self.extract_text_from_pdf(pdf_path)
        
        # Step 2: Split into chapters
        chapters = self.split_into_chapters(full_text)
        
        # Step 3: Process each chapter with detailed subtopics
        all_audio_files = []
        
        for chapter in chapters:
            chapter_num = chapter['chapter_num']
            chapter_title = chapter['title']
            chapter_content = chapter['content']
            
            print(f"\n{'='*60}")
            print(f"Chapter {chapter_num}: {chapter_title}")
            print(f"{'='*60}\n")
            
            # Break into detailed subtopics
            subtopics = self.break_into_subtopics(chapter_content, chapter_title)
            
            print(f"Generated {len(subtopics)} subtopics for this chapter\n")
            
            # Process each subtopic
            for subtopic in subtopics:
                subtopic_num = subtopic['subtopic_num']
                subtopic_title = subtopic['title']
                subtopic_content = subtopic['content']
                
                print(f"--- Subtopic {chapter_num}.{subtopic_num}: {subtopic_title} ---")
                
                # Save subtopic text
                text_file = os.path.join(
                    TEMP_DIR, 
                    f"{book_name}_ch{chapter_num:02d}_sub{subtopic_num:02d}.txt"
                )
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(f"Chapter {chapter_num}: {chapter_title}\n")
                    f.write(f"Subtopic {subtopic_num}: {subtopic_title}\n\n")
                    f.write(subtopic_content)
                
                # Generate audio
                audio_file = os.path.join(
                    OUTPUT_DIR, 
                    f"{book_name}_ch{chapter_num:02d}_sub{subtopic_num:02d}.wav"
                )
                self.text_to_speech(subtopic_content, audio_file)
                all_audio_files.append(audio_file)
                
                print()
        
        print(f"\n{'='*60}")
        print(f"✅ Conversion complete!")
        print(f"Generated {len(all_audio_files)} detailed audio segments")
        print(f"Output directory: {OUTPUT_DIR}")
        print(f"Text files saved in: {TEMP_DIR}")
        print(f"{'='*60}\n")
        
        return all_audio_files


def main():
    """Main execution function"""
    
    # Configuration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    PDF_PATH = "PlayingtoWinPDF.pdf"  # Change this to your PDF path
    
    if not GEMINI_API_KEY:
        print("Error: GEMINI_API_KEY environment variable not set!")
        print("Set it using: export GEMINI_API_KEY='your-api-key-here'")
        return
    
    if not os.path.exists(PDF_PATH):
        print(f"Error: PDF file not found: {PDF_PATH}")
        return
    
    # Initialize converter
    converter = PDFToAudiobookConverter(
        gemini_api_key=GEMINI_API_KEY,
        voice='hf_alpha',  # Female voice
        speed=1.0  # Normal speed
    )
    
    # Convert the book
    audio_files = converter.convert_book(PDF_PATH)
    
    print("\n📊 Summary:")
    print(f"Total audio segments: {len(audio_files)}")
    print("\nGenerated audio files:")
    for audio_file in audio_files:
        print(f"  - {audio_file}")


if __name__ == "__main__":
    main()