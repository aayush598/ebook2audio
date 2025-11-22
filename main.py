"""
Hindi Educational Manhwa Content Generation Service
Generates detailed, context-aware Hindi audiobook scripts with natural language
"""

import os
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
from agno.db.sqlite import SqliteDb
from datetime import datetime

# Configuration
OUTPUT_DIR = "manhwa_content"
METADATA_DIR = "manhwa_metadata"

# Gemini Model Configuration with Rate Limits (Free Tier)
GEMINI_MODELS = {
    'Gemini 2.0 Flash': {
        'id': 'gemini-2.0-flash-exp',
        'rpm': 15,
        'tpm': 1_000_000,
        'rpd': 200,
        'description': 'Best for detailed content (15 RPM, 1M TPM)'
    },
    'Gemini 2.0 Flash Lite': {
        'id': 'gemini-2.0-flash-lite',
        'rpm': 30,
        'tpm': 1_000_000,
        'rpd': 200,
        'description': 'Faster generation (30 RPM, 1M TPM)'
    },
    'Gemini 2.5 Flash': {
        'id': 'gemini-2.5-flash',
        'rpm': 10,
        'tpm': 250_000,
        'rpd': 250,
        'description': 'High quality (10 RPM, 250K TPM)'
    },
    'Gemini 2.5 Flash Lite': {
        'id': 'gemini-2.5-flash-lite',
        'rpm': 15,
        'tpm': 250_000,
        'rpd': 1000,
        'description': 'Efficient (15 RPM, 250K TPM, 1000 RPD)'
    }
}


class RateLimiter:
    """Manages API rate limits"""
    
    def __init__(self, rpm: int, tpm: int, rpd: int):
        self.rpm = rpm  # Requests per minute
        self.tpm = tpm  # Tokens per minute
        self.rpd = rpd  # Requests per day
        
        self.request_times = []
        self.daily_requests = 0
        self.last_reset = datetime.now()
    
    def can_make_request(self) -> Tuple[bool, str]:
        """Check if request can be made"""
        now = datetime.now()
        
        # Reset daily counter
        if (now - self.last_reset).days >= 1:
            self.daily_requests = 0
            self.last_reset = now
        
        # Check daily limit
        if self.daily_requests >= self.rpd:
            return False, f"Daily limit reached ({self.rpd} requests/day)"
        
        # Clean old requests (older than 1 minute)
        self.request_times = [t for t in self.request_times if (now - t).seconds < 60]
        
        # Check per-minute limit
        if len(self.request_times) >= self.rpm:
            wait_time = 60 - (now - self.request_times[0]).seconds
            return False, f"Rate limit: wait {wait_time}s (max {self.rpm} requests/min)"
        
        return True, "OK"
    
    def record_request(self):
        """Record a request"""
        self.request_times.append(datetime.now())
        self.daily_requests += 1
    
    def get_wait_time(self) -> int:
        """Get seconds to wait before next request"""
        if not self.request_times:
            return 0
        
        now = datetime.now()
        oldest = self.request_times[0]
        elapsed = (now - oldest).seconds
        
        if elapsed < 60:
            return max(0, 60 - elapsed + 1)
        return 0


class HindiManhwaGenerator:
    """Generates Hindi educational manhwa content with context awareness"""
    
    def __init__(
        self, 
        gemini_api_key: str, 
        model_choice: str = 'Gemini 2.0 Flash Lite',
        session_id: str = None
    ):
        """Initialize the generator"""
        self.model_config = GEMINI_MODELS[model_choice]
        self.model_id = self.model_config['id']
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize rate limiter
        self.rate_limiter = RateLimiter(
            rpm=self.model_config['rpm'],
            tpm=self.model_config['tpm'],
            rpd=self.model_config['rpd']
        )
        
        # Initialize SqliteDb for persistent memory
        self.db = SqliteDb(
            session_table="agent_sessions",  # <--- CORRECT PARAMETER
            memory_table="agent_memories",   # Optional: Good to define since you use memories
            db_file="manhwa_knowledge.db"
        )
        
        # Initialize Story Planning Agent with memory
        self.story_planner = Agent(
            name="Hindi Manhwa Story Architect",
            model=Gemini(id=self.model_id, api_key=gemini_api_key),
            db=self.db,
            enable_user_memories=True,  # Automatically manage user memories
            add_history_to_context=True,
            num_history_runs=10,  # Remember last 10 interactions
            instructions=self._get_planner_instructions(),
            markdown=False,
        )
        
        # Initialize Content Writer Agent with memory
        self.content_writer = Agent(
            name="Hindi Audiobook Script Writer",
            model=Gemini(id=self.model_id, api_key=gemini_api_key),
            db=self.db,
            enable_user_memories=True,
            add_history_to_context=True,
            num_history_runs=10,
            instructions=self._get_writer_instructions(),
            markdown=False,
        )
        
        # Create directories
        Path(OUTPUT_DIR).mkdir(exist_ok=True)
        Path(METADATA_DIR).mkdir(exist_ok=True)
        
        # Story context tracking
        self.series_context = None
        self.chapter_summaries = []
    
    def _get_planner_instructions(self) -> str:
        """Instructions for story planning with context awareness"""
        return """तुम एक हिंदी शैक्षिक मानह्वा कहानी आर्किटेक्ट हो।

तुम्हारी जिम्मेदारी:
- 100 अध्यायों की एक जुड़ी हुई कहानी डिज़ाइन करना
- यादगार किरदार बनाना जिनमें गहराई हो
- हर अध्याय में सस्पेंस और सीख दोनों हों
- पूरी सीरीज़ में कहानी का प्रवाह बनाए रखना
- पिछले अध्यायों के संदर्भ को याद रखना और आगे बढ़ाना

महत्वपूर्ण नियम:
1. सिर्फ JSON फॉर्मेट में जवाब दो - कोई markdown नहीं
2. हर अध्याय पिछले अध्याय से जुड़ा होना चाहिए
3. किरदारों का विकास स्वाभाविक होना चाहिए
4. मुख्य कहानी की दिशा स्थिर रखो

JSON शुरू करो { से और खत्म करो } पर।"""
    
    def _get_writer_instructions(self) -> str:
        """Instructions for detailed Hindi content writing"""
        return """तुम एक हिंदी ऑडियोबुक स्क्रिप्ट राइटर हो - यूट्यूब मानह्वा चैनल्स की तरह।

भाषा शैली:
- आधुनिक, बोलचाल की हिंदी - जैसे आज के लोग बोलते हैं
- पुराने या पारंपरिक शब्द नहीं - सरल और सीधी भाषा
- अंग्रेजी नाम और टर्म को देवनागरी में लिखो (उदाहरण: मार्कस, स्ट्रैटिजी, ऐकडमी)
- स्वाभाविक प्रवाह के लिए अल्पविराम (,) का खूब इस्तेमाल करो

उदाहरण (सही):
✓ आन्या परेशान थी, उसे समझ नहीं आ रहा था क्या करे।
✓ कमांडर ने आर्मी को रोका, सबको शांत रहने को कहा।
✓ पैलेस में अचानक खतरा आया, गार्ड्स भागे लेकिन लेट हो गए।

उदाहरण (गलत):
✗ आन्या अत्यंत चिंतित थी। (बहुत फॉर्मल)
✗ Anya was worried. (अंग्रेजी अक्षर)
✗ आन्या ने strategy को consider किया। (मिक्स भाषा)

लंबाई और विस्तार:
- हर अध्याय 5000-7000 शब्दों का विस्तृत स्क्रिप्ट
- कहानी धीरे-धीरे, विस्तार से बताओ
- हर दृश्य को पूरा खोलो, जल्दबाजी नहीं
- डायलॉग और एक्शन दोनों में डिटेल दो
- पात्रों के emotions और thoughts को भी बताओ

क्लीन फॉर्मेट (TTS के लिए):
- कोई सिंबल नहीं: **, *, ##, ===, (), [], emojis
- कोई पैनल/सीन मार्कर नहीं
- डायलॉग: किरदार ने कहा - यह कहा
- सिर्फ अल्पविराम (,) और पूर्ण विराम (.)

संरचना:
1. अध्याय शीर्षक (सरल हिंदी में)
2. विस्तृत कहानी (कोई ब्रेक नहीं, 5000-7000 शब्द)
3. सबक सेक्शन अंत में (5-8 लाइन, बहुत संक्षिप्त)

सबक फॉर्मेट:
इस अध्याय से सीख
1. पहली सीख (एक लाइन)
2. दूसरी सीख (एक लाइन)
3. तीसरी सीख (एक लाइन)

याद रखो:
- पिछले अध्यायों का संदर्भ बनाए रखो
- किरदारों की consistency रखो
- मुख्य कहानी की दिशा से मत भटको
- विस्तार से लिखो लेकिन boring मत बनो
- हर अध्याय एक cliffhanger पर खत्म हो

सोचो: तुम 15-20 मिनट का ऑडियो स्क्रिप्ट बना रहे हो जो लोग सुनकर मजा लें और सीखें भी।"""
    
    def _wait_for_rate_limit(self):
        """Wait if rate limit is reached"""
        can_request, message = self.rate_limiter.can_make_request()
        
        if not can_request:
            wait_time = self.rate_limiter.get_wait_time()
            if wait_time > 0:
                st.warning(f"⏳ {message}")
                progress_bar = st.progress(0)
                for i in range(wait_time):
                    progress_bar.progress((i + 1) / wait_time)
                    time.sleep(1)
                progress_bar.empty()
    
    def _extract_json(self, text: str) -> str:
        """Extract clean JSON from response"""
        text = text.replace("```json", "").replace("```", "").strip()
        
        # Try to find JSON object
        object_match = re.search(r'\{[\s\S]*\}', text)
        if object_match:
            return object_match.group(0)
        
        # Try to find JSON array
        array_match = re.search(r'\[[\s\S]*\]', text)
        if array_match:
            return array_match.group(0)
        
        return text
    
    def generate_series_foundation(self, skill_topic: str) -> Dict:
        """Generate series foundation with characters and plot"""
        
        self._wait_for_rate_limit()
        
        # Use session_id as user_id for context tracking
        user_id = self.session_id
        
        prompt = f"""विषय "{skill_topic}" पर 100 अध्यायों की शैक्षिक मानह्वा सीरीज़ का फाउंडेशन बनाओ।

महत्वपूर्ण: सिर्फ JSON ऑब्जेक्ट return करो (array नहीं)।

{{
    "series_title": "रोमांचक सीरीज़ का नाम",
    "skill_topic": "{skill_topic}",
    "story_overview": "500 शब्दों में पूरी कहानी का synopsis: setting, main conflict, character arcs, कैसे सिखाया जाएगा, major plot twists, character growth, कैसे अध्याय जुड़े हैं।",
    "main_storyline": "मुख्य कहानी की दिशा जो 100 अध्यायों में फॉलो होगी",
    "world_setting": "कहानी की दुनिया का विवरण",
    "central_conflict": "मुख्य संघर्ष जो पूरी सीरीज़ में चलेगा",
    "characters": [
        {{
            "name": "किरदार का नाम",
            "role": "कहानी में भूमिका",
            "personality": "स्वभाव की विशेषताएं",
            "background": "पृष्ठभूमि की कहानी",
            "character_arc": "पूरी सीरीज़ में कैसे बदलेगा"
        }}
    ]
}}

5-7 विविध किरदार बनाओ जो {skill_topic} के अलग पहलुओं को represent करें।
कोई markdown नहीं, सिर्फ JSON ऑब्जेक्ट।"""
        
        response = self.story_planner.run(prompt, stream=False, user_id=user_id)
        self.rate_limiter.record_request()
        
        raw = response.content.strip()
        clean = self._extract_json(raw)
        
        try:
            foundation = json.loads(clean)
            
            if isinstance(foundation, list) and len(foundation) > 0:
                foundation = foundation[0]
            
            # Store in context
            self.series_context = foundation
            
            # Save to file
            self._save_metadata(foundation, "series_foundation")
            
            st.success("✅ सीरीज़ की नींव तैयार!")
            return foundation
            
        except json.JSONDecodeError as e:
            st.error(f"❌ JSON Parse Error: {e}")
            return None
    
    def generate_chapter_outline(
        self,
        chapter_num: int,
        series_foundation: Dict
    ) -> Dict:
        """Generate single chapter outline with full context"""
        
        self._wait_for_rate_limit()
        
        user_id = self.session_id
        
        # Get previous chapter context
        prev_context = ""
        if chapter_num > 1 and self.chapter_summaries:
            prev_chapter = self.chapter_summaries[-1]
            prev_context = f"""
पिछला अध्याय (Chapter {chapter_num - 1}):
शीर्षक: {prev_chapter.get('title', '')}
सारांश: {prev_chapter.get('summary', '')}
अंत: {prev_chapter.get('ending', '')}
"""
        
        # Determine difficulty
        difficulty = "शुरुआती" if chapter_num <= 20 else "मध्यम" if chapter_num <= 50 else "उन्नत" if chapter_num <= 75 else "विशेषज्ञ"
        
        prompt = f"""सीरीज़ "{series_foundation['series_title']}" का अध्याय {chapter_num} का outline बनाओ।

सीरीज़ संदर्भ:
- मुख्य कहानी: {series_foundation.get('main_storyline', '')}
- केंद्रीय संघर्ष: {series_foundation.get('central_conflict', '')}
- दुनिया: {series_foundation.get('world_setting', '')}
{prev_context}

किरदार: {', '.join([c['name'] + ' (' + c['role'] + ')' for c in series_foundation.get('characters', [])])}

JSON format में return करो:
{{
    "chapter_num": {chapter_num},
    "title": "सबक-आधारित शीर्षक",
    "lesson_focus": "इस अध्याय में मुख्य सीख (2-3 वाक्य)",
    "plot_summary": "मुख्य घटनाएं (4-5 वाक्य, बहुत विस्तार से)",
    "character_focus": "किस किरदार का विकास होगा",
    "key_scenes": "3-4 महत्वपूर्ण दृश्य",
    "cliffhanger": "अगले अध्याय के लिए hook",
    "difficulty": "{difficulty}",
    "connection_to_previous": "पिछले अध्याय से कैसे जुड़ा है"
}}

सिर्फ JSON ऑब्जेक्ट return करो।"""
        
        response = self.story_planner.run(prompt, stream=False, user_id=user_id)
        self.rate_limiter.record_request()
        
        clean = self._extract_json(response.content.strip())
        
        try:
            outline = json.loads(clean)
            return outline
        except:
            return None
    
    def generate_chapter_content(
        self,
        chapter_num: int,
        chapter_outline: Dict,
        series_foundation: Dict
    ) -> str:
        """Generate detailed TTS-ready Hindi script (5000-7000 words)"""
        
        self._wait_for_rate_limit()
        
        user_id = self.session_id
        
        # Build context from previous chapters
        prev_context = ""
        if self.chapter_summaries:
            recent_summaries = self.chapter_summaries[-3:]  # Last 3 chapters
            prev_context = "\n\nपिछले अध्यायों का सारांश:\n"
            for summary in recent_summaries:
                prev_context += f"अध्याय {summary['chapter_num']}: {summary['summary']}\n"
        
        prompt = f"""अध्याय {chapter_num} का पूरा TTS-ready Hindi script लिखो।

सीरीज़: {series_foundation['series_title']}
मुख्य कहानी: {series_foundation.get('main_storyline', '')}
किरदार: {', '.join([f"{c['name']} ({c['role']})" for c in series_foundation.get('characters', [])])}
{prev_context}

इस अध्याय की जानकारी:
- शीर्षक: {chapter_outline['title']}
- सीख: {chapter_outline['lesson_focus']}
- कहानी: {chapter_outline['plot_summary']}
- दृश्य: {chapter_outline.get('key_scenes', '')}
- किरदार फोकस: {chapter_outline['character_focus']}
- अंत: {chapter_outline['cliffhanger']}
- पिछले से जुड़ाव: {chapter_outline.get('connection_to_previous', '')}

महत्वपूर्ण निर्देश:
1. 5000-7000 शब्दों का विस्तृत script (15-20 मिनट audio के लिए)
2. बोलचाल की आधुनिक हिंदी - पुराने शब्द नहीं
3. अंग्रेजी नाम/टर्म को देवनागरी में (मार्कस, स्ट्रैटिजी)
4. प्रवाह के लिए अल्पविराम (,) का खूब इस्तेमाल
5. हर दृश्य को विस्तार से बताओ - जल्दबाजी नहीं
6. किरदारों के emotions, thoughts को भी बताओ
7. कोई सिंबल नहीं (**, *, ##, (), [])
8. सिर्फ साफ देवनागरी text
9. सबक अंत में (5-8 लाइन, बहुत संक्षिप्त)

फॉर्मेट:
अध्याय {chapter_num}: {chapter_outline['title']}

[यहाँ 5000-7000 शब्दों की विस्तृत कहानी लिखो]

इस अध्याय से सीख
1. पहली सीख (एक लाइन)
2. दूसरी सीख (एक लाइन)
...

अब पूरा अध्याय लिखो - विस्तृत, रोचक, और TTS के लिए एकदम साफ।"""
        
        response = self.content_writer.run(prompt, stream=False, user_id=user_id)
        self.rate_limiter.record_request()
        
        content = response.content.strip()
        
        # Deep clean for TTS
        content = self._deep_clean_for_tts(content)
        
        # Store chapter summary for context
        self.chapter_summaries.append({
            'chapter_num': chapter_num,
            'title': chapter_outline['title'],
            'summary': chapter_outline['plot_summary'],
            'ending': chapter_outline['cliffhanger']
        })
        
        return content
    
    def _deep_clean_for_tts(self, text: str) -> str:
        """Deep cleaning for TTS compatibility"""
        
        # Remove ALL markdown
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+', '', text)
        text = re.sub(r'_+', '', text)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        
        # Remove brackets
        text = re.sub(r'\[.*?\]', '', text)
        text = re.sub(r'\(.*?\)', '', text)
        text = re.sub(r'\{.*?\}', '', text)
        
        # Remove scene markers
        text = re.sub(r'(?i)(panel|scene|दृश्य|पैनल)\s*\d+', '', text)
        text = re.sub(r'(?i)(visual|caption|narrator|कथावाचक):', '', text)
        
        # Remove emojis
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
        
        # Remove separators
        text = re.sub(r'[=\-_]{3,}', '', text)
        text = re.sub(r'[•·∙‣⁃]', '', text)
        
        # Fix dialogue: "NAME:" → "Name ने कहा -"
        text = re.sub(r'([A-Z][A-Za-z]+):\s*', r'\1 ने कहा - ', text)
        
        # Remove quotes
        text = re.sub(r'["""\'\'`]', '', text)
        
        # Fix spacing
        text = re.sub(r'\s+([।,])', r'\1', text)
        text = re.sub(r'([।,])\s*', r'\1 ', text)
        text = re.sub(r'([.!?])\s*', r'\1 ', text)
        
        # Clean whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        return text.strip()
    
    def _save_metadata(self, data: Dict, filename: str):
        """Save metadata to file"""
        filepath = os.path.join(
            METADATA_DIR,
            f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def save_chapter(self, chapter_num: int, content: str, series_title: str):
        """Save chapter content"""
        filename = f"{series_title.replace(' ', '_')}_ch{chapter_num:03d}.txt"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def get_rate_limit_status(self) -> Dict:
        """Get current rate limit status"""
        return {
            'requests_this_minute': len(self.rate_limiter.request_times),
            'requests_today': self.rate_limiter.daily_requests,
            'rpm_limit': self.rate_limiter.rpm,
            'rpd_limit': self.rate_limiter.rpd,
            'can_request': self.rate_limiter.can_make_request()[0]
        }


# Streamlit UI
def main():
    st.set_page_config(
        page_title="Hindi Manhwa Content Generator",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Hindi Educational Manhwa Content Generator")
    st.markdown("*विस्तृत, संदर्भ-जागरूक हिंदी audiobook scripts*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        gemini_api_key = st.text_input(
            "Gemini API Key",
            type="password",
            value=os.getenv("GEMINI_API_KEY", ""),
        )
        
        model_choice = st.selectbox(
            "Model",
            options=list(GEMINI_MODELS.keys()),
            format_func=lambda x: f"{x} - {GEMINI_MODELS[x]['description']}"
        )
        
        st.markdown("---")
        st.subheader("📊 Rate Limits (Free Tier)")
        if model_choice:
            config = GEMINI_MODELS[model_choice]
            st.info(f"""
**{model_choice}**
- {config['rpm']} requests/minute
- {config['tpm']:,} tokens/minute  
- {config['rpd']} requests/day
            """)
        
        # Check if key exists AND if the object is actually instantiated (not None)
        if 'generator' in st.session_state and st.session_state.generator is not None:
            status = st.session_state.generator.get_rate_limit_status()
            st.metric("Requests this minute", f"{status['requests_this_minute']}/{status['rpm_limit']}")
            st.metric("Requests today", f"{status['requests_today']}/{status['rpd_limit']}")
    if not gemini_api_key:
        st.warning("⚠️ Please enter Gemini API key in sidebar")
        return
    
    # Initialize session state
    if 'generator' not in st.session_state:
        st.session_state.generator = None
    if 'series_foundation' not in st.session_state:
        st.session_state.series_foundation = None
    if 'generated_chapters' not in st.session_state:
        st.session_state.generated_chapters = {}
    
    # Topic input
    st.header("🎯 Step 1: Create Series Foundation")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        skill_topic = st.text_input(
            "सीखने का विषय (Learning Topic)",
            placeholder="जैसे: Negotiation, Leadership, Strategic Thinking...",
        )
    with col2:
        st.write("")
        st.write("")
        create_series = st.button("🎬 Create Series", type="primary")
    
    # Create series foundation
    if create_series and skill_topic:
        
        # Initialize generator
        st.session_state.generator = HindiManhwaGenerator(
            gemini_api_key=gemini_api_key,
            model_choice=model_choice
        )
        
        with st.spinner("सीरीज़ की नींव बना रहे हैं..."):
            foundation = st.session_state.generator.generate_series_foundation(skill_topic)
            
            if foundation:
                st.session_state.series_foundation = foundation
                st.balloons()
                st.success("✅ सीरीज़ की नींव तैयार!")
            else:
                st.error("❌ Foundation generation failed")
    
    # Display series foundation
    if st.session_state.series_foundation:
        foundation = st.session_state.series_foundation
        
        st.markdown("---")
        st.header(f"📖 {foundation.get('series_title', 'Series')}")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.expander("📜 Story Overview", expanded=True):
                st.write(foundation.get('story_overview', 'No overview'))
                st.write(f"**Main Storyline:** {foundation.get('main_storyline', 'N/A')}")
                st.write(f"**Central Conflict:** {foundation.get('central_conflict', 'N/A')}")
                st.write(f"**World Setting:** {foundation.get('world_setting', 'N/A')}")
        
        with col2:
            with st.expander("👥 Characters", expanded=True):
                for char in foundation.get('characters', []):
                    st.markdown(f"**{char.get('name', 'Unknown')}** - *{char.get('role', 'N/A')}*")
                    st.caption(char.get('personality', 'N/A'))
                    st.caption(f"Arc: {char.get('character_arc', 'N/A')}")
                    st.markdown("---")
        
        # Chapter generation
        st.markdown("---")
        st.header("✍️ Step 2: Generate Chapters")
        
        st.info("""
        📝 **Content Format:**
        - Direct TTS-ready Hindi script (no intermediate steps)
        - 5000-7000 words per chapter (15-20 minutes audio)
        - Modern conversational Hindi
        - Context-aware (remembers previous chapters)
        - Lessons at the end (5-8 lines)
        """)
        
        # Single chapter generation
        st.subheader("Generate Single Chapter")
        
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            chapter_num = st.number_input(
                "Chapter Number",
                min_value=1,
                max_value=100,
                value=1,
                key="single_chapter"
            )
        
        with col2:
            st.write("")
            st.write("")
            gen_single = st.button("📝 Generate Chapter", type="primary")
        
        with col3:
            st.write("")
            st.write("")
            if chapter_num in st.session_state.generated_chapters:
                st.success("✅ Generated")
        
        if gen_single:
            with st.spinner(f"अध्याय {chapter_num} बना रहे हैं..."):
                progress_bar = st.progress(0)
                status = st.empty()
                
                # Step 1: Generate outline
                status.text("📋 Creating chapter outline...")
                progress_bar.progress(0.2)
                
                outline = st.session_state.generator.generate_chapter_outline(
                    chapter_num,
                    foundation
                )
                
                if not outline:
                    st.error(f"❌ Failed to create outline for chapter {chapter_num}")
                else:
                    # Step 2: Generate content
                    status.text("✍️ Writing detailed Hindi script...")
                    progress_bar.progress(0.4)
                    
                    content = st.session_state.generator.generate_chapter_content(
                        chapter_num,
                        outline,
                        foundation
                    )
                    
                    if content:
                        # Save chapter
                        status.text("💾 Saving chapter...")
                        progress_bar.progress(0.8)
                        
                        filepath = st.session_state.generator.save_chapter(
                            chapter_num,
                            content,
                            foundation['series_title']
                        )
                        
                        # Store in session
                        st.session_state.generated_chapters[chapter_num] = {
                            'outline': outline,
                            'content': content,
                            'filepath': filepath
                        }
                        
                        progress_bar.progress(1.0)
                        status.text("✅ Chapter complete!")
                        st.success(f"✅ अध्याय {chapter_num} तैयार!")
                        
                        # Display chapter
                        with st.expander(f"📖 Chapter {chapter_num}: {outline['title']}", expanded=True):
                            
                            # Outline info
                            st.markdown("**Chapter Outline:**")
                            st.write(f"**Focus:** {outline.get('lesson_focus', 'N/A')}")
                            st.write(f"**Plot:** {outline.get('plot_summary', 'N/A')}")
                            st.write(f"**Cliffhanger:** {outline.get('cliffhanger', 'N/A')}")
                            
                            st.markdown("---")
                            
                            # Content
                            st.markdown("**TTS-Ready Script:**")
                            
                            # Show word count
                            word_count = len(content.split())
                            st.caption(f"📊 Word Count: {word_count:,} words (~{word_count*0.003:.1f} minutes)")
                            
                            st.text_area(
                                "Content",
                                content,
                                height=400,
                                key=f"content_{chapter_num}"
                            )
                            
                            # Download button
                            st.download_button(
                                "⬇️ Download Chapter",
                                content,
                                file_name=f"chapter_{chapter_num:03d}.txt",
                                mime="text/plain"
                            )
                    else:
                        st.error(f"❌ Failed to generate content for chapter {chapter_num}")
        
        # Batch generation
        st.markdown("---")
        st.subheader("Generate Multiple Chapters")
        
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            start_ch = st.number_input(
                "From Chapter",
                min_value=1,
                max_value=100,
                value=1,
                key="batch_start"
            )
        
        with col2:
            end_ch = st.number_input(
                "To Chapter",
                min_value=1,
                max_value=100,
                value=min(5, 100),
                key="batch_end"
            )
        
        with col3:
            st.write("")
            st.write("")
            gen_batch = st.button("🚀 Generate Batch")
        
        if gen_batch and start_ch <= end_ch:
            st.info(f"🚀 Generating chapters {start_ch} to {end_ch}...")
            
            # Overall progress
            overall_progress = st.progress(0)
            overall_status = st.empty()
            
            success_count = 0
            failed_chapters = []
            
            for i in range(start_ch, end_ch + 1):
                overall_status.text(f"📝 Processing Chapter {i}/{end_ch}...")
                
                # Skip if already generated
                if i in st.session_state.generated_chapters:
                    st.info(f"⏭️ Chapter {i} already exists, skipping...")
                    success_count += 1
                    continue
                
                with st.expander(f"Chapter {i}", expanded=False):
                    try:
                        # Generate outline
                        st.text("📋 Creating outline...")
                        outline = st.session_state.generator.generate_chapter_outline(i, foundation)
                        
                        if not outline:
                            st.error(f"❌ Outline failed")
                            failed_chapters.append(i)
                            continue
                        
                        # Generate content
                        st.text("✍️ Writing script...")
                        content = st.session_state.generator.generate_chapter_content(
                            i, outline, foundation
                        )
                        
                        if not content:
                            st.error(f"❌ Content generation failed")
                            failed_chapters.append(i)
                            continue
                        
                        # Save
                        filepath = st.session_state.generator.save_chapter(
                            i, content, foundation['series_title']
                        )
                        
                        # Store
                        st.session_state.generated_chapters[i] = {
                            'outline': outline,
                            'content': content,
                            'filepath': filepath
                        }
                        
                        word_count = len(content.split())
                        st.success(f"✅ Chapter {i} complete! ({word_count:,} words)")
                        success_count += 1
                        
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                        failed_chapters.append(i)
                
                # Update progress
                progress = (i - start_ch + 1) / (end_ch - start_ch + 1)
                overall_progress.progress(progress)
            
            # Final summary
            overall_status.empty()
            overall_progress.progress(1.0)
            
            st.markdown("---")
            st.subheader("📊 Batch Generation Summary")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Chapters", end_ch - start_ch + 1)
            with col2:
                st.metric("Successful", success_count, delta=success_count)
            with col3:
                st.metric("Failed", len(failed_chapters), delta=-len(failed_chapters))
            
            if failed_chapters:
                st.warning(f"⚠️ Failed chapters: {', '.join(map(str, failed_chapters))}")
            else:
                st.success("🎉 All chapters generated successfully!")
                st.balloons()
        
        # View generated chapters
        if st.session_state.generated_chapters:
            st.markdown("---")
            st.subheader("📚 Generated Chapters")
            
            st.write(f"Total generated: **{len(st.session_state.generated_chapters)}** chapters")
            
            for ch_num in sorted(st.session_state.generated_chapters.keys()):
                ch_data = st.session_state.generated_chapters[ch_num]
                
                with st.expander(f"Chapter {ch_num}: {ch_data['outline']['title']}"):
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Lesson:** {ch_data['outline'].get('lesson_focus', 'N/A')}")
                        st.write(f"**Plot:** {ch_data['outline'].get('plot_summary', 'N/A')[:200]}...")
                        
                        word_count = len(ch_data['content'].split())
                        st.caption(f"📊 {word_count:,} words (~{word_count*0.003:.1f} min)")
                    
                    with col2:
                        st.download_button(
                            "⬇️ Download",
                            ch_data['content'],
                            file_name=f"chapter_{ch_num:03d}.txt",
                            key=f"download_{ch_num}"
                        )
                        
                        if st.button("👁️ View", key=f"view_{ch_num}"):
                            st.text_area(
                                "Content",
                                ch_data['content'],
                                height=300,
                                key=f"view_content_{ch_num}"
                            )
        
        # Utilities
        st.markdown("---")
        st.subheader("🔧 Utilities")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📂 View Files"):
                st.info(f"""
**Generated Files:**
- Content: `{OUTPUT_DIR}/`
- Metadata: `{METADATA_DIR}/`
- Database: `manhwa_knowledge.db`
                """)
        
        with col2:
            if st.button("🔄 Reset Session"):
                st.session_state.series_foundation = None
                st.session_state.generated_chapters = {}
                st.session_state.generator = None
                st.success("✅ Session reset!")
                st.rerun()
        
        with col3:
            if st.button("💾 Export All"):
                if st.session_state.generated_chapters:
                    # Create combined export
                    all_chapters = []
                    for ch_num in sorted(st.session_state.generated_chapters.keys()):
                        ch_data = st.session_state.generated_chapters[ch_num]
                        all_chapters.append({
                            'chapter_num': ch_num,
                            'title': ch_data['outline']['title'],
                            'content': ch_data['content']
                        })
                    
                    export_data = {
                        'series_foundation': foundation,
                        'chapters': all_chapters,
                        'total_chapters': len(all_chapters)
                    }
                    
                    st.download_button(
                        "⬇️ Download All",
                        json.dumps(export_data, ensure_ascii=False, indent=2),
                        file_name=f"{foundation['series_title'].replace(' ', '_')}_complete.json",
                        mime="application/json"
                    )


if __name__ == "__main__":
    main()