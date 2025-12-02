# Hindi Manhwa Content Generator

A modular, scalable, resume-capable system for generating long-form Hindi manhwa-style audiobook scripts.
The generator produces context-aware, suspense-oriented, spoken-Hindi narration using Gemini models.
All logic is cleanly separated into modules to ensure maintainability, extensibility, and clarity.

---

## 1. Overview

The Hindi Manhwa Content Generator is a command-line application that creates:

1. A complete 100-chapter story foundation
2. Chapter outlines in structured batches
3. Full chapter scripts ranging from 6000–8000 words
4. Context-aware, indirect-speech narration in natural conversational Hindi
5. Automatic resume capability in every stage of generation

The project architecture is designed for production use, with clear separation of concerns, checkpointing, layered instructions, and extensible generators.

---

## 2. Key Features

### 2.1 Modular Code Structure

All core functionality is separated into dedicated modules:

* Agent configuration
* Story foundation builder
* Chapter outline generator
* Context extractor and manager
* Chapter content writer
* Rate-limiting controller
* Resume and progress manager

This allows the project to scale without affecting existing logic.

### 2.2 Spoken Hindi Manhwa Style

The generator produces:

* Simple, everyday conversational Hindi
* Continuous indirect speech
* Psychological tension and strategic interactions
* Hidden intentions for the main character
* Heavy emphasis on suspense and layered personality
* No literary or uncommon Hindi vocabulary

The system is optimized to match the narration style used in popular Hindi manhwa YouTube channels.

### 2.3 Resume Capability

The generator stores checkpoints at every stage:

Foundation Progress
`manhwa_metadata/<session_id>_foundation.json`

Chapter Outline Progress
`manhwa_metadata/<session_id>_outline_progress.json`

Chapter Content Progress
`manhwa_metadata/<session_id>_chapter_progress.json`

A new session automatically checks for previous progress and resumes from the last completed step.
This prevents loss of work due to rate limits, interruptions, or manual exit.

### 2.4 Rate Limit Management

A dedicated rate limiter monitors:

* Requests per minute
* Tokens per minute
* Requests per day

The generator automatically waits for cooldown periods when needed.

---

## 3. Requirements

### 3.1 Software Requirements

* Python 3.9 or above
* pip package manager

### 3.2 Python Dependencies

The following primary libraries are required:

* agno
* python-dotenv
* sqlite3 (included with Python)
* pathlib

Install dependencies via:

```
pip install -r requirements.txt
```

---

## 4. Directory Structure

```
project_root/
│
├── manhwa_generator/
│   ├── main.py
│   ├── core/
│   │   ├── rate_limiter.py
│   │   ├── resume_manager.py
│   │
│   ├── config/
│   │   └── settings.py
│   │
│   ├── agents/
│   │   ├── planner_agent.py
│   │   └── writer_agent.py
│   │
│   ├── generator/
│   │   ├── hindi_manhwa_generator.py
│   │   ├── foundation_builder.py
│   │   ├── chapter_outline_builder.py
│   │   ├── chapter_content_builder.py
│   │   └── context_manager.py
│   │
│   └── utils/
│       ├── json_utils.py
│       └── text_cleaner.py
│
├── manhwa_content/
├── manhwa_metadata/
├── chapter_context/
├── manhwa_knowledge.db
└── README.md
```

---

## 5. Environment Configuration

Create a `.env` file in the project root and add:

```
GEMINI_API_KEY=your_api_key_here
```

This key is used by the Gemini models for all generation tasks.

---

## 6. How to Run the Application

Run the generator using:

```
python manhwa_generator/main.py
```

The application will automatically:

1. Detect existing progress (if any)
2. Resume where it last stopped
3. Allow full, range-based, or single-chapter generation
4. Save all outputs in the designated folders

---

## 7. Workflow

### 7.1 Foundation Generation

The system first generates a complete story foundation:

* World
* Characters
* Plot structure
* Skills theme

If a session foundation exists, it is reused.

### 7.2 Chapter Outline Generation

Chapter outlines are generated in batches.
If previous batches exist, only missing batches are generated.

### 7.3 Chapter Content Generation

Each chapter is generated individually with:

* Previous summaries
* Extracted context
* Character personalities
* Suspense-driven progression

The system tracks completed chapters and resumes accordingly.

---

## 8. Writing Logic and Style Rules

### 8.1 Indirect Speech

The generator is constrained to:

* Use only indirect speech
* Avoid quotation marks
* Avoid script-like lines such as “Name -”
* Produce narration-style dialog transitions:

  * उसने कहा कि
  * उसने समझाया कि
  * उसने मन में सोचा कि

### 8.2 Spoken Hindi Restrictions

The generator avoids:

* Literary vocabulary
* Outdated Hindi words
* Sanskritized terms
* Academic or philosophical tone

### 8.3 Suspense and Character Intelligence

All characters are designed to be:

* Highly intelligent
* Strategic
* Experienced
* Engaged in subtle mind games

The main character remains the most dangerous individual but hides their true nature.

---

## 9. Storage and Outputs

### 9.1 Output Directories

`manhwa_content/`
Contains the generated chapters as `.txt` files.

`manhwa_metadata/`
Stores:

* Foundation files
* Outline batches
* Progress checkpoints

`chapter_context/`
Contains chapter ending excerpts for continuity.

`manhwa_knowledge.db`
Stores memories, chat history, and agent state for context retrieval.

---

## 10. Extending the System

The modular structure allows for:

* Custom instruction models
* Additional generator modules
* Multi-language support
* External TTS integration
* Web UI frontend
* Cloud deployment

Each component can be replaced or expanded without modifying the core logic.

---

## 11. Troubleshooting

### Application restarts generation

Cause: Session ID is time-based.
Fix: Implement persistent session naming or manual session selection.

### Heavy Hindi appears in output

Cause: Incorrect or outdated writer instructions.
Fix: Ensure updated indirect-Hindi instructions are used.

### Story does not resume

Cause: Missing progress files or incorrect file paths.
Fix: Verify checkpoint JSON files exist in manhwa_metadata/.

### Gemini rate limit errors

Cause: Model restrictions.
Fix: Use built-in rate limiter; the application waits automatically.

---

## 12. License

This project is intended for private and educational use.
Ensure compliance with relevant model usage terms and API provider policies.
