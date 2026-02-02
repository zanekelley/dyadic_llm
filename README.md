
# dyadic_llm

## Overview

This Python tool takes a folder of dyadic transcripts, combines them into a single file, and feeds them into an LLM for classification.
The main entry point is `main.py`, which uses helper functions from the `utils/` folder.

Presently, there are three fake transcripts in the transcript folder to show how the tool works. All other csv files throughout were created via those files.

> **Note:** Before running `main.py`, make sure to process your folder of transcripts with `i_transcript_combiner.py` so your files are compatible.

---

## Requirements

- Python 3.9+
- Packages listed in `requirements.txt` (pandas, openai, python-dotenv)

> Make sure to create a `.env` file with your OpenAI API key, following the format in the .env.example file:
> `API_KEY=your_api_key_here`

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ZaneKelley/dyadic_llm.git
cd dyadic_llm
python -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

---

## Project Structure

```
dyadic_llm/
├── main.py
├── utils/
│   ├── i_transcript_combiner.py
│   ├── ii_transcript_structurer.py
│   ├── iii_llm_decider.py
│   └── iv_llm_parser.py
├── inputs/
│   ├── transcripts/      	# Raw or combined transcripts go here
│   └── llm/              	# Files generated for LLM input
├── outputs/  			# Processed results will be written here
├── .env  
├── requirements.txt
├── README.md
```
