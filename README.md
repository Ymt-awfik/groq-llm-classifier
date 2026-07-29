# Groq LLM Classifier

A text classification and prompt-engineering tool built on the **Groq API**
(Llama 3 70B). It demonstrates structured prompting, confidence-scored
classification, and a comparison of different prompting strategies.

## Features
- **Structured completions** — formats prompts into a consistent analysis template
- **Classification with confidence** — classifies text and assigns a confidence
  score (high/medium/low), falling back to "uncertain" below a threshold
- **Prompt-strategy comparison** — runs the same classification with basic,
  structured, and few-shot prompts to compare their outputs

## Stack
- **Language:** Python
- **LLM:** Groq API — Llama 3 70B (`llama3-70b-8192`)
- **Core:** prompt engineering, section parsing, confidence thresholding

## Setup
```bash
pip install groq python-dotenv
```
Create a `.env` file (never commit it):
```env
GROQ_API_KEY=your_api_key_here
```
Run:
```bash
python taming_llm.py
```

## How it works
The `LLMClient` wraps the Groq chat API. Text is classified via a structured
prompt, the response is parsed into category / confidence / reasoning, and a
confidence threshold decides whether to trust the result. A separate module
runs the same inputs through three prompting strategies to compare quality.
