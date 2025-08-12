# Multi-Agent System for Nuclear Energy Article Analysis

## Description

This system uses a multi-agent architecture to analyze and refine nuclear energy articles. The system consists of five specialized agents that work together to process articles through a sophisticated pipeline, supporting multiple languages and asynchronous processing.

## Agents

1. **Theory of Mind Agent** - Analyzes author intentions and extracts key nuclear energy statements
2. **Nuclear Expert Agent** - Evaluates statements and articles from a nuclear energy expert perspective
3. **Nuclear Layperson Agent** - Evaluates statements and articles from an average reader perspective
4. **Controlled Controversy Agent** - Enhances controversy in statements to reduce agreement levels
5. **Refine Agent** - Refines articles based on feedback from experts and laypersons

## Workflow

1. **Theory of Mind Analysis** - Extract statement from article
2. **Expert Evaluation** - Evaluation by expert (score 1-5)
3. **Controversy Agent** (optional) - If expert agrees with 5/5, up to 3 attempts to reduce score
4. **Refine Agent** (optional) - If expert doesn't agree with 5/5 (up to 5 rounds)
5. **Layperson Processing** - Repeat process with layperson (without Theory of Mind step)

## Key Features

### Multilingual Support
- Supports multiple languages: English, Polish, Czech, Slovak, Hungarian
- Dynamic prompt loading based on article language
- Automatic language detection and agent reloading

### Asynchronous Processing
- Concurrent processing of up to 10 articles simultaneously
- Efficient resource utilization with semaphore-based concurrency control
- Immediate result saving to prevent data loss

### Controversy Agent Enhancement
- Tracks previous failed attempts to avoid repetition
- Always updates final statement to the latest controversy agent output
- Maintains history of modification attempts

### Robust Error Handling
- JSON parsing with retry mechanism (3 attempts)
- Regex-based extraction for malformed responses
- Graceful fallback for parsing failures
- Comprehensive logging with loguru

### Deduplication System
- Prevents reprocessing of already completed articles
- Uses article_id for robust identification
- Automatic detection of existing results

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file:
```
OPENAI_API_KEY=your-api-key-here
```

Check your desired prompts and run process_all_articles_async.py

## Usage

### Single Article Processing
```bash
python test_single_article.py
```

### Batch Processing (Synchronous)
```bash
python process_all_articles.py
```

### Batch Processing (Asynchronous - Recommended)
```bash
python process_all_articles_async.py
```

### Multilingual Testing
```bash
python test_multilingual_system.py
```

## File Structure

```
├── multi_agent_system.py              # Main system file
├── process_all_articles.py            # Synchronous batch processor
├── process_all_articles_async.py      # Asynchronous batch processor
├── test_single_article.py             # Single article tester
├── test_async_system.py               # Async system tester
├── test_multilingual_system.py        # Multilingual tester
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── prompts_english/                   # English agent configurations
│   ├── theory_of_mind_agent.yaml
│   ├── nuclear_expert_agent.yaml
│   ├── nuclear_layperson_agent.yaml
│   ├── controlled_controversy_agent.yaml
│   └── refine_agent.yaml
├── prompts_polish/                    # Polish agent configurations
├── prompts_czech/                     # Czech agent configurations
├── prompts_slovak/                    # Slovak agent configurations
├── prompts_hungarian/                 # Hungarian agent configurations
├── Dataset_balanced_more_final_1591.json  # Input dataset
└── results/                           # Output results (auto-created)
```

## Input Data Format

The dataset should contain articles in JSON format with the following keys:
- `gpt_article_rating` - article rating (1-5)
- `title` - article title
- `id` - article ID
- `language` - article language (en, pl, cs, sk, hu)
- `article_body` - article content
- `date` - article date

In this repo, 1591 chosen articles from https://huggingface.co/datasets/eoplumbum/v4_nuclear_power_articles.

## Output Data Format

The system generates two JSON files for each article:
- `expert_{article_id}_{timestamp}.json` - expert results
- `layperson_{article_id}_{timestamp}.json` - layperson results

Each file contains:
- Article metadata (original_sentiment, original_title, article_id, language, original_article, date)
- Initial and final statements
- Dialog history with agents
- Cumulative linguistic and domain changes
- Agreement scores at various stages
- Controversy boosting information
- Missing information and suggested improvements

Expected results should have the same structure as https://huggingface.co/datasets/eoplumbum/persuasion-gemini-1.5-pro


## Configuration

You can customize the system by modifying YAML files in the language-specific prompt directories. Each agent has its own configuration file containing:
- Agent name
- System prompt
- Prompt template with parameters

## Technical Details

### Model Configuration
- Uses GPT-5o-mini model (configurable)
- JSON response format enforcement
- 4000 max tokens per response
- Minimal reasoning effort for efficiency

### Processing Features
- Automatic language switching
- History tracking for refinement rounds
- Controversy attempt history
- Immediate result persistence
- Concurrent processing with rate limiting

## Logging

The system uses loguru for logging with the following features:
- Daily log rotation
- Structured logging format
- Detailed execution tracing
- Error tracking and debugging information

## Performance

- Asynchronous processing with 10 concurrent articles
- Immediate result saving prevents data loss
- Efficient memory usage with streaming processing
- Automatic deduplication prevents redundant work

## Notes

- The system automatically handles language detection and agent reloading
- Results are saved immediately after each agent pipeline completion
- Controversy agent always updates the final statement, even if all attempts fail
- The system maintains complete history for refinement agents
- All comments and logs are in English for consistency
# v4_persuasion_dataset_gpt
