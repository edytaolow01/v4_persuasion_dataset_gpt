import json
import asyncio
import os
from dotenv import load_dotenv
from multi_agent_system import MultiAgentSystem
from loguru import logger

# Load environment variables
load_dotenv()

async def test_async_system():
    """Test asynchronous system on several articles"""
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables")
    
    # Initialize system
    system = MultiAgentSystem(api_key, language="polish")
    
    # Load several articles from dataset
    with open('Dataset_balanced_more_final_1591.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Select 3 articles for testing
    test_articles = []
    for article in dataset[:3]:  # First 3 articles
        test_articles.append({
            "gpt_article_rating": article.get('gpt_article_rating'),
            "title": article.get('title'),
            "article_id": article.get('id'),
            "language": article.get('language'),
            "article": article.get('article_body'),
            "date": article.get('date', '')
        })
    
    logger.info(f"Starting asynchronous system test on {len(test_articles)} articles")
    
    # Process articles in parallel
    tasks = []
    for article in test_articles:
        task = asyncio.create_task(system.process_article_with_expert(article))
        tasks.append(task)
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Summary of results
    successful = sum(1 for result in results if not isinstance(result, Exception))
    errors = sum(1 for result in results if isinstance(result, Exception))
    
    logger.info(f"Test completed. Success: {successful}, Errors: {errors}")
    
    # Display results
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Article {i+1}: Error - {result}")
        else:
            logger.info(f"Article {i+1}: Initial score: {result.get('initial_agreement_score')}, Final score: {result.get('final_agreement_score')}, Controversial boosted: {result.get('controversial_boosted')}")

if __name__ == "__main__":
    asyncio.run(test_async_system())
