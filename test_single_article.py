import json
import asyncio
import os
from dotenv import load_dotenv
from multi_agent_system import MultiAgentSystem
from loguru import logger

# Load environment variables from .env file
load_dotenv()

async def test_single_article():
    """Test system on single article"""
    
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables")
    
    # Initialize system with Polish language
    system = MultiAgentSystem(api_key, language="polish")
    
    # Load article 25 from dataset
    with open('Dataset_balanced_more_final_1591.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Find article 25
    test_article = None
    for article in dataset:
        if article.get('id') == 25:
            test_article = {
                "gpt_article_rating": article.get('gpt_article_rating'),
                "title": article.get('title'),
                "article_id": article.get('id'),
                "language": article.get('language'),
                "article": article.get('article_body'),
                "date": article.get('date', '')
            }
            break
    
    if not test_article:
        raise ValueError("Article 25 not found in dataset")
    
    logger.info(f"Found article 25: {test_article['title']}")
    
    logger.info("Starting single article test")
    
    try:
        # Processing with expert
        logger.info("=== EXPERT PROCESSING ===")
        expert_result = await system.process_article_with_expert(test_article)
        
        print("\n=== EXPERT RESULTS ===")
        print(f"Initial statement: {expert_result['initial_ne_statement']}")
        print(f"Final statement: {expert_result['final_ne_statement']}")
        print(f"Initial agreement score: {expert_result['initial_agreement_score']}")
        print(f"Final agreement score: {expert_result['final_agreement_score']}")
        print(f"Controversial boosted: {expert_result['controversial_boosted']}")
        print(f"Number of dialog rounds: {len(expert_result['dialog_rounds'])}")
        
        # Processing with layperson
        logger.info("=== LAYPERSON PROCESSING ===")
        layperson_result = await system.process_article_with_layperson(
            test_article, 
            expert_result['final_ne_statement']
        )
        
        print("\n=== LAYPERSON RESULTS ===")
        print(f"Initial statement: {layperson_result['initial_ne_statement']}")
        print(f"Final statement: {layperson_result['final_ne_statement']}")
        print(f"Initial agreement score: {layperson_result['initial_agreement_score']}")
        print(f"Final agreement score: {layperson_result['final_agreement_score']}")
        print(f"Controversial boosted: {layperson_result['controversial_boosted']}")
        print(f"Number of dialog rounds: {len(layperson_result['dialog_rounds'])}")
        
        # Save test results
        os.makedirs("test_results", exist_ok=True)
        
        with open("test_results/expert_test_result.json", 'w', encoding='utf-8') as f:
            json.dump(expert_result, f, ensure_ascii=False, indent=2)
        
        with open("test_results/layperson_test_result.json", 'w', encoding='utf-8') as f:
            json.dump(layperson_result, f, ensure_ascii=False, indent=2)
        
        logger.info("Test completed successfully. Results saved in test_results/ directory")
        
        return expert_result, layperson_result
        
    except Exception as e:
        logger.error(f"Error during test: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(test_single_article())
