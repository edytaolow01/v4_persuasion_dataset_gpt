import json
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from multi_agent_system import MultiAgentSystem
from loguru import logger

# Load environment variables
load_dotenv()

class AsyncArticleProcessor:
    def __init__(self, max_concurrent=10):
        """Initialize processor with concurrent task limit"""
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Check API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")
        
        self.system = MultiAgentSystem(api_key)
        self.results_dir = "results"
        self.processed_articles = set()
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Load already processed articles
        self.load_processed_articles()
    
    def load_processed_articles(self):
        """Load already processed articles based on filenames"""
        if not os.path.exists(self.results_dir):
            return
        
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.results_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        article_id = data.get('article_id')
                        if article_id:
                            self.processed_articles.add(article_id)
                except Exception as e:
                    logger.warning(f"Failed to load file {filename}: {e}")
        
        logger.info(f"Loaded {len(self.processed_articles)} already processed articles")
    
    def prepare_article_data(self, article):
        """Prepare article data"""
        return {
            "gpt_article_rating": article.get('gpt_article_rating'),
            "title": article.get('title'),
            "article_id": article.get('id'),
            "language": article.get('language'),
            "article": article.get('article_body'),
            "date": article.get('date', '')
        }
    
    def save_results(self, results, article_id, audience_type):
        """Save results to file with article_id"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{audience_type}_{article_id}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved results for article {article_id} ({audience_type}) to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            return None
    
    async def process_single_article(self, article_data):
        """Processing single article with immediate save"""
        article_id = article_data["article_id"]
        
        async with self.semaphore:  # Limit concurrent tasks
            try:
                logger.info(f"Starting article processing {article_id}")
                
                # Processing with expert
                logger.info(f"Processing with expert for article {article_id}")
                expert_results = await self.system.process_article_with_expert(article_data)
                
                # Immediate save of expert results
                expert_file = self.save_results(expert_results, article_id, "expert")
                if not expert_file:
                    logger.error(f"Failed to save expert results for article {article_id}")
                    return False
                
                # Processing with layperson
                logger.info(f"Processing with layperson for article {article_id}")
                final_statement = expert_results.get("final_ne_statement", "")
                layperson_results = await self.system.process_article_with_layperson(article_data, final_statement)
                
                # Immediate save of layperson results
                layperson_file = self.save_results(layperson_results, article_id, "layperson")
                if not layperson_file:
                    logger.error(f"Failed to save layperson results for article {article_id}")
                    return False
                
                logger.info(f"Completed article processing {article_id}")
                return True
                
            except Exception as e:
                logger.error(f"Error processing article {article_id}: {e}")
                return False
    
    async def process_all_articles(self):
        """Asynchronous processing of all articles"""
        # Load dataset
        try:
            with open('Dataset_balanced_more_final_1591.json', 'r', encoding='utf-8') as f:
                dataset = json.load(f)
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return
        
        logger.info(f"Loaded {len(dataset)} articles from dataset")
        
        # Filter articles to process
        articles_to_process = []
        for article in dataset:
            article_id = article.get('id')
            if article_id and article_id not in self.processed_articles:
                articles_to_process.append(article)
        
        logger.info(f"Found {len(articles_to_process)} articles to process")
        
        if not articles_to_process:
            logger.info("All articles have already been processed")
            return
        
        # Process articles asynchronously
        tasks = []
        for article in articles_to_process:
            article_data = self.prepare_article_data(article)
            task = asyncio.create_task(self.process_single_article(article_data))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Summary of results
        successful = sum(1 for result in results if result is True)
        failed = sum(1 for result in results if result is False)
        errors = sum(1 for result in results if isinstance(result, Exception))
        
        logger.info(f"Processing completed. Success: {successful}, Errors: {failed}, Exceptions: {errors}")
        logger.info(f"Progress: {successful + len(self.processed_articles)}/{len(dataset)} articles processed")

async def main():
    """Main function"""
    logger.info("Starting asynchronous processing of all articles")
    
    try:
        processor = AsyncArticleProcessor(max_concurrent=10)
        await processor.process_all_articles()
        logger.info("Processing completed successfully")
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
