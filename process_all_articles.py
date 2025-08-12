import json
import asyncio
import os
from datetime import datetime
from dotenv import load_dotenv
from multi_agent_system import MultiAgentSystem
from loguru import logger
import glob

# Load environment variables from .env file
load_dotenv()

class ArticleProcessor:
    def __init__(self, dataset_path: str, results_dir: str = "results"):
        """
        Article processor initialization
        
        Args:
            dataset_path: Path to dataset file
            results_dir: Results directory
        """
        self.dataset_path = dataset_path
        self.results_dir = results_dir
        self.processed_articles = set()
        
        # Check API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")
        
        # Initialize multi-agent system
        self.system = MultiAgentSystem(api_key)
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        
        # Load list of already processed articles
        self.load_processed_articles()
    
    def load_processed_articles(self):
        """Loading list of already processed articles"""
        try:
            # Check existing result files
            expert_files = glob.glob(f"{self.results_dir}/expert_*.json")
            layperson_files = glob.glob(f"{self.results_dir}/layperson_*.json")
            
            for file_path in expert_files + layperson_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        article_id = data.get('article_id')
                        if article_id:
                            self.processed_articles.add(article_id)
                except Exception as e:
                    logger.warning(f"Failed to read file {file_path}: {e}")
            
            logger.info(f"Found {len(self.processed_articles)} already processed articles")
            
        except Exception as e:
            logger.error(f"Error loading processed articles: {e}")
    
    def load_dataset(self):
        """Loading dataset"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            logger.info(f"Loaded {len(dataset)} articles from dataset")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return []
    
    def prepare_article_data(self, article):
        """Prepare article data for processing"""
        return {
            "gpt_article_rating": article.get("gpt_article_rating", 3),
            "title": article.get("title", ""),
            "article_id": article.get("id", 0),
            "language": article.get("language", "en"),
            "article": article.get("article_body", ""),
            "date": article.get("date", "")
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
            
            # Add to processed list
            self.processed_articles.add(article_id)
            
            logger.info(f"Completed article processing {article_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing article {article_id}: {e}")
            return False
    
    async def process_all_articles(self):
        """Processing all articles from dataset"""
        dataset = self.load_dataset()
        if not dataset:
            logger.error("Failed to load dataset")
            return
        
        # Filter articles that haven't been processed yet
        articles_to_process = []
        for article in dataset:
            article_id = article.get("id")
            if article_id and article_id not in self.processed_articles:
                articles_to_process.append(article)
        
        logger.info(f"Found {len(articles_to_process)} articles to process")
        
        if not articles_to_process:
            logger.info("All articles have already been processed")
            return
        
        # Processing articles
        successful = 0
        failed = 0
        
        for i, article in enumerate(articles_to_process, 1):
            logger.info(f"Processing article {i}/{len(articles_to_process)} (ID: {article.get('id')})")
            
            article_data = self.prepare_article_data(article)
            success = await self.process_single_article(article_data)
            
            if success:
                successful += 1
                logger.info(f" Success: Article {article.get('id')} processed successfully")
            else:
                failed += 1
                logger.error(f" Error: Article {article.get('id')} was not processed")
            
            # Short pause between articles
            await asyncio.sleep(1)
        
        logger.info(f"Processing completed. Success: {successful}, Errors: {failed}")
        logger.info(f"Progress: {successful + len(self.processed_articles)}/{len(dataset)} articles processed")

async def main():
    """Main function"""
    try:
        processor = ArticleProcessor("Dataset_balanced_more_final_1591.json")
        
        # Processing all articles
        await processor.process_all_articles()
        
    except Exception as e:
        logger.error(f"Error in main function: {e}")

if __name__ == "__main__":
    asyncio.run(main())
