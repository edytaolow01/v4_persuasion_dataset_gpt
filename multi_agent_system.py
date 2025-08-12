import json
import yaml
import asyncio
from typing import Dict, List, Any, Optional
from loguru import logger
import openai
from dataclasses import dataclass
from datetime import datetime
import os

# Loguru configuration
logger.add("multi_agent_system.log", rotation="1 day", retention="7 days", level="INFO")

@dataclass
class AgentConfig:
    """Agent configuration"""
    name: str
    system_prompt: str
    prompt_template: str

class MultiAgentSystem:
    def __init__(self, openai_api_key: str, model: str = "gpt-5-mini", language: str = "en"):
        """
        Multi-agent system initialization
        
        Args:
            openai_api_key: OpenAI API key
            model: Model to use (default gpt-5-mini)
            language: Article language (default "en")
        """
        self.client = openai.AsyncOpenAI(api_key=openai_api_key)
        self.model = model
        self.language = language
        self.agents = {}
        self.load_agents(language)
        
    def load_agents(self, language='en'):
        """Loading agent configurations from YAML files"""
        # Determine prompt folder based on language
        if language == 'polish' or language == 'pl':
            prompt_folder = 'prompts_polish'
        elif language == 'czech' or language == 'cs' or language == 'cz':
            prompt_folder = 'prompts_czech'
        elif language == 'hungarian' or language == 'hu' or language == 'hun':
            prompt_folder = 'prompts_hungarian'
        elif language == 'slovak' or language == 'sk':
            prompt_folder = 'prompts_slovak'
        else:
            prompt_folder = 'prompts_english'
            
        agent_files = {
            'theory_of_mind': f'{prompt_folder}/theory_of_mind_agent.yaml',
            'nuclear_expert': f'{prompt_folder}/nuclear_expert_agent.yaml',
            'nuclear_layperson': f'{prompt_folder}/nuclear_layperson_agent.yaml',
            'controlled_controversy': f'{prompt_folder}/controlled_controversy_agent.yaml',
            'refine': f'{prompt_folder}/refine_agent.yaml'
        }
        
        for agent_name, file_path in agent_files.items():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    
                if agent_name == 'theory_of_mind':
                    prompt_template = config['combined_analysis_prompt']
                elif agent_name == 'nuclear_expert':
                    prompt_template = config['expert_evaluation_prompt']
                elif agent_name == 'nuclear_layperson':
                    prompt_template = config['layperson_evaluation_prompt']
                elif agent_name == 'controlled_controversy':
                    prompt_template = config['controversy_boost_prompt']
                elif agent_name == 'refine':
                    prompt_template = config['article_refinement_prompt']
                
                self.agents[agent_name] = AgentConfig(
                    name=config['name'],
                    system_prompt=config['system_prompt'],
                    prompt_template=prompt_template
                )
                logger.info(f"Loaded agent: {agent_name}")
                
            except Exception as e:
                logger.error(f"Error loading agent {agent_name}: {e}")
                raise
    
    async def call_agent(self, agent_name: str, **kwargs) -> Dict[str, Any]:
        """
        Call agent with given parameters
        
        Args:
            agent_name: Agent name
            **kwargs: Parameters to pass to prompt
            
        Returns:
            Agent response in JSON format
        """
        if agent_name not in self.agents:
            raise ValueError(f"Unknown agent: {agent_name}")
        
        agent = self.agents[agent_name]
        
        # Format prompt with additional JSON instructions
        prompt = agent.prompt_template.format(**kwargs)
        prompt += "\n\nIMPORTANT: Response MUST be in JSON format. Do not use markdown, do not add comments before or after JSON."
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": agent.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    reasoning_effort="minimal",
                    response_format={"type": "json_object"}
                )
                
                content = response.choices[0].message.content
                logger.info(f"Response from agent {agent_name} (attempt {attempt + 1}): {content[:100]}...")
                
                # JSON parsing
                try:
                    result = json.loads(content)
                    # Check if result doesn't contain errors
                    if "error" not in result:
                        return result
                    else:
                        logger.warning(f"Agent {agent_name} returned error: {result['error']}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            return result
                            
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error from agent {agent_name} (attempt {attempt + 1}): {content[:500]}...")
                    logger.error(f"Error details: {e}")
                    
                    if attempt < max_retries - 1:
                        logger.info(f"Retrying for agent {agent_name}...")
                        await asyncio.sleep(1)  # Short pause before retry
                        continue
                    else:
                        # Last attempt - return basic structure in case of error
                        return {"error": f"JSON parsing error after {max_retries} attempts: {e}", "raw_response": content[:500]}
                    
            except Exception as e:
                logger.error(f"Error calling agent {agent_name} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                else:
                    raise
    
    async def theory_of_mind_analysis(self, title: str, article: str) -> Dict[str, Any]:
        """Theory of Mind analysis - extracting statement from article"""
        logger.info("Starting Theory of Mind analysis")
        
        result = await self.call_agent('theory_of_mind', title=title, article=article)
        
        logger.info(f"ToM analysis result: {result}")
        return result
    
    async def expert_evaluation(self, statement: str, article: str, article_history: str = "") -> Dict[str, Any]:
        """Expert evaluation"""
        logger.info("Starting expert evaluation")
        
        result = await self.call_agent('nuclear_expert', 
                                     statement=statement, 
                                     article=article, 
                                     article_history=article_history)
        
        logger.info(f"Expert evaluation result: {result}")
        return result
    
    async def layperson_evaluation(self, statement: str, article: str, article_history: str = "") -> Dict[str, Any]:
        """Layperson evaluation"""
        logger.info("Starting layperson evaluation")
        
        result = await self.call_agent('nuclear_layperson', 
                                     statement=statement, 
                                     article=article, 
                                     article_history=article_history)
        
        logger.info(f"Layperson evaluation result: {result}")
        return result
    
    async def controversy_boost(self, original_statement: str, article: str, previous_attempts: str = "") -> Dict[str, Any]:
        """Boosting controversy in statement"""
        logger.info("Starting controversy boost")
        
        result = await self.call_agent('controlled_controversy', 
                                     original_statement=original_statement, 
                                     article=article,
                                     previous_attempts=previous_attempts)
        
        logger.info(f"Controversy boost result: {result}")
        return result
    
    async def refine_article(self, original_title: str, original_article: str, 
                           current_article: str, statement: str, feedback: str,
                           missing_information: List[str], suggested_improvements: List[str],
                           previous_rounds: str, audience_type: str) -> Dict[str, Any]:
        """Article refinement"""
        logger.info(f"Starting article refinement for {audience_type}")
        
        original_length = len(original_article)
        current_length = len(current_article)
        max_allowed_length = int(original_length * 1.1)
        
        result = await self.call_agent('refine', 
                                     original_title=original_title,
                                     original_article=original_article,
                                     original_length=original_length,
                                     current_length=current_length,
                                     max_allowed_length=max_allowed_length,
                                     statement=statement,
                                     feedback=feedback,
                                     missing_information=", ".join(missing_information),
                                     suggested_improvements=", ".join(suggested_improvements),
                                     previous_rounds=previous_rounds,
                                     audience_type=audience_type)
        
        logger.info(f"Refine result: {result}")
        return result
    
    async def process_article_with_expert(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """Processing article with expert"""
        logger.info(f"Starting article processing {article_data.get('article_id', 'unknown')} with expert")
        
        # Determine language from article data
        article_language = article_data.get('language', 'en')
        if article_language != self.language:
            logger.info(f"Changing language from {self.language} to {article_language}")
            self.language = article_language
            self.load_agents(article_language)
        
        # 1. Theory of Mind Analysis
        tom_result = await self.theory_of_mind_analysis(
            article_data['title'], 
            article_data['article']
        )
        
        initial_statement = tom_result['nuclear_energy_statement']
        
        # 2. Expert Evaluation
        expert_result = await self.expert_evaluation(
            initial_statement, 
            article_data['article']
        )
        
        logger.info(f"Expert result agreement_score: {expert_result['agreement_score']} (type: {type(expert_result['agreement_score'])})")
        agreement_score = int(expert_result['agreement_score'])
        final_statement = initial_statement
        controversial_boosted = False
        
        logger.info(f"Expert initial agreement_score: {agreement_score} (type: {type(agreement_score)})")
        
        # 3. Controversy Agent (if expert agrees with 5)
        logger.info(f"Checking condition: agreement_score == 5 -> {agreement_score} == 5 -> {agreement_score == 5}")
        if agreement_score == 5:
            logger.info("Expert agreed with 5, starting Controversy Agent")
            
            previous_attempts = ""
            for attempt in range(3):
                logger.info(f"Controversy Agent (expert) - passing history: '{previous_attempts}'")
                controversy_result = await self.controversy_boost(
                    final_statement, 
                    article_data['article'],
                    previous_attempts
                )
                
                boosted_statement = controversy_result['boosted_statement']
                
                # Always update final_statement to the latest from Controversy Agent
                final_statement = boosted_statement
                controversial_boosted = True
                
                # Re-evaluation by expert
                new_expert_result = await self.expert_evaluation(
                    boosted_statement, 
                    article_data['article']
                )
                
                new_agreement_score = int(new_expert_result['agreement_score'])
                
                if new_agreement_score < 5:
                    expert_result = new_expert_result
                    agreement_score = new_agreement_score
                    logger.info(f"Controversy Agent successful in attempt {attempt + 1}")
                    break
                else:
                    logger.info(f"Controversy Agent attempt {attempt + 1} failed")
                    # Add failed attempt to history
                    if previous_attempts:
                        previous_attempts += f"\nAttempt {attempt + 1} (failed): {boosted_statement}"
                    else:
                        previous_attempts = f"Attempt {attempt + 1} (failed): {boosted_statement}"
        
        # 4. Refine Agent (if expert doesn't agree with 5)
        dialog_rounds = []
        current_article = article_data['article']
        article_history = ""
        
        if agreement_score < 5:
            logger.info("Expert didn't agree with 5, starting Refine Agent")
            
            for round_num in range(6):  # 0-5 rounds
                logger.info(f"Refinement round {round_num}")
                
                # Article refinement
                refine_result = await self.refine_article(
                    original_title=article_data['title'],
                    original_article=article_data['article'],
                    current_article=current_article,
                    statement=final_statement,
                    feedback=expert_result['detailed_feedback'],
                    missing_information=expert_result.get('missing_information', []),
                    suggested_improvements=expert_result.get('suggested_improvements', []),
                    previous_rounds=article_history,
                    audience_type="expert"
                )
                
                refined_article = refine_result['refined_article']
                persuasion_strategies = refine_result.get('persuasion_strategies', [])
                extracted_domain_changes = refine_result.get('extracted_domain_changes', [])
                
                # Update history
                article_history += f"\nRound {round_num}: {refined_article}\n"
                
                # Re-evaluation by expert
                new_expert_result = await self.expert_evaluation(
                    final_statement, 
                    refined_article,
                    article_history
                )
                
                new_agreement_score = int(new_expert_result['agreement_score'])
                
                # Save round
                round_data = {
                    "round": round_num,
                    "audience_feedback": new_expert_result['detailed_feedback'],
                    "agreement_score": new_agreement_score,
                    "refined_article": refined_article,
                    "persuasion_strategies": persuasion_strategies,
                    "extracted_domain_changes": extracted_domain_changes,
                    "missing_information": new_expert_result.get('missing_information', []),
                    "suggested_improvements": new_expert_result.get('suggested_improvements', [])
                }
                
                dialog_rounds.append(round_data)
                
                # Update for next round
                current_article = refined_article
                expert_result = new_expert_result
                agreement_score = new_agreement_score
                
                logger.info(f"Round {round_num} completed, score: {agreement_score}")
                
                # If 5 is achieved, can break
                if agreement_score == 5:
                    logger.info("Score 5 achieved, ending refinement")
                    break
        else:
            # If expert immediately agreed with <5, add round 0
            round_data = {
                "round": 0,
                "audience_feedback": expert_result['detailed_feedback'],
                "agreement_score": agreement_score,
                "refined_article": article_data['article'],
                "persuasion_strategies": [],
                "extracted_domain_changes": [],
                "missing_information": expert_result.get('missing_information', []),
                "suggested_improvements": expert_result.get('suggested_improvements', [])
            }
            dialog_rounds.append(round_data)
        
        # Prepare results
        result = {
            "original_sentiment": article_data.get('gpt_article_rating'),
            "original_title": article_data['title'],
            "date": article_data.get('date', ''),
            "article_id": article_data['article_id'],
            "language": article_data['language'],
            "original_article": article_data['article'],
            "author_intention": tom_result['author_intention'],
            "initial_ne_statement": initial_statement,
            "final_ne_statement": final_statement,
            "controversial_boosted": controversial_boosted,
            "audience_type": "expert",
            "dialog_rounds": dialog_rounds,
            "final_refined_article": current_article,
            "final_agreement_score": agreement_score,
            "initial_agreement_score": int(expert_result['agreement_score']) if not controversial_boosted else agreement_score
        }
        
        # Dodanie skumulowanych zmian
        self._add_cumulative_changes(result)
        
        logger.info(f"Zakończono przetwarzanie z ekspertem, final score: {agreement_score}")
        return result
    
    async def process_article_with_layperson(self, article_data: Dict[str, Any], final_statement: str) -> Dict[str, Any]:
        """Processing article with layperson"""
        logger.info(f"Starting article processing {article_data.get('article_id', 'unknown')} with layperson")
        
        # Determine language from article data
        article_language = article_data.get('language', 'en')
        if article_language != self.language:
            logger.info(f"Changing language from {self.language} to {article_language}")
            self.language = article_language
            self.load_agents(article_language)
        
        # 1. Layperson Evaluation
        layperson_result = await self.layperson_evaluation(
            final_statement, 
            article_data['article']
        )
        
        agreement_score = int(layperson_result['agreement_score'])
        controversial_boosted = False
        
        # 2. Controversy Agent (if layperson agrees with 5)
        if agreement_score == 5:
            logger.info("Layperson agreed with 5, starting Controversy Agent")
            
            previous_attempts = ""
            for attempt in range(3):
                logger.info(f"Controversy Agent (layperson) - passing history: '{previous_attempts}'")
                controversy_result = await self.controversy_boost(
                    final_statement, 
                    article_data['article'],
                    previous_attempts
                )
                
                boosted_statement = controversy_result['boosted_statement']
                
                # Always update final_statement to the latest from Controversy Agent
                final_statement = boosted_statement
                controversial_boosted = True
                
                # Re-evaluation by layperson
                new_layperson_result = await self.layperson_evaluation(
                    boosted_statement, 
                    article_data['article']
                )
                
                new_agreement_score = int(new_layperson_result['agreement_score'])
                
                if new_agreement_score < 5:
                    layperson_result = new_layperson_result
                    agreement_score = new_agreement_score
                    logger.info(f"Controversy Agent successful in attempt {attempt + 1}")
                    break
                else:
                    logger.info(f"Controversy Agent attempt {attempt + 1} failed")
                    # Add failed attempt to history
                    if previous_attempts:
                        previous_attempts += f"\nAttempt {attempt + 1} (failed): {boosted_statement}"
                    else:
                        previous_attempts = f"Attempt {attempt + 1} (failed): {boosted_statement}"
        
        # 3. Refine Agent (if layperson doesn't agree with 5)
        dialog_rounds = []
        current_article = article_data['article']
        article_history = ""
        
        if agreement_score < 5:
            logger.info("Layperson didn't agree with 5, starting Refine Agent")
            
            for round_num in range(6):  # 0-5 rounds
                logger.info(f"Layperson refinement round {round_num}")
                
                # Article refinement
                refine_result = await self.refine_article(
                    original_title=article_data['title'],
                    original_article=article_data['article'],
                    current_article=current_article,
                    statement=final_statement,
                    feedback=layperson_result['detailed_feedback'],
                    missing_information=layperson_result.get('missing_information', []),
                    suggested_improvements=layperson_result.get('suggested_improvements', []),
                    previous_rounds=article_history,
                    audience_type="layperson"
                )
                
                refined_article = refine_result['refined_article']
                persuasion_strategies = refine_result.get('persuasion_strategies', [])
                extracted_domain_changes = refine_result.get('extracted_domain_changes', [])
                
                # Update history
                article_history += f"\nRound {round_num}: {refined_article}\n"
                
                # Re-evaluation by layperson
                new_layperson_result = await self.layperson_evaluation(
                    final_statement, 
                    refined_article,
                    article_history
                )
                
                new_agreement_score = int(new_layperson_result['agreement_score'])
                
                # Save round
                round_data = {
                    "round": round_num,
                    "audience_feedback": new_layperson_result['detailed_feedback'],
                    "agreement_score": new_agreement_score,
                    "refined_article": refined_article,
                    "persuasion_strategies": persuasion_strategies,
                    "extracted_domain_changes": extracted_domain_changes,
                    "missing_information": new_layperson_result.get('missing_information', []),
                    "suggested_improvements": new_layperson_result.get('suggested_improvements', [])
                }
                
                dialog_rounds.append(round_data)
                
                # Update for next round
                current_article = refined_article
                layperson_result = new_layperson_result
                agreement_score = new_agreement_score
                
                logger.info(f"Layperson round {round_num} completed, score: {agreement_score}")
                
                # If 5 is achieved, can break
                if agreement_score == 5:
                    logger.info("Score 5 achieved, ending refinement")
                    break
        else:
            # If layperson immediately agreed with <5, add round 0
            round_data = {
                "round": 0,
                "audience_feedback": layperson_result['detailed_feedback'],
                "agreement_score": agreement_score,
                "refined_article": article_data['article'],
                "persuasion_strategies": [],
                "extracted_domain_changes": [],
                "missing_information": layperson_result.get('missing_information', []),
                "suggested_improvements": layperson_result.get('suggested_improvements', [])
            }
            dialog_rounds.append(round_data)
        
        # Prepare results
        result = {
            "original_sentiment": article_data.get('gpt_article_rating'),
            "original_title": article_data['title'],
            "date": article_data.get('date', ''),
            "article_id": article_data['article_id'],
            "language": article_data['language'],
            "original_article": article_data['article'],
            "initial_ne_statement": final_statement,  # Using final_statement from expert
            "final_ne_statement": final_statement,
            "controversial_boosted": controversial_boosted,
            "audience_type": "layperson",
            "dialog_rounds": dialog_rounds,
            "final_refined_article": current_article,
            "final_agreement_score": agreement_score,
            "initial_agreement_score": int(layperson_result['agreement_score']) if not controversial_boosted else agreement_score
        }
        
        # Dodanie skumulowanych zmian
        self._add_cumulative_changes(result)
        
        logger.info(f"Zakończono przetwarzanie z laikiem, final score: {agreement_score}")
        return result
    
    def _add_cumulative_changes(self, result: Dict[str, Any]):
        """Dodanie skumulowanych zmian do wyniku"""
        dialog_rounds = result.get('dialog_rounds', [])
        
        # Skumulowane zmiany lingwistyczne
        linguistic_changes = []
        for round_data in dialog_rounds:
            strategies = round_data.get('persuasion_strategies', [])
            for strategy in strategies:
                strategy_name = strategy.get('strategy', '')
                if strategy_name and strategy_name not in linguistic_changes:
                    linguistic_changes.append(strategy_name)
        
        # Skumulowane zmiany domenowe
        domain_changes = []
        for round_data in dialog_rounds:
            changes = round_data.get('extracted_domain_changes', [])
            for change in changes:
                if change and change not in domain_changes:
                    domain_changes.append(change)
        
        # Cumulative missing information
        missing_information = []
        for round_data in dialog_rounds:
            missing = round_data.get('missing_information', [])
            for info in missing:
                if info and info not in missing_information:
                    missing_information.append(info)
        
        # Cumulative suggested improvements
        suggested_improvements = []
        for round_data in dialog_rounds:
            improvements = round_data.get('suggested_improvements', [])
            for improvement in improvements:
                if improvement and improvement not in suggested_improvements:
                    suggested_improvements.append(improvement)
        
        # Add to result
        result['extracted_linguistic_changes'] = linguistic_changes
        result['extracted_domain_changes'] = domain_changes
        result['cumulative_missing_information'] = missing_information
        result['cumulative_suggested_improvements'] = suggested_improvements
        
        # Add addressed information and improvements
        addressed_missing = []
        addressed_suggested = []
        
        for round_data in dialog_rounds:
            strategies = round_data.get('persuasion_strategies', [])
            for strategy in strategies:
                changes = strategy.get('linguistic_changes', [])
                for change in changes:
                    addressed_missing.extend(change.get('addressed_missing_information', []))
                    addressed_suggested.extend(change.get('addressed_suggested_improvements', []))
        
        result['addressed_missing_information'] = list(set(addressed_missing))
        result['addressed_suggested_improvements'] = list(set(addressed_suggested))
    
    async def process_dataset(self, dataset_path: str, output_dir: str = "results"):
        """Processing entire dataset"""
        logger.info(f"Starting dataset processing: {dataset_path}")
        
        # Create results directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        logger.info(f"Loaded {len(dataset)} articles")
        
        for i, article_data in enumerate(dataset):
            try:
                logger.info(f"Processing article {i+1}/{len(dataset)} (ID: {article_data.get('article_id', 'unknown')})")
                
                # Processing with expert
                expert_result = await self.process_article_with_expert(article_data)
                
                # Processing with layperson (using final_statement from expert)
                layperson_result = await self.process_article_with_layperson(
                    article_data, 
                    expert_result['final_ne_statement']
                )
                
                # Save results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Save expert result
                expert_filename = f"{output_dir}/expert_result_article_{article_data.get('article_id', i+1)}_{timestamp}.json"
                with open(expert_filename, 'w', encoding='utf-8') as f:
                    json.dump(expert_result, f, ensure_ascii=False, indent=2)
                
                # Save layperson result
                layperson_filename = f"{output_dir}/layperson_result_article_{article_data.get('article_id', i+1)}_{timestamp}.json"
                with open(layperson_filename, 'w', encoding='utf-8') as f:
                    json.dump(layperson_result, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Saved results for article {article_data.get('article_id', i+1)}")
                
                # Short pause between articles
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error processing article {article_data.get('article_id', i+1)}: {e}")
                continue
        
        logger.info("Dataset processing completed")

async def main():
    """Main function"""
    # Check API key
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY in environment variables")
    
    # Initialize system
    system = MultiAgentSystem(api_key)
    
    # Process dataset
    await system.process_dataset('Dataset_balanced_more_final_1591.json')

if __name__ == "__main__":
    asyncio.run(main())
