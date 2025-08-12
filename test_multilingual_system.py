import json
import asyncio
import os
from dotenv import load_dotenv
from multi_agent_system import MultiAgentSystem
from loguru import logger

# Load environment variables
load_dotenv()

async def test_language(language_code, language_name):
    """Test system for specific language"""
    logger.info(f"Testing language: {language_name} ({language_code})")
    
    try:
        # Initialize system with given language
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY in environment variables")
        
        system = MultiAgentSystem(api_key, language=language_code)
        
        # Sample article in given language
        test_articles = {
            'cs': {
                "title": "Česká republika zvažuje výstavbu jaderné elektrárny",
                "article": "Česká republika zvažuje možnosti výstavby nové jaderné elektrárny jako součást svého plánu na snížení emisí oxidu uhličitého. Ministerstvo průmyslu a obchodu oznámilo, že země potřebuje stabilní zdroj energie pro budoucnost. Odborníci se shodují, že jaderná energie může být klíčová pro dosažení klimatických cílů."
            },
            'sk': {
                "title": "Slovensko uvažuje o výstavbe jadrovej elektrárne",
                "article": "Slovensko uvažuje o možnostiach výstavby novej jadrovej elektrárne ako súčasť svojho plánu na zníženie emisií oxidu uhličitého. Ministerstvo priemyslu a obchodu oznámilo, že krajina potrebuje stabilný zdroj energie pre budúcnosť. Odborníci sa zhodujú, že jadrová energia môže byť kľúčová pre dosiahnutie klimatických cieľov."
            },
            'hu': {
                "title": "Magyarország fontolóra veszi az atomerőmű építését",
                "article": "Magyarország fontolóra veszi az új atomerőmű építésének lehetőségét a szén-dioxid-kibocsátás csökkentésére vonatkozó tervének részeként. Az Ipari és Kereskedelmi Minisztérium bejelentette, hogy az országnak stabil energiaforrásra van szüksége a jövőre nézve. A szakértők egyetértenek abban, hogy a nukleáris energia kulcsfontosságú lehet a klímacélok elérésében."
            }
        }
        
        if language_code not in test_articles:
            logger.warning(f"No test article for language {language_code}")
            return False
        
        test_article = test_articles[language_code]
        
        # Test Theory of Mind agent
        logger.info(f"Testing Theory of Mind agent for {language_name}")
        tom_result = await system.theory_of_mind_analysis(
            test_article['title'], 
            test_article['article']
        )
        
        if not tom_result or 'nuclear_energy_statement' not in tom_result:
            logger.error(f"Error in Theory of Mind agent for {language_name}")
            return False
        
        logger.info(f"Theory of Mind agent works for {language_name}")
        logger.info(f"   Generated statement: {tom_result['nuclear_energy_statement'][:100]}...")
        
        # Test Expert agent
        logger.info(f"Testing Expert agent for {language_name}")
        expert_result = await system.expert_evaluation(
            tom_result['nuclear_energy_statement'], 
            test_article['article']
        )
        
        if not expert_result or 'agreement_score' not in expert_result:
            logger.error(f"Error in Expert agent for {language_name}")
            return False
        
        logger.info(f"Expert agent works for {language_name}")
        logger.info(f"   Score: {expert_result['agreement_score']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error testing {language_name}: {e}")
        return False

async def main():
    """Main test function"""
    logger.info("Starting multilingual system test")
    
    # List of languages to test
    languages = [
        ('cs', 'Czech'),
        ('sk', 'Slovak'), 
        ('hu', 'Hungarian')
    ]
    
    results = {}
    
    for lang_code, lang_name in languages:
        success = await test_language(lang_code, lang_name)
        results[lang_name] = success
        
        # Short pause between tests
        await asyncio.sleep(2)
    
    # Summary of results
    logger.info("Test summary:")
    for lang_name, success in results.items():
        status = "WORKS" if success else "ERROR"
        logger.info(f"   {lang_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        logger.info("All tests passed successfully!")
    else:
        logger.error("Some tests failed!")
    
    return all_passed

if __name__ == "__main__":
    asyncio.run(main())
