import os
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langsmith import Client

from app.services.logger import setup_logger

logger = setup_logger(__name__)

EVAL_PROMPT = PromptTemplate(
    input_variables=["source_documents", "quiz_questions"],
    template="""You are evaluating quiz questions based on source documents. Analyze the questions using the following criteria and provide numerical scores (1-5) for each:

    Source Documents:
    {source_documents}

    Quiz Questions:
    {quiz_questions}

    Please evaluate and score each criterion on a scale of 1-5 (where 1 is lowest and 5 is highest):

    1. Content Alignment (1-5):
    - How well do questions match the source content?
    - Are key concepts from documents represented?
    - Is the information accurate according to sources?

    2. Uniqueness (1-5):
    - Are questions distinct from each other?
    - Do they test different aspects of the content?
    - Is there any redundancy in concepts tested?

    3. Coverage (1-5):
    - How well are all documents represented?
    - Is there balanced coverage across documents?
    - Are important topics from each document included?

    Provide your evaluation in the following JSON format:
    {{
        "content_alignment": {{
            "score": <number 1-5>,
            "reasoning": "<detailed explanation>"
        }},
        "uniqueness": {{
            "score": <number 1-5>,
            "reasoning": "<detailed explanation>"
        }},
        "coverage": {{
            "score": <number 1-5>,
            "reasoning": "<detailed explanation>"
        }},
        "overall_feedback": "<general feedback and suggestions for improvement>"
    }}
    """
)

class QuizEvaluator:   
    def __init__(self, model_name: str = "gpt-4o-mini", verbose: bool = False):
        self.llm = ChatOpenAI(model_name=model_name, temperature=0)
        self.eval_chain = EVAL_PROMPT | self.llm | JsonOutputParser()
        self.verbose = verbose

        
    def evaluate_quiz(self, source_documents: List[str], quiz_questions: List[Dict]) -> Dict:
        """
        Evaluate quiz questions against source documents.
        
        Args:
            source_documents (List[str]): List of source documents
            quiz_questions (List[Dict]): List of questions with their answers
            
        Returns:
            Dict: Evaluation results with scores and reasoning
        """
        # Format documents and questions for evaluation
        formatted_docs = "\n\n--- Document {} ---\n{}".format
        documents_text = "\n".join(
            formatted_docs(i+1, doc) for i, doc in enumerate(source_documents)
        )
        
        formatted_questions = "\n\n--- Question {} ---\n{}".format
        questions_text = "\n".join(
            formatted_questions(i+1, str(q)) for i, q in enumerate(quiz_questions)
        )
        
        # Run evaluation
        evaluation = self.eval_chain.invoke({
            "source_documents": documents_text,
            "quiz_questions": questions_text
        })
            
        # Calculate overall score
        scores = [
            evaluation["content_alignment"]["score"],
            evaluation["uniqueness"]["score"],
            evaluation["coverage"]["score"]
        ]
        evaluation["overall_score"] = round(sum(scores) / len(scores), 0)
        
        return evaluation
            
    def log_evaluation(self, evaluation: Dict, run_id):
        """
        Log evaluation results.
        
        Args:
            evaluation (Dict): Evaluation results
        """
        smith_client = Client()
        smith_client.create_feedback(
            run_id=run_id, key="content_alignment", 
            value=evaluation['content_alignment']["score"],
            comment=evaluation['content_alignment']["reasoning"])
        smith_client.create_feedback(
            run_id=run_id, 
            key="uniqueness", 
            value=evaluation['uniqueness']["score"],
            comment=evaluation['uniqueness']["reasoning"])
        smith_client.create_feedback(
            run_id=run_id, 
            key="coverage", 
            value=evaluation['coverage']["score"],
            comment=evaluation['coverage']["reasoning"])
        smith_client.create_feedback(
            run_id=run_id, 
            key="overall_score", 
            value=evaluation['overall_score'],
            comment=evaluation['overall_feedback'])
        

    def invoke(self, inputs: Dict) -> Dict:
        try:
            logger.info("Evaluating quiz questions...") if self.verbose else None
            # Run evaluation
            evaluation = self.evaluate_quiz(inputs["source_documents"], inputs["quiz_questions"]["questions_list"])
            self.log_evaluation(evaluation, inputs["run_id"])
            if self.verbose:
                logger.info("Quiz Evaluation Results:")
                logger.info(f"Content Alignment: {evaluation['content_alignment']}")
                logger.info(f"Uniqueness: {evaluation['uniqueness']}")
                logger.info(f"Coverage: {evaluation['coverage']}")
                logger.info(f"Overall Feedback: {evaluation['overall_feedback']}")

            # Return quiz questions to maintain consistency with other tools
            return {"quiz_questions": inputs["quiz_questions"], "evaluation": evaluation}  
        except Exception as e:
            if self.verbose:
                logger.error(f"Error evaluating quiz questions: {e}")
            return {"quiz_questions": inputs["quiz_questions"]}

def main():
    # Initialize evaluator
    evaluator = QuizEvaluator()
    
    # Example source documents
    source_documents = [
        """The Python programming language was created by Guido van Rossum 
        and was released in 1991. Python is known for its simple syntax 
        and readability.""",
        """Python supports multiple programming paradigms, including 
        procedural, object-oriented, and functional programming."""
    ]
    
    # Example quiz questions
    quiz_questions = [
        {
            "question": "Who created Python?",
            "options": [
                "Guido van Rossum",
                "James Gosling",
                "Bjarne Stroustrup",
                "Larry Wall"
            ],
            "correct_answer": "Guido van Rossum"
        },
        {
            "question": "What programming paradigms does Python support?",
            "options": [
                "Only procedural",
                "Only object-oriented",
                "Procedural, object-oriented, and functional",
                "Only functional"
            ],
            "correct_answer": "Procedural, object-oriented, and functional"
        }
    ]
    
    # Run evaluation
    results = evaluator.evaluate_quiz(source_documents, quiz_questions)
    print(results["evaluation_summary"])

if __name__ == "__main__":
    main()