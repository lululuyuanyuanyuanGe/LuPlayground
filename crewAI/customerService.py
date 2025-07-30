import warnings
warnings.filterwarnings("ignore")

from crewai import Agent, Task, Crew
from crewai_tools import ScrapeWebsiteTool
from models import CustomerLLM

import os
import requests
from typing import List
from dotenv import load_dotenv

# Import ChromaDB types for custom embedder
from chromadb import Documents, EmbeddingFunction, Embeddings

load_dotenv()

silicon_flow_api_key = os.getenv("SILICON_FLOW_API_KEY")

# Custom LLM
custoer_llm = CustomerLLM(
    model="Qwen/Qwen3-235B-A22B-Instruct-2507",
    api_key=silicon_flow_api_key,
    endpoint="https://api.siliconflow.cn/v1/chat/completions",
)

# Custom Embedder Implementation
class SiliconFlowEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model: str = "Qwen/Qwen3-Embedding-8B"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.siliconflow.cn/v1/embeddings"
    
    def __call__(self, input: Documents) -> Embeddings:
        """
        Generate embeddings for a list of documents.
        
        Args:
            input (Documents): List of text documents to embed
            
        Returns:
            Embeddings: List of embedding vectors
        """
        embeddings = []
        
        for text in input:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model,
                "input": text
            }
            
            try:
                response = requests.post(self.base_url, headers=headers, json=data)
                response.raise_for_status()
                
                result = response.json()
                # Adjust this based on Silicon Flow's response format
                embedding = result["data"][0]["embedding"]
                embeddings.append(embedding)
                
            except Exception as e:
                print(f"Error generating embedding for text: {text[:50]}...")
                print(f"Error: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)  # Adjust dimension as needed
        
        return embeddings

# Create embedder instance
silicon_flow_embedder = SiliconFlowEmbeddingFunction(
    api_key=silicon_flow_api_key,
    model="Qwen/Qwen3-Embedding-8B"
)

# Your existing agents
support_agent = Agent(
    role="Senior Support Representative",
    goal="Be the most friendly and helpful support representative in your team",
    backstory=(
        "You work at crewAI (https://crewai.com) and are now working on providing "
        "support to {customer}, a super important customer for your company. "
        "You need to make sure that you provide the best support! "
        "Make sure to provide full complete answers, and make no assumptions."
    ),
    llm=custoer_llm,
    allow_delegation=False,
    verbose=True
)

support_quality_assurance_agent = Agent(
    role="Support Quality Assurance Specialist",
    goal="Get recognition for providing the best support quality assurance in your team",
    backstory=(
        "You work at crewAI (https://crewai.com) and are now working with your team "
        "on a request from {customer} ensuring that the support representative is "
        "providing the best support possible. You need to make sure that the support "
        "representative is providing full complete answers, and make no assumptions."
    ),
    llm=custoer_llm,
    verbose=True
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://docs.crewai.com/how-to/Creating-a-Crew-and-kick-it-off/"
)

# Your existing tasks
inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
        "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
        "Make sure to use everything you know to provide the best support possible. "
        "You must strive to provide a complete and accurate response to the customer's inquiry."
    ),
    expected_output=(
        "A detailed, informative response to the customer's inquiry that addresses "
        "all aspects of their question. The response should include references "
        "to everything you used to find the answer, including external data or solutions. "
        "Ensure the answer is complete, leaving no questions unanswered, and maintain "
        "a helpful and friendly tone throughout."
    ),
    tools=[docs_scrape_tool],
    agent=support_agent,
)

quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Senior Support Representative for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
        "high-quality standards expected for customer support. "
        "Verify that all parts of the customer's inquiry have been addressed "
        "thoroughly, with a helpful and friendly tone. "
        "Check for references and sources used to find the information, "
        "ensuring the response is well-supported and leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response ready to be sent to the customer. "
        "This response should fully address the customer's inquiry, incorporating all "
        "relevant feedback and improvements. "
        "Don't be too formal, we are a chill and cool company "
        "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)

# Create Crew with custom embedder
crew = Crew(
    agents=[support_agent, support_quality_assurance_agent],
    tasks=[inquiry_resolution, quality_assurance_review],
    verbose=True,
    embedder={
        "provider": "custom",
        "config": {
            "embedder": silicon_flow_embedder  # Pass the instance directly
        }
    },
    memory=True
)

inputs = {
    "customer": "DeepLearningAI",
    "person": "Andrew Ng",
    "inquiry": "I need help with setting up a Crew and kicking it off, specifically "
               "how can I add memory to my crew? Can you provide guidance?"
}

result = crew.kickoff(inputs=inputs)

from IPython.display import Markdown
Markdown(result)
