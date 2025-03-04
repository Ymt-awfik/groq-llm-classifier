import os
import time
from dotenv import load_dotenv
import groq

# Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class LLMClient:
    """Handles interactions with the Groq API."""
    
    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is missing in .env file.")
        
        self.client = groq.Client(api_key=GROQ_API_KEY)
        self.model = "llama3-70b-8192"  # Choose an appropriate model

    def complete(self, prompt, max_tokens=100, temperature=0.7):
        """Send a request to the Groq API and return the model's response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error: {e}")
            return None


# PART 2: STRUCTURED COMPLETIONS
def create_structured_prompt(text, question):
    """Creates a structured prompt to get well-formatted responses."""
    prompt = f"""
    # Analysis Report

    ## Input Text
    {text}

    ## Question
    {question}

    ## Analysis
    """
    return prompt


def extract_section(completion, section_start, section_end=None):
    """Extracts content between section_start and section_end from a response."""
    start_idx = completion.find(section_start)
    if start_idx == -1:
        return None

    start_idx += len(section_start)
    
    if section_end is None:
        return completion[start_idx:].strip()

    end_idx = completion.find(section_end, start_idx)
    if end_idx == -1:
        return completion[start_idx:].strip()

    return completion[start_idx:end_idx].strip()


# PART 3: CLASSIFICATION WITH CONFIDENCE ANALYSIS
def classify_with_confidence(client, text, categories, confidence_threshold=0.8):
    """Classifies text into predefined categories and analyzes confidence."""
    prompt = f"""
    Classify the following text into one of these categories: {', '.join(categories)}.

    Response format:
    1. CATEGORY: [one of: {', '.join(categories)}]
    2. CONFIDENCE: [high|medium|low]
    3. REASONING: [explanation]

    Text to classify:
    {text}
    """
    response = client.complete(prompt, max_tokens=500)
    
    category = extract_section(response, "1. CATEGORY: ", "\n")
    confidence = extract_section(response, "2. CONFIDENCE: ", "\n")

    confidence_score = {"high": 1.0, "medium": 0.7, "low": 0.4}.get(confidence, 0)

    if confidence_score > confidence_threshold:
        return {
            "category": category,
            "confidence": confidence_score,
            "reasoning": extract_section(response, "3. REASONING: ")
        }
    else:
        return {
            "category": "uncertain",
            "confidence": confidence_score,
            "reasoning": "Confidence below threshold"
        }


# PART 4: PROMPT STRATEGY COMPARISON
def compare_prompt_strategies(client, texts, categories):
    """Compares different prompt strategies for classification."""
    strategies = {
        "basic": lambda text: f"Classify this text into one of these categories: {', '.join(categories)}.\n\nText: {text}",
        "structured": lambda text: f"""
        Classification Task
        Categories: {', '.join(categories)}
        Text: {text}
        Classification: """,
        "few_shot": lambda text: f"""
        Here are some examples of text classification:

        Example 1:
        Text: "The product arrived damaged and customer service was unhelpful."
        Classification: Negative

        Example 2:
        Text: "While delivery was slow, the quality exceeded my expectations."
        Classification: Mixed

        Example 3:
        Text: "Absolutely love this! Best purchase I've made all year."
        Classification: Positive

        Now classify this text:
        Text: "{text}"
        Classification: """
    }

    results = {}

    for strategy_name, prompt_func in strategies.items():
        strategy_results = []
        
        for text in texts:
            prompt = prompt_func(text)
            response = client.complete(prompt, max_tokens=200)
            strategy_results.append(response)

        results[strategy_name] = strategy_results

    return results


# MAIN SCRIPT TO TEST IMPLEMENTATION
if __name__ == "__main__":
    client = LLMClient()
    
    # Test Structured Completion
    text = "The product was amazing and delivered on time!"
    question = "What is the sentiment of this review?"
    structured_prompt = create_structured_prompt(text, question)
    structured_response = client.complete(structured_prompt)
    print("\nStructured Completion Result:")
    print(extract_section(structured_response, "## Analysis"))

    # Test Classification with Confidence Analysis
    categories = ["Positive", "Negative", "Neutral"]
    classification_result = classify_with_confidence(client, text, categories)
    print("\nClassification Result:")
    print(classification_result)

    # Test Prompt Strategy Comparison
    sample_texts = [
        "The product was excellent and worked perfectly.",
        "I am disappointed with the service.",
        "It's okay, nothing special but not bad."
    ]
    comparison_results = compare_prompt_strategies(client, sample_texts, categories)
    print("\nPrompt Strategy Comparison Results:")
    for strategy, results in comparison_results.items():
        print(f"\nStrategy: {strategy}")
        for res in results:
            print(res)
