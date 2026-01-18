import os
from serpapi import GoogleSearch
from ddgs import DDGS
from dotenv import load_dotenv

def serp_search(query: str) -> str:
    """
    A practical web search engine tool based on SerpApi.
    It intelligently parses search results, prioritizing direct answers or knowledge graph information.
    """
    print(f"Executing [SerpApi] web search: {query}")
    try:
        load_dotenv()
        api_key = os.getenv("SERPAPI_API_KEY")

        if not api_key:
            return "Error: SERPAPI_API_KEY not configured in .env file."

        params = {
            "q": query,
            "api_key": api_key,
            "google_domain": "google.com",
            "gl": "sg",  # Country code
            "hl": "en", # Language code
        }
        
        client = GoogleSearch(params)
        results = client.get_dict()
        
        # Intelligent parsing: prioritize finding the most direct answer
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # If no direct answer, return summaries of the first three organic results
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"Sorry, no information found about '{query}'."

    except Exception as e:
        return f"Error during search: {str(e)}"

def ddgs_search(query: str) -> str:
    """
    A practical web search engine tool based on DuckDuckGo Search API.
    It retrieves and summarizes search results.
    """
    print(f"Executing [DuckDuckGo] web search: {query}")
    try:

        results = DDGS().text(query, max_results=3)
        if not results:
            return f"Sorry, no information found about '{query}'."

        snippets = [
            f"[{i+1}] {res.get('title', '')}\n{res.get('body', '')}\n{res.get('href', '')}"
            for i, res in enumerate(results)
        ]
        return "\n\n".join(snippets)

    except ImportError:
        return "Error: duckduckgo_search package is not installed."
    except Exception as e:
        return f"Error during search: {str(e)}"

if __name__ == '__main__':
    print("\nDuckDuckGo Search Result:")
    print(ddgs_search("What is the capital of France?"))
    
    print("\n")
    print(ddgs_search("What is NVIDIA's latest GPU model?"))