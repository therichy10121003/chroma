"""
Chroma Generative Features Examples
=====================================

This file demonstrates how to use the new generative features in Chroma,
including RAG (Retrieval-Augmented Generation) with various algorithms
and LLM providers.

Requirements:
    pip install chromadb openai anthropic

Environment variables:
    OPENAI_API_KEY - Your OpenAI API key
    ANTHROPIC_API_KEY - Your Anthropic API key
"""

import chromadb
from chromadb.utils.generative_functions import (
    OpenAIGenerativeFunction,
    AnthropicGenerativeFunction,
)
from chromadb.utils.generative_functions.rag_algorithms import (
    RAGConfig,
    create_rag_algorithm,
)


def example_1_basic_rag():
    """Example 1: Basic RAG with OpenAI."""
    print("=" * 80)
    print("Example 1: Basic RAG with OpenAI")
    print("=" * 80)

    # Initialize Chroma client
    client = chromadb.Client()

    # Create a collection
    collection = client.create_collection(name="tech_docs")

    # Add some documents
    collection.add(
        documents=[
            "Machine learning is a subset of AI that enables systems to learn from data.",
            "Deep learning uses neural networks with multiple layers to process data.",
            "Natural language processing helps computers understand human language.",
            "Computer vision enables machines to interpret and understand visual information.",
            "Reinforcement learning trains agents through rewards and penalties.",
        ],
        ids=["doc1", "doc2", "doc3", "doc4", "doc5"],
        metadatas=[
            {"topic": "ML"},
            {"topic": "DL"},
            {"topic": "NLP"},
            {"topic": "CV"},
            {"topic": "RL"},
        ],
    )

    # Create a generative function
    gen_fn = OpenAIGenerativeFunction(
        model_name="gpt-4",
        temperature=0.7,
        max_tokens=200,
    )

    # Generate a response using RAG
    result = collection.generate(
        query_text="What is machine learning and how does it relate to neural networks?",
        generative_function=gen_fn,
        n_results=3,
        rag_algorithm="simple",
    )

    print(f"\nQuery: What is machine learning and how does it relate to neural networks?")
    print(f"\nGenerated Response:\n{result['response']}")
    print(f"\nSource Documents:")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"{i}. {doc}")

    # Cleanup
    client.delete_collection(name="tech_docs")


def example_2_anthropic_claude():
    """Example 2: RAG with Anthropic Claude."""
    print("\n" + "=" * 80)
    print("Example 2: RAG with Anthropic Claude")
    print("=" * 80)

    # Initialize Chroma client
    client = chromadb.Client()

    # Create a collection
    collection = client.create_collection(name="science_facts")

    # Add documents
    collection.add(
        documents=[
            "The speed of light in vacuum is approximately 299,792,458 meters per second.",
            "DNA carries genetic information in all living organisms.",
            "The Earth's core is primarily composed of iron and nickel.",
            "Photosynthesis converts light energy into chemical energy in plants.",
            "Quantum mechanics describes the behavior of matter at atomic scales.",
        ],
        ids=[f"fact{i}" for i in range(1, 6)],
    )

    # Use Anthropic Claude
    gen_fn = AnthropicGenerativeFunction(
        model_name="claude-3-5-sonnet-20241022",
        temperature=0.5,
        max_tokens=300,
    )

    # Generate with RAG
    result = collection.generate(
        query_text="Explain how energy transformation works in nature.",
        generative_function=gen_fn,
        n_results=2,
    )

    print(f"\nQuery: Explain how energy transformation works in nature.")
    print(f"\nClaude's Response:\n{result['response']}")


def example_3_adaptive_rag():
    """Example 3: Adaptive RAG algorithm."""
    print("\n" + "=" * 80)
    print("Example 3: Adaptive RAG Algorithm")
    print("=" * 80)

    client = chromadb.Client()
    collection = client.create_collection(name="adaptive_test")

    # Add technical documentation
    docs = [
        "Python is a high-level programming language known for readability.",
        "JavaScript is primarily used for web development and runs in browsers.",
        "Rust provides memory safety without garbage collection.",
        "Go is designed for concurrent programming and scalability.",
        "TypeScript adds static typing to JavaScript.",
        "C++ offers low-level memory manipulation and high performance.",
        "Java uses virtual machines for platform independence.",
        "Swift is Apple's language for iOS and macOS development.",
    ]

    collection.add(
        documents=docs,
        ids=[f"lang{i}" for i in range(len(docs))],
    )

    gen_fn = OpenAIGenerativeFunction(model_name="gpt-3.5-turbo")

    # Adaptive RAG adjusts based on query complexity
    simple_query = "What is Python?"
    complex_query = (
        "Compare and contrast the memory management approaches, "
        "type systems, and use cases of Rust, Go, and C++"
    )

    print("\n--- Simple Query (Adaptive RAG will use fewer documents) ---")
    result1 = collection.generate(
        query_text=simple_query,
        generative_function=gen_fn,
        n_results=5,
        rag_algorithm="adaptive",
    )
    print(f"Query: {simple_query}")
    print(f"Response: {result1['response']}")
    print(f"Documents used: {len(result1['source_documents'])}")

    print("\n--- Complex Query (Adaptive RAG will use more documents) ---")
    result2 = collection.generate(
        query_text=complex_query,
        generative_function=gen_fn,
        n_results=5,
        rag_algorithm="adaptive",
    )
    print(f"Query: {complex_query}")
    print(f"Response: {result2['response']}")
    print(f"Documents used: {len(result2['source_documents'])}")

    client.delete_collection(name="adaptive_test")


def example_4_custom_rag_config():
    """Example 4: Custom RAG configuration."""
    print("\n" + "=" * 80)
    print("Example 4: Custom RAG Configuration")
    print("=" * 80)

    client = chromadb.Client()
    collection = client.create_collection(name="custom_rag")

    collection.add(
        documents=[
            "Climate change is caused by greenhouse gas emissions.",
            "Renewable energy sources include solar, wind, and hydro power.",
            "Deforestation contributes to carbon dioxide levels in the atmosphere.",
            "Electric vehicles reduce transportation-related emissions.",
            "Carbon capture technology removes CO2 from the atmosphere.",
        ],
        ids=[f"climate{i}" for i in range(1, 6)],
    )

    gen_fn = OpenAIGenerativeFunction(
        model_name="gpt-4",
        system_prompt=(
            "You are an environmental science expert. "
            "Provide accurate, science-based answers."
        ),
    )

    # Generate with custom RAG parameters
    result = collection.generate(
        query_text="What are the main solutions to climate change?",
        generative_function=gen_fn,
        n_results=5,
        rag_algorithm="simple",
        # RAG-specific parameters
        relevance_threshold=0.5,
        max_context_length=2000,
        # Generation parameters
        temperature=0.3,  # Lower temperature for more factual responses
        max_tokens=400,
    )

    print(f"\nQuery: What are the main solutions to climate change?")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nRelevance scores: {result['source_distances']}")

    client.delete_collection(name="custom_rag")


def example_5_direct_algorithm_usage():
    """Example 5: Using RAG algorithms directly."""
    print("\n" + "=" * 80)
    print("Example 5: Direct RAG Algorithm Usage")
    print("=" * 80)

    # Create different RAG algorithms
    simple_rag = create_rag_algorithm("simple", RAGConfig(top_k=3))
    adaptive_rag = create_rag_algorithm("adaptive", RAGConfig(top_k=5))

    # Sample documents and distances
    documents = [
        "Document about AI and machine learning",
        "Document about deep neural networks",
        "Document about data science",
        "Document about computer vision",
        "Document about natural language processing",
    ]
    distances = [0.1, 0.15, 0.3, 0.4, 0.45]

    # Simple RAG
    print("\n--- Simple RAG ---")
    simple_context = simple_rag.prepare_context(documents, distances)
    print(f"Selected {len(simple_context)} documents:")
    for doc in simple_context:
        print(f"  - {doc}")

    # Adaptive RAG
    print("\n--- Adaptive RAG ---")
    query = "Tell me everything about artificial intelligence and its applications"
    adaptive_context = adaptive_rag.prepare_context(
        documents, distances, query=query
    )
    print(f"Selected {len(adaptive_context)} documents for complex query:")
    for doc in adaptive_context:
        print(f"  - {doc}")


def example_6_batch_generation():
    """Example 6: Batch generation with RAG."""
    print("\n" + "=" * 80)
    print("Example 6: Batch Generation")
    print("=" * 80)

    client = chromadb.Client()
    collection = client.create_collection(name="batch_test")

    collection.add(
        documents=[
            "The mitochondria is the powerhouse of the cell.",
            "DNA replication occurs during the S phase of the cell cycle.",
            "Proteins are synthesized by ribosomes using mRNA as a template.",
            "Cellular respiration produces ATP from glucose.",
        ],
        ids=[f"bio{i}" for i in range(1, 5)],
    )

    gen_fn = OpenAIGenerativeFunction(model_name="gpt-3.5-turbo")

    # Multiple queries
    queries = [
        "What produces energy in cells?",
        "How are proteins made?",
        "What is DNA replication?",
    ]

    print("\n--- Batch Processing Multiple Queries ---")
    for query in queries:
        result = collection.generate(
            query_text=query,
            generative_function=gen_fn,
            n_results=2,
        )
        print(f"\nQ: {query}")
        print(f"A: {result['response']}")

    client.delete_collection(name="batch_test")


def main():
    """Run all examples."""
    print("\n")
    print("*" * 80)
    print("CHROMA GENERATIVE FEATURES EXAMPLES")
    print("*" * 80)

    examples = [
        ("Basic RAG with OpenAI", example_1_basic_rag),
        ("RAG with Anthropic Claude", example_2_anthropic_claude),
        ("Adaptive RAG Algorithm", example_3_adaptive_rag),
        ("Custom RAG Configuration", example_4_custom_rag_config),
        ("Direct Algorithm Usage", example_5_direct_algorithm_usage),
        ("Batch Generation", example_6_batch_generation),
    ]

    for name, example_fn in examples:
        try:
            example_fn()
        except Exception as e:
            print(f"\n❌ Example '{name}' failed: {e}")
            print("This might be due to missing API keys or dependencies.")
            continue

    print("\n" + "*" * 80)
    print("Examples completed!")
    print("*" * 80 + "\n")


if __name__ == "__main__":
    main()
