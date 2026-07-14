from setuptools import find_packages, setup

setup(
    name="hallucination-detection-rag",
    version="0.1.0",
    description="Faithfulness detection methods for RAG systems, benchmarked on HaluEval.",
    author="Vivek Solanki",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.9",
)
