# import the retrieval and answer generation components from the main module
from app import chain, retriever

# create test questions and expected keywords to validate responses
test_cases = [
    {
        "question": "What do people think of the cheese pizza?", 
        "expected_keywords": ["cheese", "crust"]
     },
    {
        "question": "How’s the service quality?", 
        "expected_keywords": ["friendly", "slow", "attentive"]
    },
]
# loop through each test case
for test in test_cases:

    # use the retriever to get relevant reviews for each question
    reviews = retriever.invoke(test["question"])

    # pass the question and reviews into the chain to generate an answer
    result = chain.invoke({"question": test["question"], "reviews": reviews})

    # print the question and answer
    print(f"Q: {test['question']}\nA: {result}\n")

    # check if each expected keyword appears in the output
    for keyword in test["expected_keywords"]:
        print(f"✓ {keyword} in output: {keyword in result}")
    print("\n---\n")
