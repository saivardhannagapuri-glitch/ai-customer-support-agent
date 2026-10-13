
from src.langgraph_agent import graph

test_cases = [
    {
        "question": "How can I reset my password?",
        "expected": "Forgot Password",
        "should_have_source": True,
    },
    {
        "question": "How long does a refund take?",
        "expected": "5 to 7 business days",
        "should_have_source": True,
    },
    {
        "question": "How can I check my order status?",
        "expected": "My Orders",
        "should_have_source": True,
    },
    {
        "question": "What is the capital of India?",
        "expected": None,
        "should_have_source": False,
    },
]

passed = 0

for index, test in enumerate(test_cases, start=1):
    print(f"\nTest {index}: {test['question']}")

    result = graph.invoke({
        "question": test["question"],
        "retrieved_documents": [],
        "answer": "",
        "sources": [],
    })

    answer = result["answer"]
    sources = result.get("sources", [])

    print("Answer:", answer)
    print("Sources:", sources)

    answer_ok = (
        test["expected"] is None
        or test["expected"].lower() in answer.lower()
    )

    source_ok = bool(sources) == test["should_have_source"]

    if answer_ok and source_ok:
        print("Result: PASS")
        passed += 1
    else:
        print("Result: FAIL")

print(f"\nFinal score: {passed}/{len(test_cases)} tests passed")