from src.langgraph_agent import retrieve_documents

passed = 0
total = 6


def run_test(name, condition):
    global passed

    if condition:
        print(f"{name}: PASS")
        passed += 1
    else:
        print(f"{name}: FAIL")


def test_retrieval(question, expected_keyword):
    state = {
        "question": question,
        "retrieved_documents": [],
        "answer": "",
        "sources": [],
    }

    result = retrieve_documents(state)

    sources = [
        source.lower()
        for source in result["sources"]
    ]

    return any(
        expected_keyword.lower() in source
        for source in sources
    )


print("\n--- Retrieval Quality Tests ---\n")

run_test(
    "Password reset retrieval",
    test_retrieval(
        "How can I reset my password?",
        "password"
    )
)

run_test(
    "Refund duration retrieval",
    test_retrieval(
        "How long do refunds take?",
        "refund"
    )
)

run_test(
    "Order status retrieval",
    test_retrieval(
        "How can I check my order status?",
        "order status"
    )
)

run_test(
    "Cancel order retrieval",
    test_retrieval(
        "How can I cancel my order?",
        "cancel"
    )
)

run_test(
    "Locked account retrieval",
    test_retrieval(
        "What should I do if my account is locked?",
        "locked"
    )
)

unrelated_state = {
    "question": "What is the capital of India?",
    "retrieved_documents": [],
    "answer": "",
    "sources": [],
}

unrelated_result = retrieve_documents(unrelated_state)

run_test(
    "Unrelated question retrieves no sources",
    len(unrelated_result["sources"]) == 0
)

print(f"\nFinal score: {passed}/{total} tests passed")