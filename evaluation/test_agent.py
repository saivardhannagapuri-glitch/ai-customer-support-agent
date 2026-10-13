from src.langgraph_agent import (
    prompt,
    retrieve_documents,
    retrieval_fallback,
    empty_question_response,
    check_retrieval,
)

passed = 0
failed = 0
total = 10


def run_test(name, condition):
    global passed, failed

    if condition:
        print(f"{name}: PASS")
        passed += 1
    else:
        print(f"{name}: FAIL")
        failed += 1


def make_state(question, documents=None):
    return {
        "question": question,
        "retrieved_documents": documents or [],
        "answer": "",
        "sources": [],
    }


def get_system_prompt():
    templates = []

    for message in prompt.messages:
        if hasattr(message, "prompt"):
            template = getattr(message.prompt, "template", "")
            if template:
                templates.append(template)

    return " ".join(templates).lower()


print("\n--- Final AI Agent Evaluation ---\n")

# 1. Empty question routing
empty_state = make_state("")
empty_result = retrieve_documents(empty_state)
empty_state.update(empty_result)

run_test(
    "Empty question routes correctly",
    check_retrieval(empty_state) == "empty",
)

# 2. Empty question response
run_test(
    "Empty question returns a clear message",
    empty_question_response(empty_state)["answer"]
    == "Please enter a question.",
)

# 3. Unrelated question retrieval
unrelated_state = make_state("What is the capital of India?")
unrelated_result = retrieve_documents(unrelated_state)

run_test(
    "Unrelated question retrieves no relevant FAQ",
    len(unrelated_result["sources"]) == 0,
)

# 4. Unrelated question routing
unrelated_state.update(unrelated_result)

run_test(
    "Unrelated question routes to fallback",
    check_retrieval(unrelated_state) == "fallback",
)

# 5. Fallback source validation
fallback_result = retrieval_fallback(unrelated_state)

run_test(
    "Fallback does not return unsupported sources",
    fallback_result["sources"] == [],
)

# Check the actual prompt wording
system_prompt = get_system_prompt()

# 6. FAQ context restriction
run_test(
    "Prompt limits answers to company FAQ",
    "faq context" in system_prompt,
)

# 7. Prompt-injection protection
run_test(
    "Prompt includes prompt-injection protection",
    "ignore these rules" in system_prompt,
)

# 8. Hidden instruction protection
run_test(
    "Prompt protects hidden instructions",
    "hidden system instructions" in system_prompt,
)

# 9. Prevent invented information
run_test(
    "Prompt prohibits invented information",
    "never invent" in system_prompt,
)

# 10. Restrict unrelated questions
run_test(
    "Prompt restricts unrelated general knowledge",
    "unrelated general-knowledge questions" in system_prompt,
)

print("\n--- Evaluation Summary ---")
print(f"Passed: {passed}/{total}")
print(f"Failed: {failed}/{total}")

if failed == 0:
    print("Result: All tests passed!")
else:
    print("Result: Some checks failed. Review the failures above.")