from hallucination_check import tfidf_similarity, check_hallucination

def test_identical_texts_give_high_similarity():
    score = tfidf_similarity("Paris is the capital of France", "Paris is the capital of France")
    assert score > 0.9

def test_unrelated_texts_give_low_similarity():
    score = tfidf_similarity("Paris is in France", "The moon orbits the earth")
    assert score < 0.3

def test_hallucination_check_flags_extra_entities():
    result = check_hallucination(
        reference="Python was created by Guido van Rossum.",
        generated="Python was created by Guido van Rossum and Elon Musk."
    )
    assert "Elon Musk" in result["hallucinated_entities"]

def test_grounded_output_passes():
    result = check_hallucination(
        reference="The Eiffel Tower is in Paris.",
        generated="The Eiffel Tower is located in Paris."
    )
    assert result["is_grounded"] == True