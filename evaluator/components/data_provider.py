from typing import Tuple, List


def get_data() -> List[Tuple[str, str]]:
    # TODO: should fetch the actual queries from the dataset instead of returning hard-coded queries
    # TODO: the data format should be extended with other useful information according to the dataset
    #  (golden answers, sets of tools, etc.)
    return [
        ("What is the weather in New York?", "weather_info"),
        ("How many words are in 'Hello World, this is a test sentence'?", "word_count"),
        ("Reverse this text: Python Experiment", "reverse_string"),
        ("Convert this to uppercase: llamastack", "uppercase"),
        ("Give me an insurance evaluation score", "insurance_scorer")
    ]
