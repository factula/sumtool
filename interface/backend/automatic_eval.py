from typing import Tuple

from sumtool.round_trip_consistency import RoundTripConsistency


def round_trip_consistency(document: str, summary: str) -> Tuple[str]:
    """Performs automatic evaluation via round-trip consistency.
    
    Args:
        document (str): document being summarized
        summary (str): summary of document, either human- or machine-written
    
    Returns:
        Tuple (str):
            sum_answer: answer extracted from summary
            question:   question for answer
            doc_answer: answer for question extracted from document
    
    """
    rt = RoundTripConsistency()
    sum_answer = rt.extract_answer(summary)
    question = rt.generate_question(context=summary, answer=sum_answer)
    doc_answer = rt.answer_question(question=question, context=document)
    
    return (sum_answer, question, doc_answer)


