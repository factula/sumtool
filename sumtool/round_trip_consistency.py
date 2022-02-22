import argparse

import tqdm
import pandas as pd

from datasets import load_dataset
from transformers import (
    DistilBertTokenizer, AutoTokenizer, pipeline, 
    AutoModelForSeq2SeqLM, AutoModelWithLMHead, AutoModelForQuestionAnswering
)


class RoundTripConsistency:
    """Performs the round-trip consistency evaluation method from 
    "Synthetic QA Corpora Generation with Roundtrip Consistency"
    (Alberti et al., 2019).

    Uses three models:
    1. Answer extraction (celinelee/answer-extraction)
    2. Question answering (valhalla/t5-small-qa-qg-hl)
    3. Question generation (mrm8488/t5-base-finetuned-question-generation-ap)
    """
    # Model & tokenizer for answer extraction
    # (pipeline is just a model and tokenizer combined)
    ae_pipeline: pipeline
    # Model & tokenizer for question-answering
    qa_tokenizer: AutoTokenizer
    qa_model: AutoModelForSeq2SeqLM
    # Model & tokenizer for question generation
    qg_tokenizer: AutoTokenizer
    qg_model: AutoModelWithLMHead
    def __init__(self):
        ae_model_name = "celinelee/answer-extraction"
        ae_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        ae_model = AutoModelForQuestionAnswering.from_pretrained(ae_model_name)
        self.ae_pipeline = pipeline('question-answering', model=ae_model, tokenizer=ae_tokenizer)

        qa_model_name = "valhalla/t5-small-qa-qg-hl"
        self.qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
        self.qa_model = AutoModelForSeq2SeqLM.from_pretrained(qa_model_name)

        qg_model_name = "mrm8488/t5-base-finetuned-question-generation-ap"
        self.qg_tokenizer = AutoTokenizer.from_pretrained(qg_model_name)
        self.qg_model = AutoModelWithLMHead.from_pretrained(qg_model_name)

    def extract_answer(self, context: str) -> str:
        """ Extract the answer from just the context. p(A | C)
            
        Args:
            context (str): the document from which to extract an answer
            
        Returns:
            answer (str): extracted answer
        """
        output = self.ae_pipeline(context=context, question="answer?")
        answer = output["answer"]
        answer_start = output["start"]
        answer_end = output["end"]
        return answer

    def generate_question(self, context: str, answer: str) -> str:
        """ Generate the questions from answers and context. p(Q | A, C)
        
        Args: 
            context (str): document from which to generate question
            answer (str): answer to generate question from
        
        Returns:
            question (str): generated question
        """
        # input is formatted like: "answer: <answer> context: <context>"
        text = "answer: {} context: {}".format(answer, context)
        input_ids = self.qg_tokenizer(text, return_tensors="pt").input_ids
        outputs = self.qg_model.generate(input_ids)
        question = self.qg_tokenizer.decode(outputs[0])
        return question.replace('question: ', '')

    def answer_question(self, question: str, context: str) -> str:
        """ Answer the question. p(A' | Q, C)
            
        Args: 
            question (str) : question to answer
            context (str) : document from which to generate question
        
        Returns:
            answer (str) : answer to question
        """
        # to answer questions, pass in question and context
        input_text = f"question: {question} context: {context}"
        input_ids = self.qa_tokenizer(input_text, return_tensors="pt").input_ids
        outputs = self.qa_model.generate(input_ids)
        return self.qa_tokenizer.decode(outputs[0])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate summaries and evaluate them using 'round-trip consistency' method."
    )
    parser.add_argument(
        "--n", type=int, default=20, help="number of examples to summarize"
    )
    parser.add_argument(
        "--shuffle",
        default=False,
        action="store_true",
        help="whether to shuffle training data",
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # TODO: integrate with sumtool dataset loader.
    dataset = load_dataset('xsum')['train']

    # Create round-trip consistency object, which uses three models!
    # TODO: explicitly put models on GPU?
    rt = RoundTripConsistency()

    # Generate questions from `args.n` datapoints.
    data = []
    for i in tqdm.trange(args.n, desc='Generating questions from summaries'):
        document = dataset[i]['document']
        summary = dataset[i]['summary']
        sum_answer = rt.extract_answer(summary)
        question = rt.generate_question(context=summary, answer=sum_answer)
        doc_answer = rt.answer_question(question=question, context=document)
        data.append([document, summary, question, sum_answer, doc_answer])

    # Create and display dataframe. 
    df = pd.DataFrame(data, columns=['document', 'summary', 'question', 'summary_answer', 'doc_answer'])
    print(df.head())

if __name__ == '__main__': main()