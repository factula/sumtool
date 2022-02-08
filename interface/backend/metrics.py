from bert_score import score
from rouge_score import rouge_scorer

rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


def compute_summary_metrics(true_summary, pred_summary):
    rogue_scores = rouge_scorer.score(
        true_summary, 
        pred_summary,
    )

    (bert_P, bert_R, bert_F) = score(
        [true_summary], 
        [pred_summary],
        lang="en", 
        return_hash=False,
        model_type="distilbert-base-uncased"
    )
    return bert_F.mean().item(), 