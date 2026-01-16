from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from bert_score import score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

model = SentenceTransformer("all-mpnet-base-v2")


def sbert_cosine_similarity(pred: str, ref: str) -> float:
    e1 = model.encode(pred)
    e2 = model.encode(ref)
    return cosine_similarity([e1], [e2])[0][0]  # range: [-1, 1]

def bertscore_f1(pred: str, ref: str) -> float:
    _, _, F1 = score(
        [pred],
        [ref],
        lang="en",
        model_type="roberta-large",
        verbose=False
    )
    return F1.item()  # range: [0, 1]

scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

def rouge_l_f1(pred: str, ref: str) -> float:
    scores = scorer.score(ref, pred)
    return scores["rougeL"].fmeasure  # range: [0, 1]

def bleu_score(pred: str, ref: str) -> float:
    smoothie = SmoothingFunction().method1
    return sentence_bleu(
        [ref.split()],
        pred.split(),
        smoothing_function=smoothie
    )  # range: [0, 1]

def multiple_choice_eval(answer_str: str, gt_str: str) -> int:
    """
    Return 1 if the first uppercased letter in answer_str matches the first
    uppercased letter in gt_str. Return 0 if answer_str has no uppercase letter
    or the letters do not match. Raise ValueError if gt_str contains no uppercase letter.
    """
    import re

    if gt_str is None:
        raise ValueError("ground truth string is None")

    m_gt = re.search(r"[A-Z]", gt_str)
    if not m_gt:
        raise ValueError("No uppercase letter found in ground truth string")

    gt_letter = m_gt.group(0)

    if answer_str is None:
        return 0

    m_ans = re.search(r"[A-Z]", answer_str)
    if not m_ans:
        return 0

    ans_letter = m_ans.group(0)
    return 1 if ans_letter == gt_letter else 0

def test_sentence_similarity():
    # str_question = """
    # Why the driver.Mem events are split into two groups? one at the beginning and one at the end?
    # """
    str_base = """
    They are two distinct phases of memory operations during the simulation: 
    initial memory setup (data transfer from the host to the GPU (MemCopyH2D) 
    before the kernel execution) and final memory operations (transfer results 
    back to the host (MemCopyD2H) and deallocate or clean up memory resources)."""
    str1 = """
    The driver.Mem events appear in two clusters because they correspond to two 
    stages of the simulation: memory initialization before kernel launch, where 
    data is copied from the host to the GPU, and memory finalization after 
    execution, where results are copied back and GPU memory is released."""
    str2 = """
    The memory-related driver events occur at different times because GPU memory 
    is managed both before and after kernel execution, involving data movement 
    and resource handling at multiple points in the program."""
    str3 = """
    The driver.Mem events are split because different GPU kernels are scheduled 
    on separate streams, allowing concurrent execution and better hardware 
    utilization."""
    str4 = """
    Williamsburg, a city in the U.S. state of Virginia, was capital of the Virginia 
    Colony from 1699 to 1780 and played a significant role in the American Revolution."""

    sim1 = sentence_similarity(str1, str_base)
    sim2 = sentence_similarity(str2, str_base)
    sim3 = sentence_similarity(str3, str_base)
    sim4 = sentence_similarity(str4, str_base)

    print(f"sim1: {sim1}\nsim2: {sim2}\nsim3: {sim3}\nsim4: {sim4}")

    assert sim1 > sim2 > sim3 > sim4


if __name__ == "__main__":
    test_sentence_similarity()