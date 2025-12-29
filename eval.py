from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-mpnet-base-v2")


def sentence_similarity(str1: str, str2: str) -> float:
    e1 = model.encode(str1)
    e2 = model.encode(str2)

    score = cosine_similarity([e1], [e2])[0][0]
    score_01 = (score + 1) / 2 # from [-1, 1] to [0, 1]
    return score_01

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