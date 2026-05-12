"""
Semantic Entropy Demo  (fully offline, no model downloads)
===========================================================
Demonstrates the core algorithm from:
  "Detecting Hallucinations in Large Language Models Using Semantic Entropy"
  Farquhar et al., Nature 2024  |  github.com/jlko/semantic_uncertainty

Pipeline
--------
1. An LLM is sampled N times for the same question.
2. Responses are *clustered* into semantic groups via NLI entailment.
3. Entropy is computed over those clusters.
   High entropy  -> model is uncertain / likely hallucinating.
   Low entropy   -> model is confident and consistent.

Offline entailment
------------------
The production code uses deberta-v2-xlarge-mnli (~900 MB).
Here we substitute a TF-IDF cosine-similarity check so the demo
runs instantly without downloading anything.  The clustering logic
(get_semantic_ids) and every entropy formula are identical to the paper.
"""

import math
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------------------------------------------------
# Offline entailment: TF-IDF cosine similarity
# -----------------------------------------------------------------------------

ENTAIL_THRESHOLD  = 0.45   # cosine >= this -> "entailment"
NEUTRAL_THRESHOLD = 0.15   # cosine >= this -> "neutral"; below -> "contradiction"

class EntailmentTFIDF:
    """
    NLI approximation via TF-IDF cosine similarity.

    Returns the same integer codes the real NLI model returns:
      0 = contradiction  (very different)
      1 = neutral        (related but not equivalent)
      2 = entailment     (one follows from the other)

    check_implication(text1, text2) checks "text1 -> text2"
    (does text2 follow from text1?).
    """

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        vec = TfidfVectorizer().fit_transform([text1, text2])
        sim = float(cosine_similarity(vec[0], vec[1])[0, 0])
        if sim >= ENTAIL_THRESHOLD:
            return 2   # entailment
        elif sim >= NEUTRAL_THRESHOLD:
            return 1   # neutral
        else:
            return 0   # contradiction


# -----------------------------------------------------------------------------
# Core functions (verbatim from semantic_entropy.py in the repo)
# -----------------------------------------------------------------------------

def get_semantic_ids(strings_list, model, strict_entailment=False, example=None):
    """Group responses into semantic clusters via bidirectional entailment."""

    def are_equivalent(text1, text2):
        imp1 = model.check_implication(text1, text2, example=example)
        imp2 = model.check_implication(text2, text1, example=example)
        assert imp1 in (0, 1, 2) and imp2 in (0, 1, 2)
        if strict_entailment:
            return imp1 == 2 and imp2 == 2
        # Equivalent if no contradiction and not both neutral
        return (0 not in [imp1, imp2]) and ([1, 1] != [imp1, imp2])

    semantic_ids = [-1] * len(strings_list)
    next_id = 0
    for i, s1 in enumerate(strings_list):
        if semantic_ids[i] == -1:
            semantic_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(s1, strings_list[j]):
                    semantic_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_ids
    return semantic_ids


def logsumexp_by_id(semantic_ids, log_likelihoods):
    """
    Aggregate per-token log-likelihoods into per-cluster log-probabilities.

    For cluster c:
        log p(c) = log sum_{i in c} p(x_i) / sum_all p(x_j)
                 = LogSumExp(ll_i for i in c)  - log sum_all exp(ll_j)
    """
    unique_ids = sorted(set(semantic_ids))
    log_Z = math.log(sum(math.exp(ll) for ll in log_likelihoods))
    result = []
    for uid in unique_ids:
        indices = [i for i, x in enumerate(semantic_ids) if x == uid]
        cluster_lls = [log_likelihoods[i] for i in indices]
        normed = [ll - log_Z for ll in cluster_lls]
        logsumexp = math.log(sum(math.exp(v) for v in normed))
        result.append(logsumexp)
    return result


def semantic_entropy(log_likelihoods_per_cluster):
    """Shannon entropy over normalised cluster probabilities."""
    probs = np.exp(log_likelihoods_per_cluster)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def cluster_assignment_entropy(semantic_ids):
    """
    Entropy from cluster *frequencies* alone (no log-likelihoods needed).

    Treats each response as equally likely, then asks:
    how spread out are the cluster assignments?
    """
    counts = np.bincount(semantic_ids)
    probs  = counts / len(semantic_ids)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def predictive_entropy(log_probs):
    """
    Naive token-level entropy  =  -1/N  sum log p(x_i).

    Does NOT distinguish paraphrases from truly different answers.
    """
    return float(-np.mean(log_probs))


# -----------------------------------------------------------------------------
# Three example questions with different uncertainty levels
# -----------------------------------------------------------------------------

SCENARIOS = [
    {
        "label": "LOW UNCERTAINTY  (all answers mean the same thing)",
        "question": "What is the capital of France?",
        "responses": [
            "The capital of France is Paris.",
            "Paris is the capital city of France.",
            "France's capital is Paris.",
            "Paris serves as the capital of France.",
            "The capital is Paris.",
        ],
        "log_likelihoods": [-0.50, -0.60, -0.55, -0.65, -0.40],
    },
    {
        "label": "MEDIUM UNCERTAINTY  (two or three distinct answers)",
        "question": "Who invented the telephone?",
        "responses": [
            "Alexander Graham Bell invented the telephone.",
            "The telephone was invented by Alexander Graham Bell.",
            "Bell is widely credited with inventing the telephone.",
            "Elisha Gray also filed a patent on the same day as Bell.",
            "Antonio Meucci is credited by some as the true inventor.",
        ],
        "log_likelihoods": [-0.80, -0.90, -0.85, -2.10, -2.50],
    },
    {
        "label": "HIGH UNCERTAINTY  (every answer is semantically different)",
        "question": "What is the best programming language?",
        "responses": [
            "Python is the best language due to its simplicity.",
            "Rust is the best language because of its memory safety guarantees.",
            "JavaScript is the best because it runs in every browser.",
            "C++ is the best for high-performance system programming.",
            "There is no single best language; it depends entirely on context.",
        ],
        "log_likelihoods": [-1.20, -1.50, -1.30, -1.80, -1.60],
    },
]


# -----------------------------------------------------------------------------
# Pretty-print helpers
# -----------------------------------------------------------------------------

W = 72

def sep(char="-"): print(char * W)

def banner(text, char="="):
    sep(char)
    print(f"  {text}")
    sep(char)


# -----------------------------------------------------------------------------
# Main demo
# -----------------------------------------------------------------------------

def run_demo():
    model = EntailmentTFIDF()
    print(f"\nEntailment model : TF-IDF cosine similarity (offline)")
    print(f"  threshold entailment >= {ENTAIL_THRESHOLD}  |  neutral >= {NEUTRAL_THRESHOLD}\n")

    for scenario in SCENARIOS:
        banner(scenario["label"])
        print(f"  Question: {scenario['question']}\n")

        responses     = scenario["responses"]
        log_liks      = scenario["log_likelihoods"]

        # -- 1. Show sampled responses ----------------------------------------
        print("  Sampled LLM responses (with simulated log-likelihoods):")
        for i, (r, ll) in enumerate(zip(responses, log_liks)):
            print(f"    [{i}] logP={ll:+.2f}  {r}")

        # -- 2. Pairwise similarity matrix ------------------------------------
        print("\n  Pairwise cosine similarity matrix:")
        vecs = TfidfVectorizer().fit_transform(responses)
        sim_mat = cosine_similarity(vecs)
        header = "        " + "".join(f"[{i}]   " for i in range(len(responses)))
        print("  " + header)
        for i in range(len(responses)):
            row = f"  [{i}]   " + "  ".join(f"{sim_mat[i,j]:.2f}" for j in range(len(responses)))
            print(row)

        # -- 3. Cluster via get_semantic_ids ----------------------------------
        print("\n  Clustering responses into semantic groups...")
        semantic_ids = get_semantic_ids(responses, model)

        clusters: dict[int, list[tuple[int, str]]] = {}
        for i, (r, sid) in enumerate(zip(responses, semantic_ids)):
            clusters.setdefault(sid, []).append((i, r))

        print(f"\n  Semantic cluster assignments: {semantic_ids}")
        print(f"  Number of distinct clusters : {len(clusters)}\n")
        for cid, members in sorted(clusters.items()):
            print(f"  Cluster {cid}:")
            for idx, r in members:
                print(f"    [{idx}] {r}")

        # -- 4. Compute entropy measures --------------------------------------
        ca_ent  = cluster_assignment_entropy(semantic_ids)
        ll_per_c = logsumexp_by_id(semantic_ids, log_liks)
        se       = semantic_entropy(ll_per_c)
        pe       = predictive_entropy(log_liks)

        # per-cluster probability breakdown
        cluster_probs = np.exp(ll_per_c)
        cluster_probs = cluster_probs / cluster_probs.sum()

        print(f"\n  Per-cluster probability mass (from log-likelihoods):")
        for cid, prob in enumerate(cluster_probs):
            first = clusters[cid][0][1][:52]
            bar   = "#" * int(prob * 30)
            print(f"    Cluster {cid} ({prob*100:5.1f}%)  {bar}  \"{first}...\"")

        print(f"\n  +{'-'*45}+")
        print(f"  |  Entropy measure                  value  |")
        print(f"  +{'-'*45}+")
        print(f"  |  Cluster-assignment entropy       {ca_ent:.4f} |  (no log-lik)")
        print(f"  |  Semantic entropy  (paper method) {se:.4f} |  (weighted by log-lik)")
        print(f"  |  Naive predictive entropy         {pe:.4f} |  (token-level baseline)")
        print(f"  +{'-'*45}+\n")

    # -- Interpretation guide -------------------------------------------------
    banner("INTERPRETATION GUIDE", "=")
    lines = [
        "Semantic entropy groups responses by MEANING before computing entropy.",
        "",
        "  Entropy ~= 0   All samples cluster into ONE group   -> confident model",
        "  Entropy high   Samples spread across MANY clusters  -> uncertain / hallucinating",
        "",
        "Why semantic entropy beats naive token entropy:",
        "  Naive entropy is high if responses use different words even for the",
        "  same meaning (paraphrases).  Semantic entropy collapses paraphrases",
        "  into one cluster first, so it only rises when meanings truly differ.",
        "",
        "  SE > naive  -> distinct meanings hidden behind similar-sounding tokens",
        "  SE < naive  -> paraphrases inflate token entropy artificially",
        "",
        "Cluster-assignment entropy (no log-lik) is a fast approximation that",
        "treats every sample equally.  Semantic entropy weights by model confidence.",
    ]
    for l in lines:
        print(f"  {l}")
    sep()
    print()


if __name__ == "__main__":
    run_demo()
