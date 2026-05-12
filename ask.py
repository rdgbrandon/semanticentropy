"""
ask.py  --  interactive semantic entropy explorer
==================================================
Usage:
    python ask.py                          # manual mode: you paste responses
    python ask.py --openai                 # auto-generate responses with GPT-4o-mini
    python ask.py --openai --n 10          # generate 10 responses instead of 5

In auto mode the script samples the same question N times at temperature=1
to get a spread of responses, then runs the semantic clustering + entropy
analysis exactly as described in the paper.

Requirements:
    pip install openai scikit-learn numpy
    export OPENAI_API_KEY=sk-...   (only needed for --openai mode)
"""

import argparse
import math
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------------------------------
# Entailment model  (content-word heuristic, fully offline)
#
# Pure TF-IDF fails on short answers: "the color is yellow" vs
# "the color is black" scores 0.81 because structure words dominate.
# This version compares *content words* (non-stopwords) instead.
#
# Rules (applied in order):
#   1. If one response's content words are a SUBSET of the other's -> entailment
#      ("yellow" is a subset of "yellow fur" -> same answer, more specific)
#   2. If content-word Jaccard >= 0.5 -> entailment
#      ("invented telephone Bell" ~ "Bell invented telephone")
#   3. If BOTH sides have unique content words not in the other -> contradiction
#      ("yellow" unique to left, "black" unique to right -> different answers)
#   4. Otherwise -> neutral
# --------------------------------------------------------------------------

class EntailmentContent:

    def _content(self, text: str) -> set:
        words = text.lower().replace("'s", "").split()
        cleaned = {w.strip('.,!?;:\'"()[]') for w in words}
        return {w for w in cleaned if w and w not in ENGLISH_STOP_WORDS}

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        c1, c2 = self._content(text1), self._content(text2)

        # Fall back to TF-IDF if one side is all stopwords (e.g. "it is")
        if not c1 or not c2:
            vec = TfidfVectorizer().fit_transform([text1, text2])
            sim = float(cosine_similarity(vec[0], vec[1])[0, 0])
            return 2 if sim >= 0.45 else (1 if sim >= 0.15 else 0)

        # Rule 1: subset -> entailment ("yellow" c "yellow fur")
        if c1.issubset(c2) or c2.issubset(c1):
            return 2

        # Rule 2: high Jaccard -> entailment
        jaccard = len(c1 & c2) / len(c1 | c2)
        if jaccard >= 0.5:
            return 2

        # Rule 3: both sides carry distinct content -> contradiction
        if (c1 - c2) and (c2 - c1):
            return 0

        return 1

# --------------------------------------------------------------------------
# Core functions from semantic_entropy.py (unchanged)
# --------------------------------------------------------------------------

def get_semantic_ids(strings_list, model, strict_entailment=False):
    def are_equivalent(t1, t2):
        i1 = model.check_implication(t1, t2)
        i2 = model.check_implication(t2, t1)
        if strict_entailment:
            return i1 == 2 and i2 == 2
        return (0 not in [i1, i2]) and ([1, 1] != [i1, i2])

    ids = [-1] * len(strings_list)
    nid = 0
    for i, s1 in enumerate(strings_list):
        if ids[i] == -1:
            ids[i] = nid
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(s1, strings_list[j]):
                    ids[j] = nid
            nid += 1
    return ids


def logsumexp_by_id(semantic_ids, log_likelihoods):
    unique_ids = sorted(set(semantic_ids))
    log_Z = math.log(sum(math.exp(ll) for ll in log_likelihoods))
    result = []
    for uid in unique_ids:
        indices = [i for i, x in enumerate(semantic_ids) if x == uid]
        normed = [log_likelihoods[i] - log_Z for i in indices]
        result.append(math.log(sum(math.exp(v) for v in normed)))
    return result


def semantic_entropy(log_probs_per_cluster):
    probs = np.exp(log_probs_per_cluster)
    probs /= probs.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def cluster_assignment_entropy(semantic_ids):
    counts = np.bincount(semantic_ids)
    probs  = counts / len(semantic_ids)
    return float(-(probs * np.log(probs + 1e-12)).sum())


def predictive_entropy(log_probs):
    return float(-np.mean(log_probs))

# --------------------------------------------------------------------------
# OpenAI sampler
# --------------------------------------------------------------------------

def sample_responses_openai(question: str, n: int, model: str = "gpt-4o-mini") -> list[tuple[str, float]]:
    """Return list of (response_text, log_prob) tuples."""
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set. Export it first:\n  $env:OPENAI_API_KEY = 'sk-...'")

    client = OpenAI(api_key=api_key)

    print(f"\nSampling {n} responses from {model} (temperature=1.0) ...")
    results = []
    for i in range(n):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Answer the question concisely in 1-2 sentences."},
                {"role": "user",   "content": question},
            ],
            temperature=1.0,
            logprobs=True,
            top_logprobs=1,
            max_tokens=120,
        )
        choice  = resp.choices[0]
        text    = choice.message.content.strip()

        # Average token log-probability as a proxy for sequence log-likelihood
        token_logprobs = [t.logprob for t in choice.logprobs.content if t.logprob is not None]
        avg_logprob = float(np.mean(token_logprobs)) if token_logprobs else -1.0

        results.append((text, avg_logprob))
        print(f"  [{i+1}/{n}] {text[:80]}")

    return results


# --------------------------------------------------------------------------
# Manual input mode
# --------------------------------------------------------------------------

def collect_responses_manually() -> list[tuple[str, float]]:
    print("\nEnter each response on its own line.")
    print("Press Enter twice (blank line) when done. Minimum 2 responses.\n")
    responses = []
    idx = 0
    while True:
        line = input(f"  Response {idx+1}: ").strip()
        if line == "":
            if len(responses) >= 2:
                break
            print("  (need at least 2 responses)")
            continue
        responses.append(line)
        idx += 1

    # No real log-likelihoods in manual mode: use uniform
    log_probs = [-1.0] * len(responses)
    return list(zip(responses, log_probs))

# --------------------------------------------------------------------------
# Analysis + display
# --------------------------------------------------------------------------

W = 72

def sep(c="-"):  print(c * W)
def h(t, c="="): sep(c); print(f"  {t}"); sep(c)


def analyse(question: str, responses_with_lp: list[tuple[str, float]], source: str):
    responses   = [r for r, _ in responses_with_lp]
    log_probs   = [lp for _, lp in responses_with_lp]
    model       = EntailmentContent()

    print()
    h(f"Question: {question}")

    print("\nResponses sampled:")
    for i, (r, lp) in enumerate(responses_with_lp):
        lp_str = f"logP={lp:+.3f}" if source == "openai" else "logP=uniform"
        print(f"  [{i}] {lp_str}  {r}")

    # Similarity matrix  (content-word Jaccard -- matches what clustering uses)
    print("\nPairwise content-word Jaccard similarity:")
    n = len(responses)
    contents = [model._content(r) for r in responses]

    def jaccard(a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    header = "         " + "".join(f"[{i}]   " for i in range(n))
    print("  " + header)
    for i in range(n):
        row = f"  [{i}]    " + "  ".join(f"{jaccard(contents[i], contents[j]):.2f}" for j in range(n))
        print(row)
    print(f"  (entailment >= 0.50 or subset | contradiction = both sides unique)")


    # Clustering
    print("\nClustering into semantic groups (bidirectional NLI) ...")
    sem_ids = get_semantic_ids(responses, model)

    clusters: dict[int, list] = {}
    for i, (r, sid) in enumerate(zip(responses, sem_ids)):
        clusters.setdefault(sid, []).append((i, r))

    print(f"\nCluster assignments : {sem_ids}")
    print(f"Distinct clusters   : {len(clusters)}\n")
    for cid, members in sorted(clusters.items()):
        print(f"  Cluster {cid}:")
        for idx, r in members:
            print(f"    [{idx}] {r}")

    # Entropy
    ca_ent  = cluster_assignment_entropy(sem_ids)
    ll_per_c = logsumexp_by_id(sem_ids, log_probs)
    se       = semantic_entropy(ll_per_c)
    pe       = predictive_entropy(log_probs)

    cluster_probs = np.exp(ll_per_c) / np.exp(ll_per_c).sum()

    print("\nPer-cluster probability mass:")
    for cid, prob in enumerate(cluster_probs):
        bar   = "#" * max(1, int(prob * 30))
        first = clusters[cid][0][1][:55]
        print(f"  Cluster {cid} ({prob*100:5.1f}%)  {bar}  \"{first}\"")

    print()
    sep()
    print(f"  Cluster-assignment entropy  (freq only) : {ca_ent:.4f}")
    print(f"  Semantic entropy  (paper method)        : {se:.4f}")
    if source == "openai":
        print(f"  Naive predictive entropy  (token-level) : {pe:.4f}")
    sep()

    # Verdict
    max_entropy = math.log(len(responses))
    frac = se / max_entropy if max_entropy > 0 else 0
    if frac < 0.25:
        verdict = "LOW uncertainty   -- model is confident and consistent"
    elif frac < 0.60:
        verdict = "MEDIUM uncertainty -- model gives somewhat different answers"
    else:
        verdict = "HIGH uncertainty  -- model is uncertain, possible hallucination risk"

    print(f"\n  Verdict: {verdict}")
    print(f"  (semantic entropy {se:.3f} out of max {max_entropy:.3f})")
    sep()
    print()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Interactive semantic entropy explorer")
    parser.add_argument("--openai",  action="store_true", help="Use OpenAI to generate responses")
    parser.add_argument("--model",   default="gpt-4o-mini", help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--n",       type=int, default=5,   help="Number of responses to sample (default: 5)")
    args = parser.parse_args()

    print("=" * W)
    print("  Semantic Entropy Explorer")
    print("  Based on: Farquhar et al., Nature 2024  (github.com/jlko/semantic_uncertainty)")
    print("=" * W)

    if args.openai:
        print(f"\nMode: OpenAI ({args.model}), sampling {args.n} responses per question")
    else:
        print("\nMode: manual (you provide the responses)")
        print("Tip:  run with --openai to auto-generate responses via GPT-4o-mini")

    while True:
        print()
        try:
            question = input("Enter a question (or 'quit' to exit):\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q", ""):
            break

        if args.openai:
            try:
                pairs = sample_responses_openai(question, args.n, args.model)
            except Exception as e:
                print(f"\nOpenAI error: {e}")
                print("Falling back to manual mode.\n")
                pairs = collect_responses_manually()
                source = "manual"
            else:
                source = "openai"
        else:
            pairs  = collect_responses_manually()
            source = "manual"

        analyse(question, pairs, source)

        again = input("Try another question? [Y/n]: ").strip().lower()
        if again in ("n", "no"):
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
