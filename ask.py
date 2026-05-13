"""
ask.py  --  interactive semantic entropy explorer
==================================================
Implements the exact pipeline from:
  Farquhar et al., "Detecting Hallucinations in Large Language Models
  Using Semantic Entropy", Nature 2024
  https://github.com/jlko/semantic_uncertainty

Pipeline (identical to paper):
  1. Sample N responses from an LLM with temperature=1.0
  2. Compute sequence log-likelihood (sum of token log-probs) for each response
  3. Cluster responses with bidirectional NLI entailment (DeBERTa-v2-xlarge-mnli)
  4. Aggregate cluster log-probabilities via logsumexp
  5. Compute Shannon entropy over cluster distribution = semantic entropy

Usage:
    python ask.py                       # manual mode (no LLM)
    python ask.py --openai              # OpenAI generates + DeBERTa clusters
    python ask.py --claude              # Claude generates + DeBERTa clusters
    python ask.py --openai-entailment   # OpenAI generates + GPT clusters
    python ask.py --claude-entailment   # Claude generates + Claude clusters
    python ask.py --fast                # use smaller NLI model (~180 MB)
    python ask.py --n 10                # number of samples (paper default: 10)

NLI models (controls cluster quality):
  Default:  microsoft/deberta-v2-xlarge-mnli     (~900 MB, paper's exact model)
  --fast:   cross-encoder/nli-deberta-v3-small   (~180 MB, faster)
  API:      --openai-entailment / --claude-entailment

Requirements:
    pip install scikit-learn numpy torch transformers
    pip install openai      (for --openai / --openai-entailment)
    pip install anthropic   (for --claude / --claude-entailment)
"""

import argparse
import math
import os
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# Entailment model 1: DeBERTa-v2-xlarge-mnli  (paper's exact model, ~900 MB)
#
# This is verbatim from EntailmentDeberta in the paper's codebase.
# Label order for microsoft/deberta-v2-xlarge-mnli (standard MNLI):
#   0 = contradiction,  1 = neutral,  2 = entailment
# No remapping needed -- matches the paper's convention.
# ---------------------------------------------------------------------------

class EntailmentDebertaLarge:

    MODEL = "microsoft/deberta-v2-xlarge-mnli"

    def __init__(self):
        os.environ["USE_TF"]    = "0"
        os.environ["USE_TORCH"] = "1"
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._F      = F
        self._torch  = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading {self.MODEL} on {self._device} ...")
        self._tok   = AutoTokenizer.from_pretrained(self.MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL).to(self._device)
        self._model.eval()
        print(f"  Model ready.\n")

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        inputs = self._tok(text1, text2, return_tensors="pt",
                           truncation=True, max_length=512).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        # Paper does: torch.argmax(F.softmax(logits, dim=1))
        pred = self._torch.argmax(self._F.softmax(logits, dim=1)).cpu().item()
        return int(pred)


# ---------------------------------------------------------------------------
# Entailment model 2: DeBERTa-v3-small  (faster alternative, ~180 MB)
#
# cross-encoder/nli-deberta-v3-small uses a different label order:
#   0 = contradiction,  1 = entailment,  2 = neutral
# We remap to the paper's convention before returning.
# ---------------------------------------------------------------------------

class EntailmentDebertaSmall:

    MODEL = "cross-encoder/nli-deberta-v3-small"
    REMAP = {0: 0, 1: 2, 2: 1}   # model order -> paper convention

    def __init__(self):
        os.environ["USE_TF"]    = "0"
        os.environ["USE_TORCH"] = "1"
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._F      = F
        self._torch  = torch
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  Loading {self.MODEL} on {self._device} ...")
        self._tok   = AutoTokenizer.from_pretrained(self.MODEL)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL).to(self._device)
        self._model.eval()
        print(f"  Model ready.\n")

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        inputs = self._tok(text1, text2, return_tensors="pt",
                           truncation=True, max_length=512).to(self._device)
        with self._torch.no_grad():
            logits = self._model(**inputs).logits
        raw = self._torch.argmax(self._F.softmax(logits, dim=1)).cpu().item()
        return self.REMAP[raw]


# ---------------------------------------------------------------------------
# Entailment model 3: OpenAI as NLI judge
# Matches EntailmentGPT4 in the paper exactly.
# ---------------------------------------------------------------------------

class EntailmentOpenAI:

    # Paper's exact prompt from EntailmentGPT4.equivalence_prompt
    PROMPT = (
        'We are evaluating answers to the question "{question}"\n'
        "Here are two possible answers:\n"
        "Possible Answer 1: {t1}\nPossible Answer 2: {t2}\n"
        "Does Possible Answer 1 semantically entail Possible Answer 2? "
        "Respond with entailment, contradiction, or neutral."
    )

    def __init__(self, model: str = "gpt-4o-mini"):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model  = model
        self._cache: dict[tuple, int] = {}

    def check_implication(self, text1: str, text2: str, question: str = "",
                          *args, **kwargs) -> int:
        key = (text1, text2, question)
        if key in self._cache:
            return self._cache[key]

        prompt = self.PROMPT.format(question=question, t1=text1, t2=text2)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )
        ans = resp.choices[0].message.content.strip().lower()
        result = 2 if "entailment" in ans else (0 if "contradiction" in ans else 1)
        self._cache[key] = result
        return result


# ---------------------------------------------------------------------------
# Entailment model 4: Claude as NLI judge
# Same prompt as EntailmentGPT4; same logic.
# ---------------------------------------------------------------------------

class EntailmentClaude:

    PROMPT = (
        'We are evaluating answers to the question "{question}"\n'
        "Here are two possible answers:\n"
        "Possible Answer 1: {t1}\nPossible Answer 2: {t2}\n"
        "Does Possible Answer 1 semantically entail Possible Answer 2? "
        "Respond with entailment, contradiction, or neutral."
    )

    def __init__(self, model: str = "claude-haiku-4-5"):
        import anthropic
        self._client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self._model  = model
        self._cache: dict[tuple, int] = {}

    def check_implication(self, text1: str, text2: str, question: str = "",
                          *args, **kwargs) -> int:
        key = (text1, text2, question)
        if key in self._cache:
            return self._cache[key]

        prompt = self.PROMPT.format(question=question, t1=text1, t2=text2)
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=10,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        ans = msg.content[0].text.strip().lower()
        result = 2 if "entailment" in ans else (0 if "contradiction" in ans else 1)
        self._cache[key] = result
        return result


# ---------------------------------------------------------------------------
# Entailment model 5: content-word heuristic  (offline fallback, no downloads)
# ---------------------------------------------------------------------------

class EntailmentContent:

    def _content(self, text: str) -> set:
        words = text.lower().replace("'s", "").split()
        cleaned = {w.strip('.,!?;:\'"()[]') for w in words}
        return {w for w in cleaned if w and w not in ENGLISH_STOP_WORDS}

    def check_implication(self, text1: str, text2: str, *args, **kwargs) -> int:
        c1, c2 = self._content(text1), self._content(text2)
        if not c1 or not c2:
            vec = TfidfVectorizer().fit_transform([text1, text2])
            sim = float(cosine_similarity(vec[0], vec[1])[0, 0])
            return 2 if sim >= 0.45 else (1 if sim >= 0.15 else 0)
        if c1.issubset(c2) or c2.issubset(c1):
            return 2
        jaccard = len(c1 & c2) / len(c1 | c2)
        if jaccard >= 0.5:
            return 2
        if (c1 - c2) and (c2 - c1):
            return 0
        return 1


def load_entailment_model(use_openai: bool = False, openai_model: str = "gpt-4o-mini",
                          use_claude: bool = False, claude_model: str = "claude-haiku-4-5",
                          use_fast: bool = False, local_path: str = None):
    if use_claude:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("  ANTHROPIC_API_KEY not set; falling back to DeBERTa.")
        else:
            return EntailmentClaude(model=claude_model), f"Claude {claude_model} (NLI judge)"

    if use_openai:
        if not os.environ.get("OPENAI_API_KEY"):
            print("  OPENAI_API_KEY not set; falling back to DeBERTa.")
        else:
            return EntailmentOpenAI(model=openai_model), f"OpenAI {openai_model} (NLI judge)"

    if local_path:
        EntailmentDebertaLarge.MODEL = local_path

    if use_fast:
        try:
            m = EntailmentDebertaSmall()
            return m, EntailmentDebertaSmall.MODEL + " (fast)"
        except Exception as e:
            print(f"  Could not load small DeBERTa: {e}")
            print("  Falling back to content-word heuristic.\n")
            return EntailmentContent(), "content-word heuristic"

    # Default: paper's exact model
    try:
        m = EntailmentDebertaLarge()
        return m, EntailmentDebertaLarge.MODEL + " (paper model)"
    except Exception as e:
        print(f"  Could not load large DeBERTa: {e}")
        print("  Trying smaller model ...")
        try:
            m = EntailmentDebertaSmall()
            return m, EntailmentDebertaSmall.MODEL + " (fast fallback)"
        except Exception as e2:
            print(f"  Could not load small DeBERTa either: {e2}")
            print("  Falling back to content-word heuristic.\n")
            return EntailmentContent(), "content-word heuristic"


# ---------------------------------------------------------------------------
# Core functions -- verbatim from semantic_entropy.py in the paper's repo
# ---------------------------------------------------------------------------

def get_semantic_ids(strings_list, model, strict_entailment=False, question=""):
    """Group list of predictions into semantic meaning.

    Identical to get_semantic_ids() in the paper's semantic_entropy.py.
    """
    def are_equivalent(text1, text2):
        implication_1 = model.check_implication(text1, text2, question=question)
        implication_2 = model.check_implication(text2, text1, question=question)
        assert implication_1 in [0, 1, 2] and implication_2 in [0, 1, 2]
        if strict_entailment:
            return implication_1 == 2 and implication_2 == 2
        implications = [implication_1, implication_2]
        return (0 not in implications) and ([1, 1] != implications)

    semantic_set_ids = [-1] * len(strings_list)
    next_id = 0
    for i, string1 in enumerate(strings_list):
        if semantic_set_ids[i] == -1:
            semantic_set_ids[i] = next_id
            for j in range(i + 1, len(strings_list)):
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    return semantic_set_ids


def logsumexp_by_id(semantic_ids, log_likelihoods):
    """Sum probabilities with the same semantic id.

    Identical to logsumexp_by_id(agg='sum_normalized') in the paper's repo.
    Input log_likelihoods are sequence-level (sum of token log-probs).
    """
    log_likelihoods = np.array(log_likelihoods, dtype=float)
    unique_ids = sorted(set(semantic_ids))
    log_likelihood_per_semantic_id = []

    for uid in unique_ids:
        id_indices = [pos for pos, x in enumerate(semantic_ids) if x == uid]
        id_log_likelihoods = log_likelihoods[id_indices]
        # normalize by total probability mass across all responses
        log_lik_norm = id_log_likelihoods - np.log(np.sum(np.exp(log_likelihoods)))
        logsumexp_value = np.log(np.sum(np.exp(log_lik_norm)))
        log_likelihood_per_semantic_id.append(float(logsumexp_value))

    return log_likelihood_per_semantic_id


def semantic_entropy(log_likelihoods_per_cluster):
    """Shannon entropy over normalised cluster probabilities.

    Identical to the paper's semantic_entropy computation.
    """
    probs = np.exp(log_likelihoods_per_cluster)
    probs = probs / probs.sum()
    return float(-(probs * np.log(probs + 1e-12)).sum())


def cluster_assignment_entropy(semantic_ids):
    """Entropy from cluster frequencies alone (no log-likelihoods).

    Identical to cluster_assignment_entropy() in the paper's repo.
    """
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts / n_generations
    return float(-(probabilities * np.log(probabilities + 1e-12)).sum())


def predictive_entropy(log_probs):
    """Naive token-level entropy: -1/N * sum(log p(x_i)).

    Identical to predictive_entropy() in the paper's repo.
    Does NOT distinguish paraphrases from truly different answers.
    """
    return float(-np.sum(log_probs) / len(log_probs))


# ---------------------------------------------------------------------------
# LLM samplers
# ---------------------------------------------------------------------------

def sample_responses_openai(question: str, n: int,
                             model: str = "gpt-4o-mini") -> tuple[list[tuple[str, float]], bool]:
    """Return (responses_with_logprob, has_logprobs).

    Log-probability = sum of token log-probs (sequence log-likelihood),
    matching the paper's treatment of per-response probability.
    """
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("openai package not installed. Run: pip install openai")

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set. Export it first:\n"
                 "  $env:OPENAI_API_KEY = 'sk-...'")

    client = OpenAI(api_key=api_key)
    print(f"\nSampling {n} responses from {model} (temperature=1.0) ...")
    results = []

    for i in range(n):
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "Answer the question concisely in 1-2 sentences."},
                {"role": "user", "content": question},
            ],
            temperature=1.0,
            logprobs=True,
            top_logprobs=1,
            max_tokens=120,
        )
        choice = resp.choices[0]
        text   = choice.message.content.strip()

        # Sequence log-likelihood = sum of token log-probs (paper convention)
        token_lps = [t.logprob for t in choice.logprobs.content
                     if t.logprob is not None]
        seq_logprob = float(np.sum(token_lps)) if token_lps else -1.0

        results.append((text, seq_logprob))
        print(f"  [{i+1}/{n}] {text[:80]}")

    return results, True   # has real log-probs


def sample_responses_claude(question: str, n: int,
                             model: str = "claude-haiku-4-5") -> tuple[list[tuple[str, float]], bool]:
    """Sample n responses from Claude. Returns (responses, has_logprobs=False).

    Claude does not expose token log-probabilities. Uniform weights are used,
    so semantic_entropy == cluster_assignment_entropy for Claude outputs.
    """
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set.\n"
                 "Get a key at console.anthropic.com, then:\n"
                 "  $env:ANTHROPIC_API_KEY = 'sk-ant-...'")

    client = anthropic.Anthropic(api_key=api_key)
    print(f"\nSampling {n} responses from {model} (temperature=1.0) ...")
    print("  Note: Claude does not expose log-probs; cluster_assignment_entropy"
          " is used.\n")
    results = []

    for i in range(n):
        msg = client.messages.create(
            model=model,
            max_tokens=120,
            temperature=1.0,
            system="Answer the question concisely in 1-2 sentences.",
            messages=[{"role": "user", "content": question}],
        )
        text = msg.content[0].text.strip()
        results.append((text, -1.0))
        print(f"  [{i+1}/{n}] {text[:80]}")

    return results, False   # no real log-probs


def collect_responses_manually() -> tuple[list[tuple[str, float]], bool]:
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
    return [(r, -1.0) for r in responses], False   # no real log-probs


# ---------------------------------------------------------------------------
# Analysis + display
# ---------------------------------------------------------------------------

W = 72

def sep(c="-"):  print(c * W)
def h(t, c="="): sep(c); print(f"  {t}"); sep(c)


def analyse(question: str, responses_with_lp: list[tuple[str, float]],
            has_logprobs: bool, model=None):
    responses  = [r  for r, _  in responses_with_lp]
    log_probs  = [lp for _,  lp in responses_with_lp]
    if model is None:
        model = EntailmentContent()

    print()
    h(f"Question: {question}")

    print("\nResponses sampled:")
    for i, (r, lp) in enumerate(responses_with_lp):
        lp_str = f"logP={lp:+.3f}" if has_logprobs else "logP=uniform"
        print(f"  [{i}] {lp_str}  {r}")

    # Similarity matrix (always content-word Jaccard; independent of NLI model)
    print("\nPairwise content-word Jaccard similarity:")
    n = len(responses)
    _heuristic = EntailmentContent()
    contents   = [_heuristic._content(r) for r in responses]

    def jaccard(a, b):
        return len(a & b) / len(a | b) if a and b else 0.0

    header = "         " + "".join(f"[{i}]   " for i in range(n))
    print("  " + header)
    for i in range(n):
        row = f"  [{i}]    " + "  ".join(
            f"{jaccard(contents[i], contents[j]):.2f}" for j in range(n))
        print(row)

    # Clustering (paper step 2)
    print("\nClustering into semantic groups (bidirectional NLI) ...")
    sem_ids = get_semantic_ids(responses, model, question=question)

    clusters: dict[int, list] = {}
    for i, (r, sid) in enumerate(zip(responses, sem_ids)):
        clusters.setdefault(sid, []).append((i, r))

    print(f"\nCluster assignments : {sem_ids}")
    print(f"Distinct clusters   : {len(clusters)}\n")
    for cid, members in sorted(clusters.items()):
        print(f"  Cluster {cid}:")
        for idx, r in members:
            print(f"    [{idx}] {r}")

    # Entropy measures (paper step 3-5)
    ca_ent   = cluster_assignment_entropy(sem_ids)
    ll_per_c = logsumexp_by_id(sem_ids, log_probs)
    se       = semantic_entropy(ll_per_c)
    pe       = predictive_entropy(log_probs)

    cluster_probs = np.exp(ll_per_c)
    cluster_probs = cluster_probs / cluster_probs.sum()

    print("\nPer-cluster probability mass:")
    for cid, prob in enumerate(cluster_probs):
        bar   = "#" * max(1, int(prob * 30))
        first = clusters[cid][0][1][:55]
        print(f"  Cluster {cid} ({prob*100:5.1f}%)  {bar}  \"{first}\"")

    print()
    sep()
    print(f"  Cluster-assignment entropy  (freq only, no log-lik) : {ca_ent:.4f}")
    if has_logprobs:
        print(f"  Semantic entropy  (paper method, weighted by logP)  : {se:.4f}")
        print(f"  Naive predictive entropy  (token-level baseline)    : {pe:.4f}")
    else:
        print(f"  Semantic entropy  (= cluster-assignment; no logP)   : {se:.4f}")
        print(f"  Note: log-probs unavailable -- all responses weighted equally.")
    sep()

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Semantic entropy explorer (Farquhar et al., Nature 2024)")
    parser.add_argument("--openai",  action="store_true",
                        help="Use OpenAI to generate responses")
    parser.add_argument("--claude",  action="store_true",
                        help="Use Claude to generate responses")
    parser.add_argument("--model",   default="gpt-4o-mini",
                        help="OpenAI model for generation (default: gpt-4o-mini)")
    parser.add_argument("--claude-model", default="claude-haiku-4-5",
                        help="Claude model for generation (default: claude-haiku-4-5)")
    parser.add_argument("--n",       type=int, default=10,
                        help="Number of responses to sample (paper default: 10)")
    parser.add_argument("--fast",    action="store_true",
                        help="Use smaller NLI model (~180 MB) instead of paper's (~900 MB)")
    parser.add_argument("--entailment-model", default=None,
                        help="Local path to a custom NLI model")
    parser.add_argument("--openai-entailment", action="store_true",
                        help="Use OpenAI GPT as the NLI judge")
    parser.add_argument("--claude-entailment", action="store_true",
                        help="Use Claude as the NLI judge")
    args = parser.parse_args()

    print("=" * W)
    print("  Semantic Entropy Explorer")
    print("  Farquhar et al., Nature 2024  (github.com/jlko/semantic_uncertainty)")
    print("=" * W)

    if args.claude:
        print(f"\nMode: Claude ({args.claude_model}), {args.n} samples per question")
    elif args.openai:
        print(f"\nMode: OpenAI ({args.model}), {args.n} samples per question")
    else:
        print("\nMode: manual (you provide the responses)")
        print("Tip:  run with --claude or --openai to auto-generate responses")

    print("\nLoading entailment model...")
    entailment_model, model_name = load_entailment_model(
        use_claude=args.claude_entailment,
        claude_model=args.claude_model,
        use_openai=args.openai_entailment,
        openai_model=args.model,
        use_fast=args.fast,
        local_path=args.entailment_model,
    )
    print(f"Entailment: {model_name}\n")

    while True:
        print()
        try:
            question = input("Enter a question (or 'quit' to exit):\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in ("quit", "exit", "q", ""):
            break

        if args.claude:
            try:
                pairs, has_lp = sample_responses_claude(
                    question, args.n, args.claude_model)
            except Exception as e:
                print(f"\nClaude error: {e}")
                print("Falling back to manual mode.\n")
                pairs, has_lp = collect_responses_manually()
        elif args.openai:
            try:
                pairs, has_lp = sample_responses_openai(
                    question, args.n, args.model)
            except Exception as e:
                print(f"\nOpenAI error: {e}")
                print("Falling back to manual mode.\n")
                pairs, has_lp = collect_responses_manually()
        else:
            pairs, has_lp = collect_responses_manually()

        analyse(question, pairs, has_lp, model=entailment_model)

        try:
            again = input("Try another question? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if again in ("n", "no"):
            break

    print("\nDone.")


if __name__ == "__main__":
    main()
