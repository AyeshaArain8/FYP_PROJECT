"""
Very simple gloss -> English cleanup layer.

PSL, like many sign languages, often drops articles and copulas and can use
different word order than English. This is a rule-based placeholder -- good
enough for short phrases from a small vocabulary, and deliberately kept as
its own module so it can later be swapped for a trained seq2seq model
without touching the rest of the pipeline.
"""

# Nouns that read more naturally in English with "the" in front of them.
ARTICLE_INSERT_BEFORE = {"doctor", "atm", "bicycle", "bird", "cow"}


def gloss_to_english(glosses: list) -> str:
    if not glosses:
        return ""

    words = []
    for i, g in enumerate(glosses):
        w = g.replace("_", " ").replace("-", " ")
        if w in ARTICLE_INSERT_BEFORE and (i == 0 or glosses[i - 1].lower() not in ("the", "a")):
            words.append("the")
        words.append(w)

    sentence = " ".join(words)
    sentence = sentence[0].upper() + sentence[1:]
    if not sentence.endswith((".", "!", "?")):
        sentence += "."
    return sentence


if __name__ == "__main__":
    import sys
    print(gloss_to_english(sys.argv[1:]))
