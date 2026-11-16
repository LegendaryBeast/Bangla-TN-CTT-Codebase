import unicodedata

GLUE = "\x02"

_BN = {}
for chars, v in [("কখগঘ", "k"), ("চছ", "k"), ("জঝয", "j"), ("টঠতথৎ", "t"), ("ডঢদধ", "d"),
                 ("ঙঞণনং", "n"), ("পফ", "p"), ("বভ", "b"), ("ম", "m"), ("র", "r"), ("ড়ঢ়", "r"), ("ল", "l"),
                 ("শষস", "s"), ("হঃ", ""), ("য়", ""), ("ঁ", ""), ("।", "."),
                 ("অআইঈউঊঋএঐওঔ", ""), ("ািীুূৃেৈোৌ্", "")]:
    for ch in chars:
        _BN[ch] = v
for i, ch in enumerate("০১২৩৪৫৬৭৮৯"):
    _BN[ch] = str(i)
_LAT = {**{c: "" for c in "aeiouwyh"}, **{c: "k" for c in "ckq"}, "x": "ks", "z": "j", "v": "b", "f": "p",
        "g": "k"}


def skeleton(s):
    out = []
    for ch in unicodedata.normalize("NFC", s.lower()):
        if ch in _BN:
            out.append(_BN[ch])
        elif "a" <= ch <= "z":
            out.append(_LAT.get(ch, ch))
        elif ch.isdigit() or not ch.isalnum():
            out.append(ch if not ch.isspace() else "")
        else:
            out.append(ch)
    s = "".join(out)
    return "".join(c for i, c in enumerate(s) if i == 0 or c != s[i - 1])


def _lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _is_symbolic(w):
    return not any(ch.isalpha() for ch in w)


class Aligner:

    def __init__(self, lexicon=None):
        self.lex = lexicon or set()
        self._sk = {}

    def _skel(self, s):
        if s not in self._sk:
            self._sk[s] = skeleton(s)
        return self._sk[s]

    def tok_cost(self, src, tgt):
        if not tgt:
            return 0.35 if _is_symbolic(src) else 1.0
        if (src.lower(), tgt) in self.lex or src == tgt:
            return 0.0
        a, b = self._skel(src), self._skel(tgt)
        if not a and not b:
            return 0.1
        return _lev(a, b) / max(len(a), len(b))

    def align(self, src, tgt):
        n, m = len(src), len(tgt)
        INF = float("inf")
        D = [[INF] * (m + 1) for _ in range(n + 1)]
        B = [[None] * (m + 1) for _ in range(n + 1)]
        D[0][0] = 0.0
        for i in range(n + 1):
            for j in range(m + 1):
                if D[i][j] == INF:
                    continue
                base = D[i][j]
                if i < n:
                    for b in range(0, 4):
                        if j + b > m:
                            break
                        span = " ".join(tgt[j:j + b])
                        c = base + self.tok_cost(src[i], span) + (0.0 if b == 1 else 0.25 * abs(b - 1))
                        if c < D[i + 1][j + b]:
                            D[i + 1][j + b], B[i + 1][j + b] = c, ("split", 1, b, None)
                if j < m:
                    w = tgt[j]
                    for a in (2, 3):
                        if i + a > n:
                            break
                        best, cut = self._best_cuts(src[i:i + a], w)
                        c = base + best + 0.25 * (a - 1)
                        if c < D[i + a][j + 1]:
                            D[i + a][j + 1], B[i + a][j + 1] = c, ("merge", a, 1, cut)
        if D[n][m] == INF:
            return None, INF
        spans, i, j = [], n, m
        while i or j:
            kind, a, b, cut = B[i][j]
            if kind == "split":
                spans.append(" ".join(tgt[j - b:j]))
            else:
                parts = [tgt[j - 1][s:e] for s, e in zip((0,) + cut, cut + (len(tgt[j - 1]),))]
                out, started = [], False
                for p in parts:
                    out.append("" if not p else (GLUE + p if started else p))
                    started = started or bool(p)
                spans += out[::-1]
            i, j = i - a, j - b
        return spans[::-1], D[n][m] / max(1, n)

    def _best_cuts(self, srcs, w):
        L = len(w)
        if len(srcs) == 2:
            cands = [(k,) for k in range(L + 1)]
        else:
            step = max(1, L // 12)
            cands = [(k1, k2) for k1 in range(0, L + 1, step) for k2 in range(k1, L + 1, step)]
        best, arg = float("inf"), None
        for cut in cands:
            parts = [w[s:e] for s, e in zip((0,) + cut, cut + (L,))]
            c = sum(self.tok_cost(s, p) for s, p in zip(srcs, parts))
            if c < best:
                best, arg = c, cut
        return best, arg


def join_spans(spans):
    out = ""
    for s in spans:
        if not s:
            continue
        if s.startswith(GLUE):
            out += s[len(GLUE):]
        else:
            out += (" " if out else "") + s
    return out


def selftest():
    al = Aligner()
    cases = [
        (["back", "."], ["ব্যাক।"]),
        (["Bhaiনাকি", ".."], ["ভাই", "নাকি", ".."]),
        (["friend", "ship✨✨"], ["ফ্রেন্ডশিপ✨✨"]),
        (["vaiya", "please", "give"], ["ভাইয়া", "প্লিজ", "গিভ"]),
        (["70theke", "75k"], ["৭০", "থেকে", "৭৫কে"]),
    ]
    for s, t in cases:
        spans, c = al.align(s, t)
        assert join_spans(spans) == " ".join(t), (s, t, spans)
        print(f"{c:.2f}", list(zip(s, [x.replace(GLUE, '+') for x in spans])))


if __name__ == "__main__":
    selftest()
