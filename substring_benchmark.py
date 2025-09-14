"""
МІНІМАЛЬНИЙ бенчмарк (stdlib-only): Boyer–Moore, KMP, Rabin–Karp.
- За замовчуванням: друкує таблицю в консоль і створює REPORT.md
- Хочеш без файлу — додай прапорець: --no-report
"""
import argparse
import statistics as stats
import timeit
from pathlib import Path


# ---- robust read ----
def read_text_robust(p: Path) -> str:
    encs = ["utf-8", "utf-8-sig", "cp1251", "koi8_u", "iso8859_5", "cp866", "utf-16", "utf-16le", "utf-16be", "latin-1"]
    for enc in encs:
        try:
            with open(p, "r", encoding=enc) as f:
                t = f.read()
            if sum(ch.isprintable() or ch in "\n\t\r" for ch in t) / max(1, len(t)) > 0.9:
                return t
        except Exception:
            pass
    with open(p, "rb") as f:
        raw = f.read()
    return raw.decode("utf-8", errors="ignore")


# ---- algorithms ----
def kmp_search(text: str, pattern: str) -> int:
    if not pattern:
        return 0
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        elif length != 0:
            length = lps[length - 1]
        else:
            lps[i] = 0
            i += 1
    i = j = 0
    n, m = len(text), len(pattern)
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == m:
                return i - j
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return -1


def rabin_karp_search(text: str, pattern: str) -> int:
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1
    base = 256
    mod = 1_000_000_007
    high = pow(base, m - 1, mod)
    th = 0
    ph = 0
    for i in range(m):
        th = (th * base + ord(text[i])) % mod
        ph = (ph * base + ord(pattern[i])) % mod
    for i in range(n - m + 1):
        if th == ph and text[i:i + m] == pattern:
            return i
        if i < n - m:
            th = ((th - ord(text[i]) * high) * base + ord(text[i + m])) % mod
            if th < 0:
                th += mod
    return -1


def bm_bad(p):
    t = {}
    for i, c in enumerate(p):
        t[c] = i
    return t


def bm_suff(p):
    m = len(p)
    s = [0] * m
    s[m - 1] = m
    g = m - 1
    f = 0
    for i in range(m - 2, -1, -1):
        if i > g and s[i + m - 1 - f] < i - g:
            s[i] = s[i + m - 1 - f]
        else:
            if i < g:
                g = i
            f = i
            while g >= 0 and p[g] == p[g + m - 1 - f]:
                g -= 1
            s[i] = f - g
    return s


def bm_good(p):
    m = len(p)
    suff = bm_suff(p)
    gs = [m] * m
    j = 0
    for i in range(m - 1, -1, -1):
        if suff[i] == i + 1:
            while j < m - 1 - i:
                if gs[j] == m:
                    gs[j] = m - 1 - i
                j += 1
    for i in range(m - 1):
        gs[m - 1 - suff[i]] = m - 1 - i
    return gs


def boyer_moore_search(text: str, pattern: str) -> int:
    n, m = len(text), len(pattern)
    if m == 0:
        return 0
    if m > n:
        return -1
    bad = bm_bad(pattern)
    gs = bm_good(pattern)
    i = 0
    while i <= n - m:
        j = m - 1
        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1
        if j < 0:
            return i
        c = text[i + j]
        bc = j - bad.get(c, -1)
        i += max(bc, gs[j])
    return -1


# ---- timing ----
def time_median(fn, text, pat, repeats=11):
    timer = timeit.Timer(lambda: fn(text, pat))
    runs = timer.repeat(repeat=repeats, number=1)
    return stats.median(runs)


# ---- pretty console table ----
def print_box_table(rows):
    headers = ["Текст","Підрядок існує?","Алгоритм",
               "Час, с (медіана 11)","Довжина тексту","Довжина підрядка"]
    data = [[r[0], r[1], r[2], f"{r[3]:.6f}",
             f"{r[4]:,}".replace(",", " "), f"{r[5]:,}".replace(",", " ")]
            for r in rows]
    num_cols = {3,4,5}
    widths = [max(len(headers[i]), *(len(row[i]) for row in data))
              for i in range(len(headers))]

    def fmt_row(row):
        cells = []
        for i,cell in enumerate(row):
            if i in num_cols:
                cells.append(cell.rjust(widths[i]))
            elif i == 1:
                pad = widths[i] - len(cell)
                cells.append(" "*(pad//2) + cell + " "*(pad - pad//2))
            else:
                cells.append(cell.ljust(widths[i]))
        return "| " + " | ".join(cells) + " |"

    sep = "+-" + "-+-".join("-"*w for w in widths) + "-+"
    sep_bold = "+=" + "=+=".join("="*w for w in widths) + "=+"

    print(sep)
    print(fmt_row(headers))
    print(sep_bold)
    for row in data:
        print(fmt_row(row))
        print(sep)


# ---- markdown table (для REPORT.md) ----
def md_table(rows):
    head = (
        "| Текст | Підрядок існує? | Алгоритм | Час, с (медіана 11) | Довжина тексту | Довжина підрядка |\n"
        "|:------|:---------------:|:---------|--------------------:|---------------:|-----------------:|\n"
    )

    def fmt_int(n: int) -> str:
        return f"{n:,}".replace(",", " ")
    body = "".join(f"| {a} | {b} | {c} | {d:.6f} | {fmt_int(e)} | {fmt_int(f)} |\n"
                   for a, b, c, d, e, f in rows)
    return head + body


# ---- main ----
def main():
    ap = argparse.ArgumentParser(description="Minimal substring benchmark (console + REPORT.md)")
    ap.add_argument("article1")
    ap.add_argument("article2")
    ap.add_argument("--exist1")
    ap.add_argument("--absent1")
    ap.add_argument("--exist2")
    ap.add_argument("--absent2")
    ap.add_argument("--repeats", type=int, default=11)
    ap.add_argument("--report", default="REPORT.md")
    ap.add_argument("--no-report", action="store_true", help="не створювати REPORT.md")
    args = ap.parse_args()

    t1 = read_text_robust(Path(args.article1))
    t2 = read_text_robust(Path(args.article2))

    # choose substrings
    def auto_pick_exist(text):
        mid = len(text) // 2
        return text[max(0, mid - 20):mid + 10].replace("\n", " ").strip()

    def auto_pick_absent(text):
        import random, string
        for _ in range(200):
            s = "".join(random.choices(string.ascii_lowercase + string.digits, k=32))
            if s not in text:
                return s
        return "__no__"

    e1 = args.exist1 or auto_pick_exist(t1)
    a1 = args.absent1 or auto_pick_absent(t1)
    e2 = args.exist2 or auto_pick_exist(t2)
    a2 = args.absent2 or auto_pick_absent(t2)

    cases = [
        ("стаття 1", t1, e1, True),
        ("стаття 1", t1, a1, False),
        ("стаття 2", t2, e2, True),
        ("стаття 2", t2, a2, False),
    ]
    algs = [
        ("Boyer–Moore", boyer_moore_search),
        ("KMP", kmp_search),
        ("Rabin–Karp", rabin_karp_search),
    ]

    rows = []
    for label, txt, pat, exists in cases:
        for name, fn in algs:
            t = time_median(fn, txt, pat, repeats=args.repeats)
            rows.append((label, "так" if exists else "ні", name, t, len(txt), len(pat)))

    # winners
    by_text = {}
    for r in rows:
        by_text.setdefault(r[0], []).append(r)
    winners = {k: min(v, key=lambda x: x[3]) for k, v in by_text.items()}

    by_alg = {}
    for r in rows:
        by_alg.setdefault(r[2], []).append(r[3])
    overall = min(by_alg, key=lambda a: stats.median(by_alg[a]))

    print_box_table(rows)
    print("\nНайшвидші:")
    for k in sorted(winners):
        print(f"- {k}: {winners[k][2]}")
    print(f"- Загалом: {overall}")

    if not args.no_report:
        lines = []
        lines.append("# Порівняння алгоритмів пошуку підрядка (мінімальний звіт)\n")
        lines.append("**Медіана часу 11 запусків (`timeit`) для Boyer–Moore, KMP, Rabin–Karp.**\n")
        lines.append("## Обрані підрядки\n")
        lines.append(f"- стаття 1 (існує): `{e1}`")
        lines.append(f"- стаття 1 (відсутній): `{a1}`")
        lines.append(f"- стаття 2 (існує): `{e2}`")
        lines.append(f"- стаття 2 (відсутній): `{a2}`\n")
        lines.append("## Результати\n")
        lines.append(md_table(rows))
        lines.append("\n## Найшвидші")
        for k in sorted(winners):
            lines.append(f"- **{k}**: {winners[k][2]}")
        lines.append(f"- **Загалом**: **{overall}**\n")
        Path(args.report).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
