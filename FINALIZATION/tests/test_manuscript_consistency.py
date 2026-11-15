import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_manuscript_consistency():
    n = json.loads((ROOT / "outputs/evaluation/fusion/iccies2027/numbers.json").read_text(encoding="utf-8"))
    readme = (ROOT / "papers/iccies2027/README.md").read_text(encoding="utf-8")
    assert "alpha_H=0.2" in readme
    assert "alpha_O=0.9" in readme
    assert "tau_H=0.14" in readme
    assert "tau_O=0.66" in readme

    # Table values.
    fusion_table = (ROOT / "papers/iccies2027/tables/fusion_variants.tex").read_text(encoding="utf-8")
    assert "0.6575 & 0.5830 & 0.4493 & 0.9403" in fusion_table
    assert "0.6602" in fusion_table
    backbones = (ROOT / "papers/iccies2027/tables/backbones.tex").read_text(encoding="utf-8")
    assert "0.6575 & \\textbf{0.6602}" in backbones
    assert "15/23" in backbones

    # Figures exist and are non-empty.
    assert (ROOT / "papers/iccies2027/figures/architecture.pdf").stat().st_size > 1000
    assert (ROOT / "papers/iccies2027/figures/results.pdf").stat().st_size > 1000

    # Citation audit counts.
    cited = set()
    for path in (ROOT / "papers/iccies2027").rglob("*.tex"):
        for match in re.finditer(r"\\cite[a-z]*\{([^}]*)\}", path.read_text(encoding="utf-8")):
            for key in match.group(1).split(","):
                cited.add(key.strip())
    bib = set()
    for line in (ROOT / "papers/iccies2027/references.bib").read_text(encoding="utf-8").splitlines():
        m = re.match(r"@\w+\{([^,]+),", line)
        if m:
            bib.add(m.group(1))
    assert len(cited) == 30
    assert len(bib) == 36
    assert len(bib - cited) == 6
