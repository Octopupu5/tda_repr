import tempfile
from pathlib import Path

from tools.aggregate.reproduction_tables import strip_cites_in_tex_dir


def test_strip_cites_in_tex_dir_removes_all_cite_commands() -> None:
	with tempfile.TemporaryDirectory() as td:
		tdp = Path(td)
		(tdp / "a.tex").write_text(r"Hello \cite{b1} world \citep{b2}.", encoding="utf-8")
		(tdp / "b.tex").write_text(r"\toprule x \citet{b3} y \citealp{b4}", encoding="utf-8")
		(tdp / "note.txt").write_text(r"\cite{b5} should not be touched", encoding="utf-8")

		strip_cites_in_tex_dir(str(tdp))

		a = (tdp / "a.tex").read_text(encoding="utf-8")
		b = (tdp / "b.tex").read_text(encoding="utf-8")
		note = (tdp / "note.txt").read_text(encoding="utf-8")

		assert r"\cite" not in a
		assert r"\cite" not in b
		assert note == r"\cite{b5} should not be touched"


def test_strip_cites_in_tex_dir_keeps_tex_without_cites_unchanged() -> None:
	with tempfile.TemporaryDirectory() as td:
		tdp = Path(td)
		src = r"\toprule A & B \\ \midrule 1 & 2 \\ \bottomrule"
		(tdp / "t.tex").write_text(src, encoding="utf-8")
		strip_cites_in_tex_dir(str(tdp))
		out = (tdp / "t.tex").read_text(encoding="utf-8")
		assert out == src

