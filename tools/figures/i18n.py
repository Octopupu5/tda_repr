from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class I18N:
	lang: str

	def __post_init__(self) -> None:
		lang = str(self.lang).lower().strip()
		if lang not in {"en", "ru"}:
			raise ValueError(f"Unsupported --lang: {self.lang!r}. Expected 'en' or 'ru'.")
		object.__setattr__(self, "lang", lang)

	def epoch(self) -> str:
		return "Epoch" if self.lang == "en" else "Эпоха"

	def validation_pct(self) -> str:
		return "Validation (%)" if self.lang == "en" else "Валидация (%)"

	def validation_value(self) -> str:
		return "Validation value" if self.lang == "en" else "Значение на валидации"

	def signal_value(self) -> str:
		return "Signal value" if self.lang == "en" else "Значение сигнала"

	def accuracy(self) -> str:
		return "Accuracy" if self.lang == "en" else "Доля правильных ответов"

	def macro_f1(self) -> str:
		return "Macro-F1" if self.lang == "en" else "Макро-F1"

	def perplexity(self) -> str:
		return "Perplexity" if self.lang == "en" else "Перплексия"

	def loss(self) -> str:
		return "Loss" if self.lang == "en" else "Лосс"

	def loss_assistant_only(self) -> str:
		return "Loss (assistant-only)" if self.lang == "en" else "Лосс (assistant-only)"

	def early_stop(self) -> str:
		return "Early-stop" if self.lang == "en" else "Ранняя остановка"

	def empirical_best(self) -> str:
		return "Empirical best" if self.lang == "en" else "Эмпирически лучший"

	def plateau_min(self) -> str:
		return "min-plateau" if self.lang == "en" else "плато минимума"

	def plateau_max(self) -> str:
		return "max-plateau" if self.lang == "en" else "плато максимума"

	def depth_tag(self, depth: str) -> str:
		d = str(depth).lower().strip()
		if d == "early":
			return "early" if self.lang == "en" else "ранний"
		if d in {"mid", "middle", "intermediate"}:
			return "mid" if self.lang == "en" else "промежуточный"
		if d == "deep":
			return "deep" if self.lang == "en" else "глубокий"
		return depth

	def bench_metric_label(self, metric: str) -> str:
		m = str(metric).lower().strip()
		if m == "accuracy":
			return self.accuracy()
		if m == "f1_macro":
			return self.macro_f1()
		if m == "ppl":
			return self.perplexity()
		if m == "loss_assistant_only":
			return self.loss_assistant_only()
		if m == "loss":
			return self.loss()
		return metric
