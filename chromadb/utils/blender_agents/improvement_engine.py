"""
Continuous Improvement Engine - Self-learning and quality feedback loops.

Drives autonomous improvement through:
- Quality metric tracking and trend analysis
- Automatic parameter tuning
- A/B testing of approaches
- Learning from successes and failures
- Generating improvement suggestions
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of quality metrics."""
    RENDER_QUALITY = "render_quality"
    GEOMETRY_QUALITY = "geometry_quality"
    MATERIAL_QUALITY = "material_quality"
    PERFORMANCE = "performance"
    ACCURACY = "accuracy"
    EFFICIENCY = "efficiency"
    CREATIVITY = "creativity"


@dataclass
class QualityMetric:
    """A tracked quality metric with history."""
    name: str
    metric_type: MetricType
    values: List[float] = field(default_factory=list)
    timestamps: List[float] = field(default_factory=list)
    target: float = 0.8
    weight: float = 1.0

    @property
    def current(self) -> float:
        return self.values[-1] if self.values else 0.0

    @property
    def average(self) -> float:
        return sum(self.values) / len(self.values) if self.values else 0.0

    @property
    def trend(self) -> str:
        """Calculate trend direction."""
        if len(self.values) < 3:
            return "insufficient_data"
        recent = sum(self.values[-3:]) / 3
        overall = self.average
        if recent > overall * 1.05:
            return "improving"
        elif recent < overall * 0.95:
            return "declining"
        return "stable"

    @property
    def meets_target(self) -> bool:
        return self.current >= self.target

    def record(self, value: float) -> None:
        self.values.append(value)
        self.timestamps.append(time.time())


@dataclass
class ImprovementSuggestion:
    """A suggested improvement with rationale."""
    suggestion_id: str
    category: str
    description: str
    rationale: str
    priority: str = "medium"  # low, medium, high, critical
    estimated_impact: float = 0.0  # 0-1
    parameters: Dict[str, Any] = field(default_factory=dict)
    applied: bool = False
    result_delta: Optional[float] = None  # quality change after applying


@dataclass
class FeedbackLoop:
    """
    A feedback loop for continuous improvement.

    Tracks input parameters, output quality, and learns correlations
    to automatically improve future operations.
    """
    name: str
    input_params: Dict[str, List[Any]] = field(default_factory=dict)
    output_scores: List[float] = field(default_factory=list)
    correlations: Dict[str, float] = field(default_factory=dict)
    optimal_params: Dict[str, Any] = field(default_factory=dict)

    def record(self, params: Dict[str, Any], score: float) -> None:
        """Record an observation."""
        for key, value in params.items():
            if key not in self.input_params:
                self.input_params[key] = []
            self.input_params[key].append(value)
        self.output_scores.append(score)

        # Update correlations and optimal params
        if len(self.output_scores) >= 3:
            self._update_correlations()
            self._update_optimal_params()

    def _update_correlations(self) -> None:
        """Calculate correlations between parameters and scores."""
        for key, values in self.input_params.items():
            if all(isinstance(v, (int, float)) for v in values):
                n = min(len(values), len(self.output_scores))
                if n < 3:
                    continue
                # Simple correlation calculation
                vals = values[-n:]
                scores = self.output_scores[-n:]
                mean_v = sum(vals) / n
                mean_s = sum(scores) / n

                numerator = sum(
                    (v - mean_v) * (s - mean_s)
                    for v, s in zip(vals, scores)
                )
                denom_v = sum((v - mean_v) ** 2 for v in vals) ** 0.5
                denom_s = sum((s - mean_s) ** 2 for s in scores) ** 0.5

                if denom_v > 0 and denom_s > 0:
                    self.correlations[key] = numerator / (denom_v * denom_s)

    def _update_optimal_params(self) -> None:
        """Find parameters that produced the best results."""
        if not self.output_scores:
            return

        best_idx = self.output_scores.index(max(self.output_scores))

        for key, values in self.input_params.items():
            if best_idx < len(values):
                self.optimal_params[key] = values[best_idx]

    def suggest_params(self) -> Dict[str, Any]:
        """Suggest optimal parameters based on learned data."""
        return self.optimal_params.copy()


class ImprovementEngine:
    """
    Continuous improvement engine for autonomous quality optimization.

    Features:
    - Multi-metric quality tracking
    - Feedback loop management
    - A/B testing of parameter variations
    - Automatic suggestion generation
    - Trend analysis and anomaly detection
    """

    def __init__(self):
        self.metrics: Dict[str, QualityMetric] = {}
        self.feedback_loops: Dict[str, FeedbackLoop] = {}
        self.suggestions: List[ImprovementSuggestion] = []
        self.experiments: List[Dict[str, Any]] = []
        self._suggestion_counter = 0

    def register_metric(
        self,
        name: str,
        metric_type: MetricType,
        target: float = 0.8,
        weight: float = 1.0,
    ) -> QualityMetric:
        """
        Register a quality metric to track.

        Args:
            name: Metric name
            metric_type: Type of metric
            target: Target value (0-1)
            weight: Weight for aggregate scoring

        Returns:
            Created QualityMetric
        """
        metric = QualityMetric(
            name=name,
            metric_type=metric_type,
            target=target,
            weight=weight,
        )
        self.metrics[name] = metric
        logger.info(f"Registered metric: {name} (target={target})")
        return metric

    def record_metric(self, name: str, value: float) -> None:
        """Record a metric value."""
        if name not in self.metrics:
            self.register_metric(name, MetricType.ACCURACY)
        self.metrics[name].record(value)

    def create_feedback_loop(self, name: str) -> FeedbackLoop:
        """Create a new feedback loop."""
        loop = FeedbackLoop(name=name)
        self.feedback_loops[name] = loop
        return loop

    def record_feedback(
        self,
        loop_name: str,
        params: Dict[str, Any],
        score: float,
    ) -> None:
        """Record feedback for a loop."""
        if loop_name not in self.feedback_loops:
            self.create_feedback_loop(loop_name)
        self.feedback_loops[loop_name].record(params, score)

    def get_aggregate_quality(self) -> float:
        """Calculate weighted aggregate quality score."""
        if not self.metrics:
            return 0.0

        total_weight = sum(m.weight for m in self.metrics.values())
        if total_weight == 0:
            return 0.0

        weighted_sum = sum(
            m.current * m.weight
            for m in self.metrics.values()
        )
        return weighted_sum / total_weight

    def generate_suggestions(self) -> List[ImprovementSuggestion]:
        """
        Analyze all metrics and feedback loops to generate improvement suggestions.

        Returns:
            List of actionable improvement suggestions
        """
        new_suggestions = []

        # Check metrics below target
        for name, metric in self.metrics.items():
            if not metric.meets_target and metric.current > 0:
                self._suggestion_counter += 1
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"sug-{self._suggestion_counter}",
                    category="metric_improvement",
                    description=f"Improve {name} from {metric.current:.2f} to {metric.target:.2f}",
                    rationale=f"Metric '{name}' is below target "
                              f"({metric.current:.2f} < {metric.target:.2f}), "
                              f"trend: {metric.trend}",
                    priority="high" if metric.current < metric.target * 0.5 else "medium",
                    estimated_impact=(metric.target - metric.current) * metric.weight,
                )
                new_suggestions.append(suggestion)

        # Check declining metrics
        for name, metric in self.metrics.items():
            if metric.trend == "declining":
                self._suggestion_counter += 1
                suggestion = ImprovementSuggestion(
                    suggestion_id=f"sug-{self._suggestion_counter}",
                    category="trend_alert",
                    description=f"Address declining trend in {name}",
                    rationale=f"Metric '{name}' is showing a declining trend",
                    priority="high",
                    estimated_impact=0.3,
                )
                new_suggestions.append(suggestion)

        # Generate parameter optimization suggestions from feedback loops
        for loop_name, loop in self.feedback_loops.items():
            if loop.optimal_params and loop.correlations:
                # Find high-correlation parameters
                for param, corr in loop.correlations.items():
                    if abs(corr) > 0.5:
                        self._suggestion_counter += 1
                        direction = "increase" if corr > 0 else "decrease"
                        optimal = loop.optimal_params.get(param)
                        suggestion = ImprovementSuggestion(
                            suggestion_id=f"sug-{self._suggestion_counter}",
                            category="parameter_optimization",
                            description=f"{direction.title()} '{param}' in {loop_name} "
                                        f"(optimal: {optimal})",
                            rationale=f"Strong correlation ({corr:.2f}) between "
                                      f"'{param}' and quality score",
                            priority="medium",
                            estimated_impact=abs(corr) * 0.5,
                            parameters={param: optimal},
                        )
                        new_suggestions.append(suggestion)

        self.suggestions.extend(new_suggestions)
        return new_suggestions

    def create_experiment(
        self,
        name: str,
        parameter: str,
        values: List[Any],
        metric_to_track: str,
    ) -> Dict[str, Any]:
        """
        Create an A/B experiment to test parameter variations.

        Args:
            name: Experiment name
            parameter: Parameter to vary
            values: Values to test
            metric_to_track: Metric to measure impact

        Returns:
            Experiment definition
        """
        experiment = {
            "name": name,
            "parameter": parameter,
            "values": values,
            "metric": metric_to_track,
            "results": {},
            "status": "pending",
            "created_at": time.time(),
            "best_value": None,
            "best_score": 0.0,
        }
        self.experiments.append(experiment)
        logger.info(f"Created experiment: {name} ({len(values)} variations)")
        return experiment

    def record_experiment_result(
        self,
        experiment_name: str,
        value: Any,
        score: float,
    ) -> None:
        """Record a result for an experiment."""
        for exp in self.experiments:
            if exp["name"] == experiment_name:
                exp["results"][str(value)] = score
                if score > exp["best_score"]:
                    exp["best_score"] = score
                    exp["best_value"] = value

                # Check if experiment is complete
                if len(exp["results"]) >= len(exp["values"]):
                    exp["status"] = "completed"
                    logger.info(
                        f"Experiment '{experiment_name}' completed. "
                        f"Best: {exp['best_value']} (score={exp['best_score']:.2f})"
                    )
                break

    def get_improvement_report(self) -> Dict[str, Any]:
        """Generate a comprehensive improvement report."""
        return {
            "aggregate_quality": f"{self.get_aggregate_quality():.2f}",
            "metrics": {
                name: {
                    "current": f"{m.current:.2f}",
                    "average": f"{m.average:.2f}",
                    "target": f"{m.target:.2f}",
                    "trend": m.trend,
                    "meets_target": m.meets_target,
                    "samples": len(m.values),
                }
                for name, m in self.metrics.items()
            },
            "feedback_loops": {
                name: {
                    "observations": len(loop.output_scores),
                    "optimal_params": loop.optimal_params,
                    "top_correlations": {
                        k: f"{v:.2f}"
                        for k, v in sorted(
                            loop.correlations.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )[:5]
                    },
                }
                for name, loop in self.feedback_loops.items()
            },
            "active_suggestions": len([s for s in self.suggestions if not s.applied]),
            "experiments": {
                exp["name"]: {
                    "status": exp["status"],
                    "best_value": exp["best_value"],
                    "best_score": f"{exp['best_score']:.2f}" if exp["best_score"] else "N/A",
                    "progress": f"{len(exp['results'])}/{len(exp['values'])}",
                }
                for exp in self.experiments
            },
        }

    def apply_suggestion(self, suggestion_id: str) -> Optional[Dict[str, Any]]:
        """
        Mark a suggestion as applied and return its parameters.

        Args:
            suggestion_id: ID of suggestion to apply

        Returns:
            Parameters from the suggestion, if any
        """
        for suggestion in self.suggestions:
            if suggestion.suggestion_id == suggestion_id:
                suggestion.applied = True
                logger.info(f"Applied suggestion: {suggestion.description}")
                return suggestion.parameters
        return None

    def auto_optimize(self) -> Dict[str, Any]:
        """
        Automatically apply the best available optimizations.

        Returns:
            Dict of optimized parameters across all feedback loops
        """
        optimized_params = {}

        for loop_name, loop in self.feedback_loops.items():
            suggested = loop.suggest_params()
            if suggested:
                optimized_params[loop_name] = suggested

        logger.info(f"Auto-optimized {len(optimized_params)} parameter sets")
        return optimized_params
