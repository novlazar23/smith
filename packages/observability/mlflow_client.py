"""MLflow — Experiment-Tracking für Trading-Modelle und Strategies.

Stellt eine Schicht für MLflow-Experiment-Tracking mit:
- Parameter- und Metriken-Recording
- Modell-Artefakt-Registration
- Vergleich von Backtest-Runs
"""

from __future__ import annotations

from typing import Any


class MLflowClient:
    """MLflow-Client für Trading-Orchestra Experiment-Tracking.

    Verwaltet Experimente, Runs und Artefakte für ML-Modelle
    und Trading-Strategien.
    """

    def __init__(self, tracking_uri: str = "http://localhost:5000", experiment_name: str = "trading-orchestra") -> None:
        """Initialisiert den MLflow-Client.

        Args:
            tracking_uri: MLflow-Server-URI.
            experiment_name: Standard-Experiment-Name.
        """
        self._tracking_uri = tracking_uri
        self._experiment_name = experiment_name
        self._client: Any = None
        self._experiment_id: str | None = None

    def _ensure_initialized(self) -> None:
        """Initialisiert den MLflow-Client lazy."""
        if self._client is not None:
            return

        import mlflow
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self._tracking_uri)
        self._client = MlflowClient(tracking_uri=self._tracking_uri)

        # Experiment erstellen oder existierendes finden
        experiment = self._client.get_experiment_by_name(self._experiment_name)
        if experiment is None:
            self._experiment_id = self._client.create_experiment(
                self._experiment_name,
                tags={"type": "trading", "orchestra_version": "0.1.0"},
            )
        else:
            self._experiment_id = experiment.experiment_id

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Startet einen neuen MLflow-Run und gibt die Run-ID zurück.

        Args:
            run_name: Optionaler Name für den Run.
            tags: Zusätzliche Tags für den Run.

        Returns:
            Die Run-ID als String.
        """
        self._ensure_initialized()

        import mlflow

        with mlflow.start_run(experiment_id=self._experiment_id, run_name=run_name) as run:
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            return run.info.run_id

    def log_parameters(self, run_id: str, params: dict[str, Any]) -> None:
        """Recordet Parameter für einen Run.

        Args:
            run_id: Die Run-ID.
            params: Parameter-Dict (z. B. Hyperparameter).
        """
        self._ensure_initialized()
        import mlflow

        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, run_id: str, metrics: dict[str, float]) -> None:
        """Recordet Metriken für einen Run.

        Args:
            run_id: Die Run-ID.
            metrics: Metriken-Dict (z. B. Sharpe Ratio, Drawdown).
        """
        self._ensure_initialized()
        import mlflow

        for key, value in metrics.items():
            mlflow.log_metric(key, value)

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str | None = None) -> None:
        """Recordet ein Artefakt für einen Run.

        Args:
            run_id: Die Run-ID.
            local_path: Lokaler Pfad zum Artefakt.
            artifact_path: Optionaler Unterpfad im Artifact-Store.
        """
        self._ensure_initialized()
        import mlflow

        mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, run_id: str, model: object, model_name: str) -> None:
        """Recordet ein ML-Modell als Artefakt.

        Args:
            run_id: Die Run-ID.
            model: Das zu recordende Modell-Objekt.
            model_name: Name des Modells.
        """
        self._ensure_initialized()
        import mlflow

        # Versucht das Modell mit dem passenden Flavor zu loggen
        mlflow.sklearn.log_model(model, model_name)

    def end_run(self, run_id: str, status: str = "SUCCESS") -> None:
        """Beendet einen MLflow-Run mit Status.

        Args:
            run_id: Die Run-ID.
            status: Status ("SUCCESS", "FAILED", "RUNNING").
        """
        self._ensure_initialized()
        import mlflow

        mlflow.set_tag("run_status", status)
        mlflow.set_tag("orchestra_component", "mlflow-tracker")

    def search_runs(self, filter_string: str | None = None, max_results: int = 20) -> list[Any]:
        """Sucht Runs basierend auf Filtern.

        Args:
            filter_string: MLflow-Filter-String.
            max_results: Maximale Anzahl von Ergebnissen.

        Returns:
            Liste von Run-Objekten.
        """
        self._ensure_initialized()
        import mlflow

        return mlflow.search_runs(
            experiment_ids=[self._experiment_id] if self._experiment_id else None,
            filter_string=filter_string,
            max_results=max_results,
            order_by=["metrics.sharpe_ratio DESC"],
        )

    @property
    def is_available(self) -> bool:
        """Prüft ob der MLflow-Server erreichbar ist."""
        try:
            self._ensure_initialized()
            self._client.get_experiment(self._experiment_id)
            return True
        except Exception:
            return False
