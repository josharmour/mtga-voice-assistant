"""
Model Registry System

Manages versioning, storage, and retrieval of RL models with SQLite database.
Supports model comparison, rollback, and metadata tracking for compliance.

Constitutional Requirements:
- Verifiable Testing: Model versioning and validation tracking
- Explainable AI: Model metadata and explainability features tracking
- Data-Driven AI Development: Training data and performance metrics tracking
"""

import json
import sqlite3
import threading
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Registry for managing RL models with versioning, metadata, and validation tracking.

    Uses SQLite for thread-safe storage of model metadata, performance metrics,
    and compliance information.
    """

    def __init__(self, db_path: str = "data/model_registry.db"):
        """
        Initialize model registry with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._local = threading.local()
        self._ensure_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'connection'):
            self._local.connection = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                timeout=30.0
            )
            self._local.connection.row_factory = sqlite3.Row
            self._local.connection.execute("PRAGMA foreign_keys = ON")
            self._local.connection.execute("PRAGMA journal_mode = WAL")
        return self._local.connection

    def _ensure_database(self) -> None:
        """Create database and tables if they don't exist."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        conn = self._get_connection()
        cursor = conn.cursor()

        # Models table - core model information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT UNIQUE NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                model_type TEXT NOT NULL,
                file_path TEXT NOT NULL,
                config_json TEXT,
                architecture_hash TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'training',
                file_size_bytes INTEGER,
                is_active BOOLEAN DEFAULT FALSE
            )
        """)

        # Training metadata table - training information
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS training_metadata (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                training_data_source TEXT,
                training_episodes INTEGER,
                training_duration_seconds REAL,
                training_loss REAL,
                validation_loss REAL,
                hyperparameters_json TEXT,
                training_start_time TIMESTAMP,
                training_end_time TIMESTAMP,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # Performance metrics table - constitutional compliance
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                test_conditions TEXT,
                measurement_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence_interval_lower REAL,
                confidence_interval_upper REAL,
                sample_size INTEGER,
                is_benchmark BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # Validation results table - testing and validation
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS validation_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                test_type TEXT NOT NULL,
                test_result TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                details_json TEXT,
                test_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                constitutional_compliance BOOLEAN,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # Explainability features table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS explainability_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                feature_type TEXT NOT NULL,
                feature_enabled BOOLEAN NOT NULL,
                implementation_details TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # Model lineage table - for continual learning tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_lineage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                parent_model_id TEXT,
                child_model_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                transfer_method TEXT,
                FOREIGN KEY (parent_model_id) REFERENCES models (model_id),
                FOREIGN KEY (child_model_id) REFERENCES models (model_id)
            )
        """)

        # Deployment history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS deployment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT NOT NULL,
                deployment_environment TEXT NOT NULL,
                deployment_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                deployment_status TEXT,
                rollback_time TIMESTAMP,
                notes TEXT,
                FOREIGN KEY (model_id) REFERENCES models (model_id)
            )
        """)

        # Create indexes for performance
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_model_id ON models (model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_models_status ON models (status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_performance_model_id ON performance_metrics (model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_validation_model_id ON validation_results (model_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_training_model_id ON training_metadata (model_id)")

        conn.commit()
        logger.info(f"Model registry database ensured at {self.db_path}")

    def register_model(self, model_info: Dict[str, Any]) -> str:
        """
        Register a new model version.

        Args:
            model_info: Dictionary containing model information

        Returns:
            Model ID of the registered model
        """
        required_fields = ['model_name', 'model_version', 'model_type', 'file_path']
        for field in required_fields:
            if field not in model_info:
                raise ValueError(f"Required field missing: {field}")

        model_id = f"{model_info['model_name']}_v{model_info['model_version']}_{int(datetime.now().timestamp())}"

        conn = self._get_connection()
        cursor = conn.cursor()

        # Calculate file size
        file_size = None
        if os.path.exists(model_info['file_path']):
            file_size = os.path.getsize(model_info['file_path'])

        try:
            cursor.execute("""
                INSERT INTO models (
                    model_id, model_name, model_version, model_type,
                    file_path, config_json, architecture_hash,
                    file_size_bytes, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                model_info['model_name'],
                model_info['model_version'],
                model_info['model_type'],
                model_info['file_path'],
                json.dumps(model_info.get('config', {})),
                model_info.get('architecture_hash'),
                file_size,
                model_info.get('status', 'training')
            ))

            # Register explainability features if provided
            if 'explainability_features' in model_info:
                for feature_type, feature_info in model_info['explainability_features'].items():
                    cursor.execute("""
                        INSERT INTO explainability_features (
                            model_id, feature_type, feature_enabled, implementation_details
                        ) VALUES (?, ?, ?, ?)
                    """, (
                        model_id,
                        feature_type,
                        feature_info.get('enabled', False),
                        json.dumps(feature_info.get('details', {}))
                    ))

            conn.commit()
            logger.info(f"Registered model: {model_id}")
            return model_id

        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise ValueError(f"Model registration failed: {e}")

    def record_training_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """
        Record training metadata for a model.

        Args:
            model_id: Model ID
            metadata: Training metadata dictionary
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO training_metadata (
                model_id, training_data_source, training_episodes,
                training_duration_seconds, training_loss, validation_loss,
                hyperparameters_json, training_start_time, training_end_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            metadata.get('training_data_source'),
            metadata.get('training_episodes'),
            metadata.get('training_duration_seconds'),
            metadata.get('training_loss'),
            metadata.get('validation_loss'),
            json.dumps(metadata.get('hyperparameters', {})),
            metadata.get('training_start_time'),
            metadata.get('training_end_time')
        ))

        conn.commit()
        logger.info(f"Recorded training metadata for {model_id}")

    def record_performance_metrics(self, model_id: str, metrics: List[Dict[str, Any]]) -> None:
        """
        Record performance metrics for a model.

        Args:
            model_id: Model ID
            metrics: List of performance metric dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        for metric in metrics:
            cursor.execute("""
                INSERT INTO performance_metrics (
                    model_id, metric_name, metric_value, test_conditions,
                    confidence_interval_lower, confidence_interval_upper,
                    sample_size, is_benchmark
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                model_id,
                metric['metric_name'],
                metric['metric_value'],
                metric.get('test_conditions'),
                metric.get('confidence_interval_lower'),
                metric.get('confidence_interval_upper'),
                metric.get('sample_size'),
                metric.get('is_benchmark', False)
            ))

        conn.commit()
        logger.info(f"Recorded {len(metrics)} performance metrics for {model_id}")

    def record_validation_result(self, model_id: str, validation_info: Dict[str, Any]) -> None:
        """
        Record validation/test results for a model.

        Args:
            model_id: Model ID
            validation_info: Validation information dictionary
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO validation_results (
                model_id, test_type, test_result, success,
                details_json, constitutional_compliance
            ) VALUES (?, ?, ?, ?, ?, ?)
        """, (
            model_id,
            validation_info['test_type'],
            validation_info['test_result'],
            validation_info['success'],
            json.dumps(validation_info.get('details', {})),
            validation_info.get('constitutional_compliance')
        ))

        conn.commit()
        logger.info(f"Recorded validation result for {model_id}: {validation_info['test_type']}")

    def get_model(self, model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get model information by ID.

        Args:
            model_id: Model ID to retrieve

        Returns:
            Model information dictionary or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM models WHERE model_id = ?
        """, (model_id,))

        row = cursor.fetchone()
        if not row:
            return None

        model_info = {
            'model_id': row['model_id'],
            'model_name': row['model_name'],
            'model_version': row['model_version'],
            'model_type': row['model_type'],
            'file_path': row['file_path'],
            'config': json.loads(row['config_json']) if row['config_json'] else {},
            'architecture_hash': row['architecture_hash'],
            'created_at': row['created_at'],
            'updated_at': row['updated_at'],
            'status': row['status'],
            'file_size_bytes': row['file_size_bytes'],
            'is_active': bool(row['is_active'])
        }

        return model_info

    def list_models(self, model_type: Optional[str] = None,
                   status: Optional[str] = None,
                   limit: int = 100) -> List[Dict[str, Any]]:
        """
        List models with optional filtering.

        Args:
            model_type: Filter by model type
            status: Filter by status
            limit: Maximum number of results

        Returns:
            List of model information dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM models WHERE 1=1"
        params = []

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)

        models = []
        for row in cursor.fetchall():
            models.append({
                'model_id': row['model_id'],
                'model_name': row['model_name'],
                'model_version': row['model_version'],
                'model_type': row['model_type'],
                'file_path': row['file_path'],
                'status': row['status'],
                'created_at': row['created_at'],
                'is_active': bool(row['is_active'])
            })

        return models

    def get_active_model(self, model_type: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get the currently active model.

        Args:
            model_type: Optional filter by model type

        Returns:
            Active model information or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        query = "SELECT * FROM models WHERE is_active = TRUE"
        params = []

        if model_type:
            query += " AND model_type = ?"
            params.append(model_type)

        cursor.execute(query, params)
        row = cursor.fetchone()

        if not row:
            return None

        return {
            'model_id': row['model_id'],
            'model_name': row['model_name'],
            'model_version': row['model_version'],
            'model_type': row['model_type'],
            'file_path': row['file_path'],
            'created_at': row['created_at']
        }

    def set_active_model(self, model_id: str) -> bool:
        """
        Set a model as the active model.

        Args:
            model_id: Model ID to activate

        Returns:
            True if successful, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        try:
            # Deactivate all models of the same type
            cursor.execute("""
                UPDATE models SET is_active = FALSE
                WHERE model_type = (SELECT model_type FROM models WHERE model_id = ?)
            """, (model_id,))

            # Activate the specified model
            cursor.execute("""
                UPDATE models SET is_active = TRUE, updated_at = CURRENT_TIMESTAMP
                WHERE model_id = ?
            """, (model_id,))

            success = cursor.rowcount > 0
            conn.commit()

            if success:
                logger.info(f"Activated model: {model_id}")
            else:
                logger.warning(f"Model not found: {model_id}")

            return success

        except Exception as e:
            conn.rollback()
            logger.error(f"Failed to activate model {model_id}: {e}")
            return False

    def get_performance_metrics(self, model_id: str) -> List[Dict[str, Any]]:
        """
        Get performance metrics for a model.

        Args:
            model_id: Model ID

        Returns:
            List of performance metrics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM performance_metrics
            WHERE model_id = ?
            ORDER BY measurement_time DESC
        """, (model_id,))

        metrics = []
        for row in cursor.fetchall():
            metrics.append({
                'metric_name': row['metric_name'],
                'metric_value': row['metric_value'],
                'test_conditions': row['test_conditions'],
                'confidence_interval_lower': row['confidence_interval_lower'],
                'confidence_interval_upper': row['confidence_interval_upper'],
                'sample_size': row['sample_size'],
                'is_benchmark': bool(row['is_benchmark']),
                'measurement_time': row['measurement_time']
            })

        return metrics

    def validate_constitutional_compliance(self, model_id: str) -> Dict[str, Any]:
        """
        Validate that a model meets constitutional requirements.

        Args:
            model_id: Model ID to validate

        Returns:
            Compliance validation results
        """
        model = self.get_model(model_id)
        if not model:
            return {'compliant': False, 'error': 'Model not found'}

        compliance_results = {
            'model_id': model_id,
            'compliant': True,
            'violations': [],
            'validation_time': datetime.now(timezone.utc).isoformat()
        }

        # Check explainability features
        cursor = self._get_connection().cursor()
        cursor.execute("""
            SELECT feature_type, feature_enabled FROM explainability_features
            WHERE model_id = ? AND feature_enabled = TRUE
        """, (model_id,))

        explainability_features = [row['feature_type'] for row in cursor.fetchall()]
        required_features = ['attention_visualization', 'decision_rationale']

        for feature in required_features:
            if feature not in explainability_features:
                compliance_results['compliant'] = False
                compliance_results['violations'].append(f"Missing explainability feature: {feature}")

        # Check performance metrics
        performance_metrics = self.get_performance_metrics(model_id)
        latency_metrics = [m for m in performance_metrics if m['metric_name'] == 'inference_latency_ms']

        if latency_metrics:
            latest_latency = latency_metrics[0]  # Most recent
            if latest_latency['metric_value'] > 100.0:  # Constitutional requirement
                compliance_results['compliant'] = False
                compliance_results['violations'].append(
                    f"Inference latency {latest_latency['metric_value']:.2f}ms exceeds 100ms requirement"
                )

        # Check validation results
        cursor.execute("""
            SELECT test_type, success, constitutional_compliance FROM validation_results
            WHERE model_id = ? ORDER BY test_time DESC
        """, (model_id,))

        validation_results = cursor.fetchall()
        required_tests = ['statistical_validation', 'performance_benchmark', 'code_coverage']

        for test in required_tests:
            test_results = [r for r in validation_results if test in r['test_type']]
            if not test_results or not test_results[0]['success']:
                compliance_results['compliant'] = False
                compliance_results['violations'].append(f"Failed required test: {test}")

        logger.info(f"Constitutional compliance validation for {model_id}: {'✅ PASS' if compliance_results['compliant'] else '❌ FAIL'}")
        return compliance_results

    def get_model_statistics(self) -> Dict[str, Any]:
        """
        Get overall model registry statistics.

        Returns:
            Dictionary with registry statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        stats = {}

        # Model counts by type
        cursor.execute("SELECT model_type, COUNT(*) as count FROM models GROUP BY model_type")
        stats['models_by_type'] = {row['model_type']: row['count'] for row in cursor.fetchall()}

        # Model counts by status
        cursor.execute("SELECT status, COUNT(*) as count FROM models GROUP BY status")
        stats['models_by_status'] = {row['status']: row['count'] for row in cursor.fetchall()}

        # Total models
        cursor.execute("SELECT COUNT(*) as total FROM models")
        stats['total_models'] = cursor.fetchone()['total']

        # Active models
        cursor.execute("SELECT COUNT(*) as active FROM models WHERE is_active = TRUE")
        stats['active_models'] = cursor.fetchone()['active']

        # Database size
        db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
        stats['database_size_bytes'] = db_size

        return stats

    def close(self) -> None:
        """Close database connection."""
        if hasattr(self._local, 'connection'):
            self._local.connection.close()


# Global model registry instance
_model_registry = None


def get_model_registry() -> ModelRegistry:
    """Get the global model registry instance."""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry


def setup_model_registry(db_path: Optional[str] = None) -> ModelRegistry:
    """
    Setup model registry with validation.

    Args:
        db_path: Optional custom database path

    Returns:
        Initialized model registry
    """
    registry = ModelRegistry(db_path or "data/model_registry.db")
    logger.info("Model registry initialized and ready")
    return registry