"""Database schema and migration scripts for parameter estimation functionality.

Extends the existing APGI Framework database with tables specific to parameter
estimation tasks, sessions, trials, and parameter estimates.
"""

import sqlite3
from pathlib import Path
from typing import Optional

from ..exceptions import APGIFrameworkError


class SchemaError(APGIFrameworkError):
    """Errors in database schema operations."""


class ParameterEstimationSchema:
    """Database schema manager for parameter estimation functionality.

    Handles creation, migration, and management of parameter estimation
    specific database tables and indices.
    """

    SCHEMA_VERSION = "1.0.0"

    # Whitelist of allowed table names for security
    ALLOWED_TABLES = {
        "trial_quality_metrics",
        "parameter_estimates",
        "oddball_trials",
        "heartbeat_trials",
        "detection_trials",
        "parameter_estimation_trials",
        "parameter_estimation_sessions",
    }

    def __init__(self, db_path: Path):
        """Initialize schema manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)

    def create_parameter_estimation_tables(self) -> None:
        """Create all parameter estimation specific tables."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Enable foreign key constraints
                conn.execute("PRAGMA foreign_keys = ON")

                # Create sessions table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS parameter_estimation_sessions (
                        session_id TEXT PRIMARY KEY,
                        participant_id TEXT NOT NULL,
                        session_date TEXT NOT NULL,
                        protocol_version TEXT NOT NULL DEFAULT '1.0.0',
                        completion_status TEXT NOT NULL DEFAULT 'in_progress',
                        total_duration_minutes REAL,
                        session_quality_score REAL DEFAULT 1.0,
                        researcher TEXT DEFAULT '',
                        notes TEXT DEFAULT '',
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        
                        CHECK (completion_status IN ('in_progress', 'completed', 'aborted')),
                        CHECK (session_quality_score >= 0.0 AND session_quality_score <= 1.0)
                    )
                """)

                # Create trials table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS parameter_estimation_trials (
                        trial_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        participant_id TEXT NOT NULL,
                        task_type TEXT NOT NULL,
                        trial_number INTEGER NOT NULL,
                        timestamp TEXT NOT NULL,
                        
                        -- Stimulus parameters
                        stimulus_modality TEXT NOT NULL,
                        stimulus_intensity REAL NOT NULL DEFAULT 0.0,
                        stimulus_parameters TEXT, -- JSON
                        
                        -- Behavioral response
                        response_time REAL,
                        detected BOOLEAN,
                        confidence REAL,
                        response_key TEXT,
                        reaction_time_valid BOOLEAN DEFAULT TRUE,
                        
                        -- Quality metrics
                        eeg_artifact_ratio REAL DEFAULT 0.0,
                        pupil_data_loss REAL DEFAULT 0.0,
                        cardiac_signal_quality REAL DEFAULT 1.0,
                        overall_quality_score REAL DEFAULT 1.0,
                        
                        -- Metadata
                        metadata TEXT, -- JSON
                        
                        FOREIGN KEY (session_id) REFERENCES parameter_estimation_sessions(session_id),
                        CHECK (task_type IN ('detection', 'heartbeat_detection', 'dual_modality_oddball')),
                        CHECK (stimulus_modality IN ('visual', 'auditory', 'interoceptive', 'exteroceptive')),
                        CHECK (confidence IS NULL OR (confidence >= 0.0 AND confidence <= 1.0)),
                        CHECK (eeg_artifact_ratio >= 0.0 AND eeg_artifact_ratio <= 1.0),
                        CHECK (pupil_data_loss >= 0.0 AND pupil_data_loss <= 1.0),
                        CHECK (cardiac_signal_quality >= 0.0 AND cardiac_signal_quality <= 1.0),
                        CHECK (overall_quality_score >= 0.0 AND overall_quality_score <= 1.0)
                    )
                """)

                # Create detection trials specific table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS detection_trials (
                        trial_id TEXT PRIMARY KEY,
                        gabor_orientation REAL,
                        tone_frequency REAL,
                        contrast_level REAL DEFAULT 0.0,
                        p3b_amplitude REAL,
                        p3b_latency REAL,
                        staircase_intensity REAL DEFAULT 0.0,
                        staircase_reversals INTEGER DEFAULT 0,
                        
                        FOREIGN KEY (trial_id) REFERENCES parameter_estimation_trials(trial_id),
                        CHECK (contrast_level >= 0.0 AND contrast_level <= 1.0)
                    )
                """)

                # Create heartbeat trials specific table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS heartbeat_trials (
                        trial_id TEXT PRIMARY KEY,
                        is_synchronous BOOLEAN NOT NULL,
                        tone_delay_ms REAL DEFAULT 0.0,
                        r_peak_timestamp TEXT,
                        heart_rate REAL DEFAULT 0.0,
                        rr_interval REAL DEFAULT 0.0,
                        hep_amplitude REAL,
                        interoceptive_p3b REAL,
                        pupil_baseline REAL,
                        pupil_dilation_peak REAL,
                        pupil_time_to_peak REAL,
                        
                        FOREIGN KEY (trial_id) REFERENCES parameter_estimation_trials(trial_id),
                        CHECK (heart_rate >= 0.0 AND heart_rate <= 200.0)
                    )
                """)

                # Create oddball trials specific table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS oddball_trials (
                        trial_id TEXT PRIMARY KEY,
                        is_deviant BOOLEAN NOT NULL,
                        deviant_type TEXT,
                        co2_puff_duration REAL,
                        co2_concentration REAL,
                        heartbeat_flash_delay REAL,
                        gabor_orientation_deviation REAL,
                        auditory_deviant_frequency REAL,
                        interoceptive_p3b REAL,
                        exteroceptive_p3b REAL,
                        p3b_ratio REAL,
                        interoceptive_precision REAL,
                        exteroceptive_precision REAL,
                        
                        FOREIGN KEY (trial_id) REFERENCES parameter_estimation_trials(trial_id),
                        CHECK (deviant_type IS NULL OR deviant_type IN ('interoceptive', 'exteroceptive'))
                    )
                """)

                # Create parameter estimates table
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS parameter_estimates (
                        estimate_id TEXT PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        participant_id TEXT NOT NULL,
                        estimation_timestamp TEXT NOT NULL,
                        
                        -- θ₀ (baseline ignition threshold)
                        theta0_mean REAL NOT NULL,
                        theta0_std REAL NOT NULL,
                        theta0_ci_lower REAL NOT NULL,
                        theta0_ci_upper REAL NOT NULL,
                        theta0_r_hat REAL,
                        theta0_ess INTEGER,
                        
                        -- Πᵢ (interoceptive precision)
                        pi_i_mean REAL NOT NULL,
                        pi_i_std REAL NOT NULL,
                        pi_i_ci_lower REAL NOT NULL,
                        pi_i_ci_upper REAL NOT NULL,
                        pi_i_r_hat REAL,
                        pi_i_ess INTEGER,
                        
                        -- β (somatic bias)
                        beta_mean REAL NOT NULL,
                        beta_std REAL NOT NULL,
                        beta_ci_lower REAL NOT NULL,
                        beta_ci_upper REAL NOT NULL,
                        beta_r_hat REAL,
                        beta_ess INTEGER,
                        
                        -- Model fit metrics
                        log_likelihood REAL DEFAULT 0.0,
                        aic REAL DEFAULT 0.0,
                        bic REAL DEFAULT 0.0,
                        waic REAL DEFAULT 0.0,
                        chains_converged BOOLEAN DEFAULT FALSE,
                        max_r_hat REAL DEFAULT 1.0,
                        min_effective_sample_size INTEGER DEFAULT 0,
                        loo_cv_score REAL,
                        
                        -- Reliability metrics
                        icc_theta0 REAL,
                        icc_pi_i REAL,
                        icc_beta REAL,
                        test_retest_interval_days INTEGER,
                        n_participants_reliability INTEGER,
                        
                        -- Task performance summaries
                        detection_accuracy REAL,
                        heartbeat_d_prime REAL,
                        oddball_discrimination REAL,
                        
                        -- Data quality
                        overall_data_quality REAL DEFAULT 1.0,
                        n_trials_detection INTEGER DEFAULT 0,
                        n_trials_heartbeat INTEGER DEFAULT 0,
                        n_trials_oddball INTEGER DEFAULT 0,
                        n_excluded_detection INTEGER DEFAULT 0,
                        n_excluded_heartbeat INTEGER DEFAULT 0,
                        n_excluded_oddball INTEGER DEFAULT 0,
                        
                        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        
                        FOREIGN KEY (session_id) REFERENCES parameter_estimation_sessions(session_id),
                        CHECK (theta0_std >= 0.0),
                        CHECK (pi_i_std >= 0.0),
                        CHECK (beta_std >= 0.0),
                        CHECK (theta0_ci_lower <= theta0_ci_upper),
                        CHECK (pi_i_ci_lower <= pi_i_ci_upper),
                        CHECK (beta_ci_lower <= beta_ci_upper),
                        CHECK (overall_data_quality >= 0.0 AND overall_data_quality <= 1.0),
                        CHECK (detection_accuracy IS NULL OR (detection_accuracy >= 0.0 AND detection_accuracy <= 1.0))
                    )
                """)

                # Create quality metrics table for detailed tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS trial_quality_metrics (
                        trial_id TEXT PRIMARY KEY,
                        electrode_impedances TEXT, -- JSON: electrode -> impedance
                        bad_channels TEXT, -- JSON: list of bad channel names
                        blink_rate REAL DEFAULT 0.0,
                        tracking_loss_episodes INTEGER DEFAULT 0,
                        r_peak_detection_confidence REAL DEFAULT 1.0,
                        heart_rate_variability REAL DEFAULT 0.0,
                        
                        FOREIGN KEY (trial_id) REFERENCES parameter_estimation_trials(trial_id),
                        CHECK (blink_rate >= 0.0),
                        CHECK (tracking_loss_episodes >= 0),
                        CHECK (r_peak_detection_confidence >= 0.0 AND r_peak_detection_confidence <= 1.0),
                        CHECK (heart_rate_variability >= 0.0)
                    )
                """)

                # Create indices for performance
                self._create_indices(conn)

                # Create schema version tracking
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS schema_versions (
                        component TEXT PRIMARY KEY,
                        version TEXT NOT NULL,
                        applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        description TEXT
                    )
                """)

                # Record schema version
                conn.execute(
                    """
                    INSERT OR REPLACE INTO schema_versions 
                    (component, version, description) 
                    VALUES (?, ?, ?)
                """,
                    (
                        "parameter_estimation",
                        self.SCHEMA_VERSION,
                        "Parameter estimation tables and indices",
                    ),
                )

                conn.commit()

        except Exception as e:
            raise SchemaError(f"Failed to create parameter estimation tables: {str(e)}")

    def _create_indices(self, conn: sqlite3.Connection) -> None:
        """Create database indices for performance optimization."""
        indices = [
            # Session indices
            "CREATE INDEX IF NOT EXISTS idx_sessions_participant ON parameter_estimation_sessions(participant_id)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_date ON parameter_estimation_sessions(session_date)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON parameter_estimation_sessions(completion_status)",
            # Trial indices
            "CREATE INDEX IF NOT EXISTS idx_trials_session ON parameter_estimation_trials(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_trials_participant ON parameter_estimation_trials(participant_id)",
            "CREATE INDEX IF NOT EXISTS idx_trials_task_type ON parameter_estimation_trials(task_type)",
            "CREATE INDEX IF NOT EXISTS idx_trials_timestamp ON parameter_estimation_trials(timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_trials_quality ON parameter_estimation_trials(overall_quality_score)",
            # Parameter estimate indices
            "CREATE INDEX IF NOT EXISTS idx_estimates_session ON parameter_estimates(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_estimates_participant ON parameter_estimates(participant_id)",
            "CREATE INDEX IF NOT EXISTS idx_estimates_timestamp ON parameter_estimates(estimation_timestamp)",
            # Composite indices for common queries
            "CREATE INDEX IF NOT EXISTS idx_trials_session_task ON parameter_estimation_trials(session_id, task_type)",
            "CREATE INDEX IF NOT EXISTS idx_trials_participant_task ON parameter_estimation_trials(participant_id, task_type)",
        ]

        for index_sql in indices:
            conn.execute(index_sql)

    def migrate_schema(self, target_version: Optional[str] = None) -> None:
        """Migrate database schema to target version.

        Args:
            target_version: Target schema version (latest if None)
        """
        if target_version is None:
            target_version = self.SCHEMA_VERSION

        try:
            current_version = self.get_schema_version()

            if current_version == target_version:
                return  # Already at target version

            # For now, we only support creating the initial schema
            # Future versions would implement incremental migrations
            if current_version is None:
                self.create_parameter_estimation_tables()
            else:
                # Implement version-specific migrations here
                pass

        except Exception as e:
            raise SchemaError(f"Failed to migrate schema to version {target_version}: {str(e)}")

    def get_schema_version(self) -> Optional[str]:
        """Get current schema version for parameter estimation component."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT version FROM schema_versions 
                    WHERE component = 'parameter_estimation'
                """)
                result = cursor.fetchone()
                return result[0] if result else None

        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return None
        except Exception as e:
            raise SchemaError(f"Failed to get schema version: {str(e)}")

    def validate_schema(self) -> bool:
        """Validate that all required tables and indices exist."""
        required_tables = [
            "parameter_estimation_sessions",
            "parameter_estimation_trials",
            "detection_trials",
            "heartbeat_trials",
            "oddball_trials",
            "parameter_estimates",
            "trial_quality_metrics",
            "schema_versions",
        ]

        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                existing_tables = {row[0] for row in cursor.fetchall()}

                missing_tables = set(required_tables) - existing_tables
                if missing_tables:
                    return False

                # Validate schema version
                version = self.get_schema_version()
                return version == self.SCHEMA_VERSION

        except Exception as e:
            raise SchemaError(f"Failed to validate schema: {str(e)}")

    def drop_parameter_estimation_tables(self) -> None:
        """Drop all parameter estimation tables (for testing/cleanup)."""
        tables_to_drop = [
            "trial_quality_metrics",
            "parameter_estimates",
            "oddball_trials",
            "heartbeat_trials",
            "detection_trials",
            "parameter_estimation_trials",
            "parameter_estimation_sessions",
        ]

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("PRAGMA foreign_keys = OFF")

                for table in tables_to_drop:
                    if table not in self.ALLOWED_TABLES:
                        raise SchemaError(f"Illegal table name: {table}")
                    _sql = f"DROP TABLE IF EXISTS {table}"  # nosemgrep: sql-injection-risk - table name validated above
                    conn.execute(_sql)

                # Remove schema version record
                conn.execute("""
                    DELETE FROM schema_versions 
                    WHERE component = 'parameter_estimation'
                """)

                conn.execute("PRAGMA foreign_keys = ON")
                conn.commit()

        except Exception as e:
            raise SchemaError(f"Failed to drop parameter estimation tables: {str(e)}")


def create_parameter_estimation_schema(db_path: Path) -> ParameterEstimationSchema:
    """Factory function to create and initialize parameter estimation schema.

    Args:
        db_path: Path to SQLite database file

    Returns:
        ParameterEstimationSchema: Initialized schema manager
    """
    schema = ParameterEstimationSchema(db_path)
    schema.migrate_schema()
    return schema
