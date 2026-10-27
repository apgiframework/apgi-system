"""Data Access Object (DAO) for parameter estimation functionality.

Provides CRUD operations for parameter estimation sessions, trials, and estimates
with the SQLite database backend.
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union

from ..exceptions import APGIFrameworkError
from .parameter_estimation_models import (
    BehavioralResponse,
    DetectionTrialResult,
    HeartbeatTrialResult,
    ModelFitMetrics,
    OddballTrialResult,
    ParameterDistribution,
    ParameterEstimates,
    QualityMetrics,
    ReliabilityMetrics,
    SessionData,
    StimulusModality,
    TrialData,
)
from .parameter_estimation_schema import ParameterEstimationSchema


class ParameterEstimationDAOError(APGIFrameworkError):
    """Errors in parameter estimation data access operations."""


class ParameterEstimationDAO:
    """Data Access Object for parameter estimation functionality.

    Provides high-level CRUD operations for sessions, trials, and parameter
    estimates with automatic schema management and data validation.
    """

    def __init__(self, db_path: Path):
        """Initialize DAO with database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.schema = ParameterEstimationSchema(db_path)

        # Ensure schema exists
        if not self.schema.validate_schema():
            self.schema.migrate_schema()

    # Session operations

    def create_session(self, session_data: SessionData) -> str:
        """Create a new parameter estimation session.

        Args:
            session_data: SessionData object to store

        Returns:
            str: Session ID of created session
        """
        try:
            if not session_data.validate():
                raise ParameterEstimationDAOError("Invalid session data")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO parameter_estimation_sessions (
                        session_id, participant_id, session_date, protocol_version,
                        completion_status, total_duration_minutes, session_quality_score,
                        researcher, notes, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        session_data.session_id,
                        session_data.participant_id,
                        session_data.session_date.isoformat(),
                        session_data.protocol_version,
                        session_data.completion_status,
                        session_data.total_duration_minutes,
                        session_data.session_quality_score,
                        session_data.researcher,
                        session_data.notes,
                        datetime.now().isoformat(),
                        datetime.now().isoformat(),
                    ),
                )
                conn.commit()

            return session_data.session_id

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to create session: {str(e)}")

    def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session by ID.

        Args:
            session_id: Session identifier

        Returns:
            SessionData: Session data or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM parameter_estimation_sessions 
                    WHERE session_id = ?
                """,
                    (session_id,),
                )
                row = cursor.fetchone()

                if not row:
                    return None

                # Load trials
                detection_trials = self.get_detection_trials(session_id)
                heartbeat_trials = self.get_heartbeat_trials(session_id)
                oddball_trials = self.get_oddball_trials(session_id)

                # Load parameter estimates if available
                parameter_estimates = self.get_parameter_estimates(session_id)

                session_data = SessionData(
                    session_id=row["session_id"],
                    participant_id=row["participant_id"],
                    session_date=datetime.fromisoformat(row["session_date"]),
                    protocol_version=row["protocol_version"],
                    completion_status=row["completion_status"],
                    total_duration_minutes=row["total_duration_minutes"],
                    detection_trials=detection_trials,
                    heartbeat_trials=heartbeat_trials,
                    oddball_trials=oddball_trials,
                    parameter_estimates=parameter_estimates,
                    session_quality_score=row["session_quality_score"],
                    researcher=row["researcher"],
                    notes=row["notes"],
                )

                return session_data

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to get session {session_id}: {str(e)}")

    def update_session(self, session_data: SessionData) -> None:
        """Update existing session.

        Args:
            session_data: Updated session data
        """
        try:
            if not session_data.validate():
                raise ParameterEstimationDAOError("Invalid session data")

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    UPDATE parameter_estimation_sessions SET
                        completion_status = ?, total_duration_minutes = ?,
                        session_quality_score = ?, researcher = ?, notes = ?,
                        updated_at = ?
                    WHERE session_id = ?
                """,
                    (
                        session_data.completion_status,
                        session_data.total_duration_minutes,
                        session_data.session_quality_score,
                        session_data.researcher,
                        session_data.notes,
                        datetime.now().isoformat(),
                        session_data.session_id,
                    ),
                )
                conn.commit()

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to update session: {str(e)}")

    def list_sessions(
        self, participant_id: Optional[str] = None, limit: Optional[int] = None
    ) -> List[str]:
        """List session IDs, optionally filtered by participant.

        Args:
            participant_id: Filter by participant ID
            limit: Maximum number of sessions to return

        Returns:
            List[str]: List of session IDs
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = "SELECT session_id FROM parameter_estimation_sessions"
                params: List[Union[str, int]] = []

                if participant_id:
                    query += " WHERE participant_id = ?"
                    params.append(participant_id)

                query += " ORDER BY session_date DESC"

                if limit:
                    query += " LIMIT ?"
                    params.append(limit)

                cursor = conn.execute(query, params)
                return [row[0] for row in cursor.fetchall()]

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to list sessions: {str(e)}")

    # Trial operations

    def create_trial(self, trial_data: TrialData) -> str:
        """Create a new trial record.

        Args:
            trial_data: Trial data to store

        Returns:
            str: Trial ID of created trial
        """
        try:
            if not trial_data.validate():
                raise ParameterEstimationDAOError("Invalid trial data")

            with sqlite3.connect(self.db_path) as conn:
                # Insert base trial data
                conn.execute(
                    """
                    INSERT INTO parameter_estimation_trials (
                        trial_id, session_id, participant_id, task_type, trial_number,
                        timestamp, stimulus_modality, stimulus_intensity, stimulus_parameters,
                        response_time, detected, confidence, response_key, reaction_time_valid,
                        eeg_artifact_ratio, pupil_data_loss, cardiac_signal_quality,
                        overall_quality_score, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        trial_data.trial_id,
                        trial_data.session_id,
                        trial_data.participant_id,
                        trial_data.task_type.value,
                        trial_data.trial_number,
                        trial_data.timestamp.isoformat(),
                        trial_data.stimulus_modality.value,
                        trial_data.stimulus_intensity,
                        json.dumps(trial_data.stimulus_parameters),
                        (
                            trial_data.behavioral_response.response_time
                            if trial_data.behavioral_response
                            else None
                        ),
                        (
                            trial_data.behavioral_response.detected
                            if trial_data.behavioral_response
                            else None
                        ),
                        (
                            trial_data.behavioral_response.confidence
                            if trial_data.behavioral_response
                            else None
                        ),
                        (
                            trial_data.behavioral_response.response_key
                            if trial_data.behavioral_response
                            else None
                        ),
                        (
                            trial_data.behavioral_response.reaction_time_valid
                            if trial_data.behavioral_response
                            else None
                        ),
                        trial_data.quality_metrics.eeg_artifact_ratio,
                        trial_data.quality_metrics.pupil_data_loss,
                        trial_data.quality_metrics.cardiac_signal_quality,
                        trial_data.quality_metrics.overall_quality_score,
                        json.dumps(trial_data.metadata),
                    ),
                )

                # Insert task-specific data
                if isinstance(trial_data, DetectionTrialResult):
                    self._create_detection_trial(conn, trial_data)
                elif isinstance(trial_data, HeartbeatTrialResult):
                    self._create_heartbeat_trial(conn, trial_data)
                elif isinstance(trial_data, OddballTrialResult):
                    self._create_oddball_trial(conn, trial_data)

                # Insert quality metrics
                self._create_quality_metrics(conn, trial_data)

                conn.commit()

            return trial_data.trial_id

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to create trial: {str(e)}")

    def create_trials_batch(self, trials: List[TrialData]) -> List[str]:
        """Create multiple trial records in a single transaction.

        Args:
            trials: List of trial data objects

        Returns:
            List[str]: List of created trial IDs
        """
        if not trials:
            return []

        try:
            trial_ids = []
            with sqlite3.connect(self.db_path) as conn:
                # Prepare batch data for base trials
                base_trials_data = []
                detection_data = []
                heartbeat_data = []
                oddball_data = []
                quality_data = []

                for trial in trials:
                    if not trial.validate():
                        raise ParameterEstimationDAOError(
                            f"Invalid trial data for ID {trial.trial_id}"
                        )

                    base_trials_data.append(
                        (
                            trial.trial_id,
                            trial.session_id,
                            trial.participant_id,
                            trial.task_type.value,
                            trial.trial_number,
                            trial.timestamp.isoformat(),
                            trial.stimulus_modality.value,
                            trial.stimulus_intensity,
                            json.dumps(trial.stimulus_parameters),
                            (
                                trial.behavioral_response.response_time
                                if trial.behavioral_response
                                else None
                            ),
                            (
                                trial.behavioral_response.detected
                                if trial.behavioral_response
                                else None
                            ),
                            (
                                trial.behavioral_response.confidence
                                if trial.behavioral_response
                                else None
                            ),
                            (
                                trial.behavioral_response.response_key
                                if trial.behavioral_response
                                else None
                            ),
                            (
                                trial.behavioral_response.reaction_time_valid
                                if trial.behavioral_response
                                else None
                            ),
                            trial.quality_metrics.eeg_artifact_ratio,
                            trial.quality_metrics.pupil_data_loss,
                            trial.quality_metrics.cardiac_signal_quality,
                            trial.quality_metrics.overall_quality_score,
                            json.dumps(trial.metadata),
                        )
                    )

                    if isinstance(trial, DetectionTrialResult):
                        detection_data.append(
                            (
                                trial.trial_id,
                                trial.gabor_orientation,
                                trial.tone_frequency,
                                trial.contrast_level,
                                trial.p3b_amplitude,
                                trial.p3b_latency,
                                trial.staircase_intensity,
                                trial.staircase_reversals,
                            )
                        )
                    elif isinstance(trial, HeartbeatTrialResult):
                        heartbeat_data.append(
                            (
                                trial.trial_id,
                                trial.is_synchronous,
                                trial.tone_delay_ms,
                                (
                                    trial.r_peak_timestamp.isoformat()
                                    if trial.r_peak_timestamp
                                    else None
                                ),
                                trial.heart_rate,
                                trial.rr_interval,
                                trial.hep_amplitude,
                                trial.interoceptive_p3b,
                                trial.pupil_baseline,
                                trial.pupil_dilation_peak,
                                trial.pupil_time_to_peak,
                            )
                        )
                    elif isinstance(trial, OddballTrialResult):
                        oddball_data.append(
                            (
                                trial.trial_id,
                                trial.is_deviant,
                                trial.deviant_type,
                                trial.co2_puff_duration,
                                trial.co2_concentration,
                                trial.heartbeat_flash_delay,
                                trial.gabor_orientation_deviation,
                                trial.auditory_deviant_frequency,
                                trial.interoceptive_p3b,
                                trial.exteroceptive_p3b,
                                trial.p3b_ratio,
                                trial.interoceptive_precision,
                                trial.exteroceptive_precision,
                            )
                        )

                    quality_data.append(
                        (
                            trial.trial_id,
                            json.dumps(trial.quality_metrics.electrode_impedances),
                            json.dumps(trial.quality_metrics.bad_channels),
                            trial.quality_metrics.blink_rate,
                            trial.quality_metrics.tracking_loss_episodes,
                            trial.quality_metrics.r_peak_detection_confidence,
                            trial.quality_metrics.heart_rate_variability,
                        )
                    )

                    trial_ids.append(trial.trial_id)

                # Batch inserts
                conn.executemany(
                    "INSERT INTO parameter_estimation_trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    base_trials_data,
                )
                if detection_data:
                    conn.executemany(
                        "INSERT INTO detection_trials VALUES (?,?,?,?,?,?,?,?)", detection_data
                    )
                if heartbeat_data:
                    conn.executemany(
                        "INSERT INTO heartbeat_trials VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        heartbeat_data,
                    )
                if oddball_data:
                    conn.executemany(
                        "INSERT INTO oddball_trials VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        oddball_data,
                    )

                conn.executemany(
                    "INSERT INTO trial_quality_metrics VALUES (?,?,?,?,?,?,?)", quality_data
                )

                conn.commit()
            return trial_ids
        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to create trials batch: {str(e)}")

    def _create_detection_trial(
        self, conn: sqlite3.Connection, trial_data: DetectionTrialResult
    ) -> None:
        """Create detection-specific trial data."""
        conn.execute(
            """
            INSERT INTO detection_trials (
                trial_id, gabor_orientation, tone_frequency, contrast_level,
                p3b_amplitude, p3b_latency, staircase_intensity, staircase_reversals
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trial_data.trial_id,
                trial_data.gabor_orientation,
                trial_data.tone_frequency,
                trial_data.contrast_level,
                trial_data.p3b_amplitude,
                trial_data.p3b_latency,
                trial_data.staircase_intensity,
                trial_data.staircase_reversals,
            ),
        )

    def _create_heartbeat_trial(
        self, conn: sqlite3.Connection, trial_data: HeartbeatTrialResult
    ) -> None:
        """Create heartbeat-specific trial data."""
        conn.execute(
            """
            INSERT INTO heartbeat_trials (
                trial_id, is_synchronous, tone_delay_ms, r_peak_timestamp,
                heart_rate, rr_interval, hep_amplitude, interoceptive_p3b,
                pupil_baseline, pupil_dilation_peak, pupil_time_to_peak
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trial_data.trial_id,
                trial_data.is_synchronous,
                trial_data.tone_delay_ms,
                (trial_data.r_peak_timestamp.isoformat() if trial_data.r_peak_timestamp else None),
                trial_data.heart_rate,
                trial_data.rr_interval,
                trial_data.hep_amplitude,
                trial_data.interoceptive_p3b,
                trial_data.pupil_baseline,
                trial_data.pupil_dilation_peak,
                trial_data.pupil_time_to_peak,
            ),
        )

    def _create_oddball_trial(
        self, conn: sqlite3.Connection, trial_data: OddballTrialResult
    ) -> None:
        """Create oddball-specific trial data."""
        conn.execute(
            """
            INSERT INTO oddball_trials (
                trial_id, is_deviant, deviant_type, co2_puff_duration,
                co2_concentration, heartbeat_flash_delay, gabor_orientation_deviation,
                auditory_deviant_frequency, interoceptive_p3b, exteroceptive_p3b,
                p3b_ratio, interoceptive_precision, exteroceptive_precision
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trial_data.trial_id,
                trial_data.is_deviant,
                trial_data.deviant_type,
                trial_data.co2_puff_duration,
                trial_data.co2_concentration,
                trial_data.heartbeat_flash_delay,
                trial_data.gabor_orientation_deviation,
                trial_data.auditory_deviant_frequency,
                trial_data.interoceptive_p3b,
                trial_data.exteroceptive_p3b,
                trial_data.p3b_ratio,
                trial_data.interoceptive_precision,
                trial_data.exteroceptive_precision,
            ),
        )

    def _create_quality_metrics(self, conn: sqlite3.Connection, trial_data: TrialData) -> None:
        """Create detailed quality metrics record."""
        conn.execute(
            """
            INSERT INTO trial_quality_metrics (
                trial_id, electrode_impedances, bad_channels, blink_rate,
                tracking_loss_episodes, r_peak_detection_confidence,
                heart_rate_variability
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                trial_data.trial_id,
                json.dumps(trial_data.quality_metrics.electrode_impedances),
                json.dumps(trial_data.quality_metrics.bad_channels),
                trial_data.quality_metrics.blink_rate,
                trial_data.quality_metrics.tracking_loss_episodes,
                trial_data.quality_metrics.r_peak_detection_confidence,
                trial_data.quality_metrics.heart_rate_variability,
            ),
        )

    def get_detection_trials(self, session_id: str) -> List[DetectionTrialResult]:
        """Get all detection trials for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT t.*, d.* FROM parameter_estimation_trials t
                    JOIN detection_trials d ON t.trial_id = d.trial_id
                    WHERE t.session_id = ? AND t.task_type = 'detection'
                    ORDER BY t.trial_number
                """,
                    (session_id,),
                )

                trials = []
                for row in cursor.fetchall():
                    trial = self._row_to_detection_trial(row)
                    trials.append(trial)

                return trials

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to get detection trials: {str(e)}")

    def get_heartbeat_trials(self, session_id: str) -> List[HeartbeatTrialResult]:
        """Get all heartbeat trials for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT t.*, h.* FROM parameter_estimation_trials t
                    JOIN heartbeat_trials h ON t.trial_id = h.trial_id
                    WHERE t.session_id = ? AND t.task_type = 'heartbeat_detection'
                    ORDER BY t.trial_number
                """,
                    (session_id,),
                )

                trials = []
                for row in cursor.fetchall():
                    trial = self._row_to_heartbeat_trial(row)
                    trials.append(trial)

                return trials

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to get heartbeat trials: {str(e)}")

    def get_oddball_trials(self, session_id: str) -> List[OddballTrialResult]:
        """Get all oddball trials for a session."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT t.*, o.* FROM parameter_estimation_trials t
                    JOIN oddball_trials o ON t.trial_id = o.trial_id
                    WHERE t.session_id = ? AND t.task_type = 'dual_modality_oddball'
                    ORDER BY t.trial_number
                """,
                    (session_id,),
                )

                trials = []
                for row in cursor.fetchall():
                    trial = self._row_to_oddball_trial(row)
                    trials.append(trial)

                return trials

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to get oddball trials: {str(e)}")

    def _row_to_detection_trial(self, row: sqlite3.Row) -> DetectionTrialResult:
        """Convert database row to DetectionTrialResult."""
        behavioral_response = None
        if row["response_time"] is not None:
            behavioral_response = BehavioralResponse(
                response_time=row["response_time"],
                detected=bool(row["detected"]),
                confidence=row["confidence"],
                response_key=row["response_key"] or "",
                reaction_time_valid=bool(row["reaction_time_valid"]),
            )

        quality_metrics = QualityMetrics(
            eeg_artifact_ratio=row["eeg_artifact_ratio"],
            pupil_data_loss=row["pupil_data_loss"],
            cardiac_signal_quality=row["cardiac_signal_quality"],
            overall_quality_score=row["overall_quality_score"],
        )

        return DetectionTrialResult(
            trial_id=row["trial_id"],
            participant_id=row["participant_id"],
            session_id=row["session_id"],
            trial_number=row["trial_number"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            stimulus_parameters=json.loads(row["stimulus_parameters"] or "{}"),
            stimulus_modality=StimulusModality(row["stimulus_modality"]),
            stimulus_intensity=row["stimulus_intensity"],
            behavioral_response=behavioral_response,
            quality_metrics=quality_metrics,
            metadata=json.loads(row["metadata"] or "{}"),
            gabor_orientation=row["gabor_orientation"],
            tone_frequency=row["tone_frequency"],
            contrast_level=row["contrast_level"],
            p3b_amplitude=row["p3b_amplitude"],
            p3b_latency=row["p3b_latency"],
            staircase_intensity=row["staircase_intensity"],
            staircase_reversals=row["staircase_reversals"],
        )

    def _row_to_heartbeat_trial(self, row: sqlite3.Row) -> HeartbeatTrialResult:
        """Convert database row to HeartbeatTrialResult."""
        behavioral_response = None
        if row["response_time"] is not None:
            behavioral_response = BehavioralResponse(
                response_time=row["response_time"],
                detected=bool(row["detected"]),
                confidence=row["confidence"],
                response_key=row["response_key"] or "",
                reaction_time_valid=bool(row["reaction_time_valid"]),
            )

        quality_metrics = QualityMetrics(
            eeg_artifact_ratio=row["eeg_artifact_ratio"],
            pupil_data_loss=row["pupil_data_loss"],
            cardiac_signal_quality=row["cardiac_signal_quality"],
            overall_quality_score=row["overall_quality_score"],
        )

        return HeartbeatTrialResult(
            trial_id=row["trial_id"],
            participant_id=row["participant_id"],
            session_id=row["session_id"],
            trial_number=row["trial_number"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            stimulus_parameters=json.loads(row["stimulus_parameters"] or "{}"),
            stimulus_modality=StimulusModality(row["stimulus_modality"]),
            stimulus_intensity=row["stimulus_intensity"],
            behavioral_response=behavioral_response,
            quality_metrics=quality_metrics,
            metadata=json.loads(row["metadata"] or "{}"),
            is_synchronous=bool(row["is_synchronous"]),
            tone_delay_ms=row["tone_delay_ms"],
            r_peak_timestamp=(
                datetime.fromisoformat(row["r_peak_timestamp"]) if row["r_peak_timestamp"] else None
            ),
            heart_rate=row["heart_rate"],
            rr_interval=row["rr_interval"],
            hep_amplitude=row["hep_amplitude"],
            interoceptive_p3b=row["interoceptive_p3b"],
            pupil_baseline=row["pupil_baseline"],
            pupil_dilation_peak=row["pupil_dilation_peak"],
            pupil_time_to_peak=row["pupil_time_to_peak"],
        )

    def _row_to_oddball_trial(self, row: sqlite3.Row) -> OddballTrialResult:
        """Convert database row to OddballTrialResult."""
        behavioral_response = None
        if row["response_time"] is not None:
            behavioral_response = BehavioralResponse(
                response_time=row["response_time"],
                detected=bool(row["detected"]),
                confidence=row["confidence"],
                response_key=row["response_key"] or "",
                reaction_time_valid=bool(row["reaction_time_valid"]),
            )

        quality_metrics = QualityMetrics(
            eeg_artifact_ratio=row["eeg_artifact_ratio"],
            pupil_data_loss=row["pupil_data_loss"],
            cardiac_signal_quality=row["cardiac_signal_quality"],
            overall_quality_score=row["overall_quality_score"],
        )

        return OddballTrialResult(
            trial_id=row["trial_id"],
            participant_id=row["participant_id"],
            session_id=row["session_id"],
            trial_number=row["trial_number"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            stimulus_parameters=json.loads(row["stimulus_parameters"] or "{}"),
            stimulus_modality=StimulusModality(row["stimulus_modality"]),
            stimulus_intensity=row["stimulus_intensity"],
            behavioral_response=behavioral_response,
            quality_metrics=quality_metrics,
            metadata=json.loads(row["metadata"] or "{}"),
            is_deviant=bool(row["is_deviant"]),
            deviant_type=row["deviant_type"],
            co2_puff_duration=row["co2_puff_duration"],
            co2_concentration=row["co2_concentration"],
            heartbeat_flash_delay=row["heartbeat_flash_delay"],
            gabor_orientation_deviation=row["gabor_orientation_deviation"],
            auditory_deviant_frequency=row["auditory_deviant_frequency"],
            interoceptive_p3b=row["interoceptive_p3b"],
            exteroceptive_p3b=row["exteroceptive_p3b"],
            p3b_ratio=row["p3b_ratio"],
            interoceptive_precision=row["interoceptive_precision"],
            exteroceptive_precision=row["exteroceptive_precision"],
        )

    # Parameter estimation operations

    def create_parameter_estimates(self, estimates: ParameterEstimates) -> str:
        """Store parameter estimates for a session.

        Args:
            estimates: ParameterEstimates to store

        Returns:
            str: Estimate ID
        """
        try:
            if not estimates.validate():
                raise ParameterEstimationDAOError("Invalid parameter estimates")

            estimate_id = str(uuid.uuid4())

            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO parameter_estimates (
                        estimate_id, session_id, participant_id, estimation_timestamp,
                        theta0_mean, theta0_std, theta0_ci_lower, theta0_ci_upper,
                        theta0_r_hat, theta0_ess,
                        pi_i_mean, pi_i_std, pi_i_ci_lower, pi_i_ci_upper,
                        pi_i_r_hat, pi_i_ess,
                        beta_mean, beta_std, beta_ci_lower, beta_ci_upper,
                        beta_r_hat, beta_ess,
                        log_likelihood, aic, bic, waic, chains_converged,
                        max_r_hat, min_effective_sample_size, loo_cv_score,
                        icc_theta0, icc_pi_i, icc_beta, test_retest_interval_days,
                        n_participants_reliability, detection_accuracy,
                        heartbeat_d_prime, oddball_discrimination,
                        overall_data_quality, n_trials_detection, n_trials_heartbeat,
                        n_trials_oddball, n_excluded_detection, n_excluded_heartbeat,
                        n_excluded_oddball
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        estimate_id,
                        estimates.session_id,
                        estimates.participant_id,
                        estimates.estimation_timestamp.isoformat(),
                        estimates.theta0.mean,
                        estimates.theta0.std,
                        estimates.theta0.credible_interval_95[0],
                        estimates.theta0.credible_interval_95[1],
                        estimates.theta0.r_hat,
                        estimates.theta0.effective_sample_size,
                        estimates.pi_i.mean,
                        estimates.pi_i.std,
                        estimates.pi_i.credible_interval_95[0],
                        estimates.pi_i.credible_interval_95[1],
                        estimates.pi_i.r_hat,
                        estimates.pi_i.effective_sample_size,
                        estimates.beta.mean,
                        estimates.beta.std,
                        estimates.beta.credible_interval_95[0],
                        estimates.beta.credible_interval_95[1],
                        estimates.beta.r_hat,
                        estimates.beta.effective_sample_size,
                        estimates.model_fit_metrics.log_likelihood,
                        estimates.model_fit_metrics.aic,
                        estimates.model_fit_metrics.bic,
                        estimates.model_fit_metrics.waic,
                        estimates.model_fit_metrics.chains_converged,
                        estimates.model_fit_metrics.max_r_hat,
                        estimates.model_fit_metrics.min_effective_sample_size,
                        estimates.model_fit_metrics.loo_cv_score,
                        estimates.reliability_metrics.icc_theta0,
                        estimates.reliability_metrics.icc_pi_i,
                        estimates.reliability_metrics.icc_beta,
                        estimates.reliability_metrics.test_retest_interval_days,
                        estimates.reliability_metrics.n_participants_reliability,
                        estimates.detection_accuracy,
                        estimates.heartbeat_d_prime,
                        estimates.oddball_discrimination,
                        estimates.overall_data_quality,
                        estimates.n_trials_used.get("detection", 0),
                        estimates.n_trials_used.get("heartbeat_detection", 0),
                        estimates.n_trials_used.get("dual_modality_oddball", 0),
                        estimates.n_trials_excluded.get("detection", 0),
                        estimates.n_trials_excluded.get("heartbeat_detection", 0),
                        estimates.n_trials_excluded.get("dual_modality_oddball", 0),
                    ),
                )
                conn.commit()

            return estimate_id

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to create parameter estimates: {str(e)}")

    def get_parameter_estimates(self, session_id: str) -> Optional[ParameterEstimates]:
        """Get parameter estimates for a session.

        Args:
            session_id: Session identifier

        Returns:
            ParameterEstimates: Parameter estimates or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    """
                    SELECT * FROM parameter_estimates WHERE session_id = ?
                """,
                    (session_id,),
                )
                row = cursor.fetchone()

                if not row:
                    return None

                return self._row_to_parameter_estimates(row)

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to get parameter estimates: {str(e)}")

    def _row_to_parameter_estimates(self, row: sqlite3.Row) -> ParameterEstimates:
        """Convert database row to ParameterEstimates."""
        theta0 = ParameterDistribution(
            mean=row["theta0_mean"],
            std=row["theta0_std"],
            credible_interval_95=(row["theta0_ci_lower"], row["theta0_ci_upper"]),
            r_hat=row["theta0_r_hat"],
            effective_sample_size=row["theta0_ess"],
        )

        pi_i = ParameterDistribution(
            mean=row["pi_i_mean"],
            std=row["pi_i_std"],
            credible_interval_95=(row["pi_i_ci_lower"], row["pi_i_ci_upper"]),
            r_hat=row["pi_i_r_hat"],
            effective_sample_size=row["pi_i_ess"],
        )

        beta = ParameterDistribution(
            mean=row["beta_mean"],
            std=row["beta_std"],
            credible_interval_95=(row["beta_ci_lower"], row["beta_ci_upper"]),
            r_hat=row["beta_r_hat"],
            effective_sample_size=row["beta_ess"],
        )

        model_fit_metrics = ModelFitMetrics(
            log_likelihood=row["log_likelihood"],
            aic=row["aic"],
            bic=row["bic"],
            waic=row["waic"],
            chains_converged=bool(row["chains_converged"]),
            max_r_hat=row["max_r_hat"],
            min_effective_sample_size=row["min_effective_sample_size"],
            loo_cv_score=row["loo_cv_score"],
        )

        reliability_metrics = ReliabilityMetrics(
            icc_theta0=row["icc_theta0"],
            icc_pi_i=row["icc_pi_i"],
            icc_beta=row["icc_beta"],
            test_retest_interval_days=row["test_retest_interval_days"],
            n_participants_reliability=row["n_participants_reliability"],
        )

        n_trials_used = {
            "detection": row["n_trials_detection"],
            "heartbeat_detection": row["n_trials_heartbeat"],
            "dual_modality_oddball": row["n_trials_oddball"],
        }

        n_trials_excluded = {
            "detection": row["n_excluded_detection"],
            "heartbeat_detection": row["n_excluded_heartbeat"],
            "dual_modality_oddball": row["n_excluded_oddball"],
        }

        return ParameterEstimates(
            participant_id=row["participant_id"],
            session_id=row["session_id"],
            estimation_timestamp=datetime.fromisoformat(row["estimation_timestamp"]),
            theta0=theta0,
            pi_i=pi_i,
            beta=beta,
            model_fit_metrics=model_fit_metrics,
            reliability_metrics=reliability_metrics,
            detection_accuracy=row["detection_accuracy"],
            heartbeat_d_prime=row["heartbeat_d_prime"],
            oddball_discrimination=row["oddball_discrimination"],
            overall_data_quality=row["overall_data_quality"],
            n_trials_used=n_trials_used,
            n_trials_excluded=n_trials_excluded,
        )

    def delete_session(self, session_id: str) -> None:
        """Delete a session and all associated data.

        Args:
            session_id: Session identifier
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Delete in reverse dependency order
                conn.execute(
                    "DELETE FROM parameter_estimates WHERE session_id = ?",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM trial_quality_metrics WHERE trial_id IN (SELECT trial_id FROM parameter_estimation_trials WHERE session_id = ?)",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM detection_trials WHERE trial_id IN (SELECT trial_id FROM parameter_estimation_trials WHERE session_id = ?)",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM heartbeat_trials WHERE trial_id IN (SELECT trial_id FROM parameter_estimation_trials WHERE session_id = ?)",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM oddball_trials WHERE trial_id IN (SELECT trial_id FROM parameter_estimation_trials WHERE session_id = ?)",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM parameter_estimation_trials WHERE session_id = ?",
                    (session_id,),
                )
                conn.execute(
                    "DELETE FROM parameter_estimation_sessions WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()

        except Exception as e:
            raise ParameterEstimationDAOError(f"Failed to delete session: {str(e)}")


def create_parameter_estimation_dao(db_path: Path) -> ParameterEstimationDAO:
    """Factory function to create parameter estimation DAO.

    Args:
        db_path: Path to SQLite database file

    Returns:
        ParameterEstimationDAO: Initialized DAO
    """
    return ParameterEstimationDAO(db_path)
