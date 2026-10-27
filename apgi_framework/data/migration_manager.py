"""Database migration manager for the APGI Framework.

Handles schema migrations and upgrades for both core framework
and parameter estimation functionality.
"""

import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from ..exceptions import APGIFrameworkError
from .parameter_estimation_schema import ParameterEstimationSchema


class MigrationError(APGIFrameworkError):
    """Errors in database migration operations."""


class MigrationManager:
    """Manages database schema migrations for the APGI Framework.

    Coordinates migrations between core framework tables and
    parameter estimation extensions.
    """

    def __init__(self, db_path: Path):
        """Initialize migration manager.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.parameter_estimation_schema = ParameterEstimationSchema(db_path)

    def _validate_table_name(self, table_name: str) -> bool:
        """Validate table name to prevent SQL injection.

        Only allows alphanumeric characters, underscores, and hyphens.
        Must start with a letter or underscore.
        """
        if not table_name or not isinstance(table_name, str):
            return False

        # Table name must match valid identifier pattern
        # Start with letter/underscore, followed by letters/numbers/underscores/hyphens
        pattern = r"^[a-zA-Z_][a-zA-Z0-9_-]*$"
        return bool(re.match(pattern, table_name)) and len(table_name) <= 128

    def get_migration_status(self) -> Dict[str, Optional[str]]:
        """Get current migration status for all components.

        Returns:
            Dict[str, Optional[str]]: Component -> version mapping
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if schema_versions table exists
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='schema_versions'
                """)

                if not cursor.fetchone():
                    return {"core": None, "parameter_estimation": None}

                # Get versions for all components
                cursor = conn.execute("SELECT component, version FROM schema_versions")
                versions = dict(cursor.fetchall())

                return {
                    "core": versions.get("core"),
                    "parameter_estimation": versions.get("parameter_estimation"),
                }

        except Exception as e:
            raise MigrationError(f"Failed to get migration status: {str(e)}")

    def migrate_all(self) -> None:
        """Migrate all components to latest versions."""
        try:
            # Ensure core tables exist (from existing persistence layer)
            self._ensure_core_tables()

            # Migrate parameter estimation schema
            self.parameter_estimation_schema.migrate_schema()

        except Exception as e:
            raise MigrationError(f"Failed to migrate all components: {str(e)}")

    def _ensure_core_tables(self) -> None:
        """Ensure core framework tables exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Check if experiments table exists (from persistence layer)
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='experiments'
                """)

                if cursor.fetchone():
                    # Core tables already exist, record version if not already recorded
                    cursor = conn.execute("""
                        SELECT version FROM schema_versions 
                        WHERE component = 'core'
                    """)

                    if not cursor.fetchone():
                        # Create schema_versions table if it doesn't exist
                        conn.execute("""
                            CREATE TABLE IF NOT EXISTS schema_versions (
                                component TEXT PRIMARY KEY,
                                version TEXT NOT NULL,
                                applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                                description TEXT
                            )
                        """)

                        # Record core version
                        conn.execute(
                            """
                            INSERT OR REPLACE INTO schema_versions 
                            (component, version, description) 
                            VALUES (?, ?, ?)
                        """,
                            ("core", "1.0.0", "Core APGI Framework tables"),
                        )

                        conn.commit()

        except Exception as e:
            raise MigrationError(f"Failed to ensure core tables: {str(e)}")

    def validate_all_schemas(self) -> bool:
        """Validate that all schemas are properly installed.

        Returns:
            bool: True if all schemas are valid
        """
        try:
            # Validate parameter estimation schema
            if not self.parameter_estimation_schema.validate_schema():
                return False

            # Check core tables exist
            with sqlite3.connect(self.db_path) as conn:
                required_core_tables = ["experiments", "versions", "backups"]
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                existing_tables = {row[0] for row in cursor.fetchall()}

                missing_core = set(required_core_tables) - existing_tables
                if missing_core:
                    return False

            return True

        except Exception as e:
            raise MigrationError(f"Failed to validate schemas: {str(e)}")

    def reset_parameter_estimation_schema(self) -> None:
        """Reset parameter estimation schema (for testing/development)."""
        try:
            self.parameter_estimation_schema.drop_parameter_estimation_tables()
            self.parameter_estimation_schema.migrate_schema()

        except Exception as e:
            raise MigrationError(f"Failed to reset parameter estimation schema: {str(e)}")

    def get_table_info(self) -> Dict[str, List[str]]:
        """Get information about all tables in the database.

        Returns:
            Dict[str, List[str]]: Table name -> column names mapping
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)

                table_info = {}
                for (table_name,) in cursor.fetchall():
                    # Validate table name to prevent SQL injection
                    if not self._validate_table_name(table_name):
                        raise MigrationError(f"Invalid table name detected: {table_name}")

                    _sql = f"PRAGMA table_info({table_name})"  # nosemgrep: sql-injection-risk - name validated above; PRAGMA cannot be parameterised

                    cursor = conn.execute(_sql)
                    columns = [row[1] for row in cursor.fetchall()]  # row[1] is column name
                    table_info[table_name] = columns

                return table_info

        except Exception as e:
            raise MigrationError(f"Failed to get table info: {str(e)}")


def create_migration_manager(db_path: Path) -> MigrationManager:
    """Factory function to create migration manager.

    Args:
        db_path: Path to SQLite database file

    Returns:
        MigrationManager: Initialized migration manager
    """
    return MigrationManager(db_path)
