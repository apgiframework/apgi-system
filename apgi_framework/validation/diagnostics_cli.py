"""Diagnostics CLI Tool.

Command-line interface for running system health checks and diagnostics.
"""

import argparse
import sys
from pathlib import Path

# Handle both module import and direct execution
try:
    from ..config import get_config_manager
    from .parameter_validator import get_validator
    from .system_health import get_health_checker
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from apgi_framework.config import get_config_manager
    from apgi_framework.validation.parameter_validator import get_validator
    from apgi_framework.validation.system_health import get_health_checker

from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


def run_health_check(args: argparse.Namespace) -> None:
    """Run system health check."""
    logger.info("\n" + "=" * 60)
    logger.info("APGI FRAMEWORK SYSTEM HEALTH CHECK")
    logger.info("=" * 60 + "\n")

    health_checker = get_health_checker()

    if args.component:
        # Check specific component
        result = health_checker.check_component(args.component)
    else:
        # Full health check
        result = health_checker.run_full_health_check()

    logger.info(result.get_report())

    # Exit with appropriate code
    if result.overall_status == "critical":
        sys.exit(1)
    elif result.overall_status == "warning":
        sys.exit(2)
    else:
        sys.exit(0)


def run_diagnostics(args: argparse.Namespace) -> None:
    """Run diagnostic information gathering."""
    logger.info("\n" + "=" * 60)
    logger.info("APGI FRAMEWORK DIAGNOSTIC INFORMATION")
    logger.info("=" * 60 + "\n")

    health_checker = get_health_checker()
    info = health_checker.get_diagnostic_info()

    logger.info("System Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")

    logger.info("\nConfiguration:")
    try:
        config_manager = get_config_manager()
        apgi_params = config_manager.get_apgi_parameters()
        exp_config = config_manager.get_experimental_config()

        logger.info("  APGI Parameters:")
        logger.info(f"    Exteroceptive Precision: {apgi_params.extero_precision}")
        logger.info(f"    Interoceptive Precision: {apgi_params.intero_precision}")
        logger.info(f"    Threshold: {apgi_params.threshold}")

        logger.info("  Experimental Config:")
        logger.info(f"    Trials: {exp_config.n_trials}")
        logger.info(f"    Participants: {exp_config.n_participants}")
        logger.info(f"    Alpha Level: {exp_config.alpha_level}")
    except Exception as e:
        logger.info(f"  Error loading configuration: {str(e)}")


def validate_parameters(args: argparse.Namespace) -> None:
    """Validate parameters from command line."""
    logger.info("\n" + "=" * 60)
    logger.info("PARAMETER VALIDATION")
    logger.info("=" * 60 + "\n")

    validator = get_validator()

    # Parse parameters
    params = {}
    if args.params:
        for param_str in args.params:
            try:
                key, value = param_str.split("=")
                params[key] = float(value)
            except ValueError:
                logger.info(f"Error: Invalid parameter format: {param_str}")
                logger.info("Expected format: key=value")
                sys.exit(1)

    if not params:
        logger.info("No parameters provided")
        sys.exit(1)

    # Validate
    result = validator.validate_apgi_parameters(**params)

    logger.info(result.get_message())

    if not result.is_valid:
        sys.exit(1)


def get_parameter_info(args: argparse.Namespace) -> None:
    """Get information about a parameter."""
    validator = get_validator()

    if not args.parameter:
        logger.info("Error: No parameter name provided")
        sys.exit(1)

    info = validator.get_parameter_info(args.parameter)
    logger.info(f"\n{info}\n")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="APGI Framework Diagnostics and Validation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full health check
  python -m apgi_framework.validation.diagnostics_cli health-check
  
  # Check specific component
  python -m apgi_framework.validation.diagnostics_cli health-check --component python
  
  # Run diagnostics
  python -m apgi_framework.validation.diagnostics_cli diagnostics
  
  # Validate parameters
  python -m apgi_framework.validation.diagnostics_cli validate --params extero_precision=2.0 intero_precision=1.5
  
  # Get parameter info
  python -m apgi_framework.validation.diagnostics_cli param-info --parameter threshold
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Health check command
    health_parser = subparsers.add_parser("health-check", help="Run system health check")
    health_parser.add_argument(
        "--component",
        type=str,
        choices=[
            "python",
            "dependencies",
            "configuration",
            "storage",
            "resources",
            "core",
        ],
        help="Check specific component",
    )

    # Diagnostics command
    subparsers.add_parser("diagnostics", help="Show diagnostic information")

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate parameters")
    validate_parser.add_argument(
        "--params", nargs="+", help="Parameters to validate (format: key=value)"
    )

    # Parameter info command
    param_parser = subparsers.add_parser("param-info", help="Get parameter information")
    param_parser.add_argument("--parameter", type=str, required=True, help="Parameter name")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Route to appropriate handler
    if args.command == "health-check":
        run_health_check(args)
    elif args.command == "diagnostics":
        run_diagnostics(args)
    elif args.command == "validate":
        validate_parameters(args)
    elif args.command == "param-info":
        get_parameter_info(args)


if __name__ == "__main__":
    main()
