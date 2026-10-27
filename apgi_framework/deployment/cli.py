"""Command-line interface for deployment automation."""

import sys
from pathlib import Path
from typing import Any, Tuple

import click

# Handle both module import and direct execution
try:
    from ..logging.standardized_logging import get_logger
    from .automation_manager import DeploymentAutomationManager
except ImportError:
    # Add parent directory to path for direct execution
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    from apgi_framework.deployment.automation_manager import DeploymentAutomationManager
    from apgi_framework.logging.standardized_logging import get_logger

logger = get_logger(__name__)


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="Deployment configuration file")
@click.pass_context
def cli(ctx: Any, config: Any) -> None:
    """APGI Framework Deployment Automation CLI."""
    ctx.ensure_object(dict)
    ctx.obj["config_path"] = Path(config) if config else None
    ctx.obj["manager"] = DeploymentAutomationManager(ctx.obj["config_path"])


@cli.command()
@click.option("--force", "-f", is_flag=True, help="Force redeployment")
@click.pass_context
def deploy(ctx: Any, force: bool) -> None:
    """Deploy the application."""
    manager = ctx.obj["manager"]

    click.echo("🚀 Starting APGI Framework Deployment...")

    if manager.deploy(force=force):
        click.echo("✅ Deployment successful!")

        # Show status
        status = manager.get_status()
        click.echo(f"Container ID: {status['container_id']}")
        click.echo(f"Health Status: {status['health_status']}")

        if status["monitoring_active"]:
            click.echo("📊 Monitoring is active")
    else:
        click.echo("❌ Deployment failed!")
        ctx.exit(1)


@cli.command()
@click.pass_context
def status(ctx: Any) -> None:
    """Show deployment status."""
    manager = ctx.obj["manager"]
    status = manager.get_status()

    click.echo("📊 Deployment Status")
    click.echo("=" * 50)
    click.echo(f"Running: {'Yes' if status['is_running'] else 'No'}")
    click.echo(f"Container ID: {status['container_id'] or 'N/A'}")
    click.echo(f"Health Status: {status['health_status']}")
    click.echo(f"CPU Usage: {status['cpu_usage']:.1f}%")
    click.echo(f"Memory Usage: {status['memory_usage']:.1f}%")
    click.echo(f"Error Count: {status['error_count']}")

    if status["last_error"]:
        click.echo(f"Last Error: {status['last_error']}")

    if status["start_time"]:
        click.echo(f"Start Time: {status['start_time']}")

    if status["last_health_check"]:
        click.echo(f"Last Health Check: {status['last_health_check']}")

    click.echo(f"Monitoring: {'Active' if status['monitoring_active'] else 'Inactive'}")


@cli.command()
@click.pass_context
def stop(ctx: Any) -> None:
    """Stop the deployment."""
    manager = ctx.obj["manager"]

    click.echo("🛑 Stopping deployment...")
    manager.cleanup()
    click.echo("✅ Deployment stopped")


@cli.command()
@click.option(
    "--environment",
    "-e",
    type=click.Choice(["development", "staging", "production"]),
    default="production",
    help="Target environment",
)
@click.option("--tag", "-t", help="Docker image tag")
@click.option("--port", "-p", multiple=True, help="Port mappings (host:container)")
@click.option("--env", multiple=True, help="Environment variables (KEY=VALUE)")
@click.pass_context
def configure(ctx: Any, environment: str, tag: Any, port: Tuple, env: Tuple) -> None:
    """Configure deployment settings."""
    manager = ctx.obj["manager"]

    # Update configuration
    manager.config.environment = environment
    if tag:
        manager.config.docker_tag = tag

    # Parse port mappings
    if port:
        ports = {}
        for p in port:
            if ":" in p:
                host, container = p.split(":", 1)
                ports[host] = int(container)
        manager.config.ports.update(ports)

    # Parse environment variables
    if env:
        env_vars = {}
        for e in env:
            if "=" in e:
                key, value = e.split("=", 1)
                env_vars[key] = value
        manager.config.environment_vars.update(env_vars)

    # Save configuration
    manager._save_config()

    click.echo("✅ Configuration updated")
    click.echo(f"Environment: {manager.config.environment}")
    click.echo(f"Docker Tag: {manager.config.docker_tag}")
    click.echo(f"Ports: {manager.config.ports}")
    click.echo(f"Environment Variables: {manager.config.environment_vars}")


@cli.command()
@click.pass_context
def validate(ctx: Any) -> None:
    """Validate deployment prerequisites."""
    manager = ctx.obj["manager"]

    click.echo("🔍 Validating deployment prerequisites...")

    validation_report = manager.validator.validate_deployment()
    summary = manager.validator.generate_summary()

    click.echo(summary)

    if validation_report.overall_passed:
        click.echo("✅ Validation passed")
    else:
        click.echo("❌ Validation failed")
        ctx.exit(1)


@cli.command()
@click.option("--output", "-o", type=click.Path(), help="Output file for report")
@click.pass_context
def health(ctx: Any, output: Any) -> None:
    """Perform detailed health check."""
    manager = ctx.obj["manager"]

    click.echo("🏥 Performing health check...")

    validation_report = manager.validator.validate_deployment(run_performance_tests=True)

    if output:
        manager.validator.save_report(Path(output))
        click.echo(f"📄 Report saved to {output}")
    else:
        summary = manager.validator.generate_summary()
        click.echo(summary)

    if not validation_report.overall_passed:
        ctx.exit(1)


@cli.command()
@click.pass_context
def logs(ctx: Any) -> None:
    """Show deployment logs."""
    manager = ctx.obj["manager"]

    if not manager._is_container_running():
        click.echo("❌ Container is not running")
        ctx.exit(1)

    import subprocess

    try:
        subprocess.run(["docker", "logs", "-f", manager.config.container_name])
    except KeyboardInterrupt:
        click.echo("\n📄 Logs display stopped")


@cli.command()
@click.pass_context
def history(ctx: Any) -> None:
    """Show deployment history."""
    manager = ctx.obj["manager"]

    if not manager.deployment_history:
        click.echo("No deployment history available")
        return

    click.echo("📜 Deployment History")
    click.echo("=" * 80)

    for i, deployment in enumerate(reversed(manager.deployment_history[-10:]), 1):
        status_icon = "✅" if deployment["status"] == "success" else "❌"
        click.echo(f"{i}. {status_icon} {deployment['timestamp']}")
        click.echo(f"   Environment: {deployment['environment']}")
        click.echo(f"   Tag: {deployment['docker_tag']}")
        click.echo(f"   Status: {deployment['status']}")
        if deployment.get("error"):
            click.echo(f"   Error: {deployment['error']}")
        click.echo()


if __name__ == "__main__":
    cli()
