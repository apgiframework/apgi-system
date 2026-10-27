"""Entry point for running the APGI Framework as a module.

This allows the framework to be executed using:
python -m apgi_framework [command] [options]
"""

from .cli import main

if __name__ == "__main__":
    main()
