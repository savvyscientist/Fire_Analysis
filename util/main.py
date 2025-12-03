#!/usr/bin/env python3
"""
Main entry point for the scientific data analysis pipeline.

Usage:
    python main.py
"""

from workflow import WorkflowRunner


def main():
    """Run the complete analysis workflow."""
    runner = WorkflowRunner(config_file="utilityEnvVar.json")
    runner.run()


if __name__ == "__main__":
    main()
