"""Multiagent system which consist of 3 specialists:
1. Researcher - finds information in Internet and analyzes data
2. Data Engineer - structures data, creates CSV and works with SQLite
3. Editor - creates final reports in md format

All agents are managed by orchestrator, who coordinates their work and distributes the results between the stages.
"""

if __name__ == "__main__":
    raise SystemExit(main())