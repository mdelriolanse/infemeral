#!/usr/bin/env python3
"""Reproduction script for 'cloaked' unbound variable error in server.py.

This script uses Python's ast module to statically check for unbound variable
references without needing the full runtime environment.
"""

import ast
import sys
from pathlib import Path

def find_unbound_variables(filepath: str) -> list[tuple[int, str]]:
    """Find variables used before definition in a Python file."""
    with open(filepath) as f:
        source = f.read()

    tree = ast.parse(source)
    issues = []

    # Simple check: find functions and look for 'cloaked' usage
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Collect all assigned names in this function
            assigned = set()
            for child in ast.walk(node):
                if isinstance(child, ast.Assign):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            assigned.add(target.id)
                elif isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                    assigned.add(child.target.id)

            # Check for 'cloaked' usage
            for child in ast.walk(node):
                if isinstance(child, ast.Name) and child.id == 'cloaked':
                    if child.id not in assigned:
                        issues.append((child.lineno, f"'{child.id}' used but never assigned in function '{node.name}'"))

    return issues


def main():
    server_path = Path(__file__).parent.parent.parent / "infemeral" / "server.py"

    if not server_path.exists():
        print(f"ERROR: {server_path} not found")
        return 1

    print(f"Checking {server_path} for unbound 'cloaked' variable...")

    issues = find_unbound_variables(str(server_path))

    if issues:
        print(f"\nFOUND {len(issues)} ISSUES:")
        for lineno, msg in issues:
            print(f"  Line {lineno}: {msg}")
        return 1
    else:
        print("\nNo issues found - 'cloaked' variable is properly handled")
        return 0


if __name__ == "__main__":
    sys.exit(main())
