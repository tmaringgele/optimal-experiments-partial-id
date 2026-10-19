"""Parse BIF (Bayesian Interchange Format) files to extract DAG structure."""
import re


def parse_bif(filepath):
    """Parse a .bif file and return variable names and directed edges.

    Only extracts graph structure (variable names and parent-child relationships).
    Probability tables are ignored.

    Returns:
        variables: list of variable names
        edges: list of (parent, child) tuples
    """
    variables = []
    edges = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract variable names from 'variable <name> {' declarations
    for match in re.finditer(r'variable\s+(\w+)\s*\{', content):
        variables.append(match.group(1))

    # Extract edges from 'probability ( child | parent1, parent2, ... )' statements
    for match in re.finditer(
        r'probability\s*\(\s*(\w+)\s*(?:\|\s*([\w\s,]+))?\s*\)', content
    ):
        child = match.group(1)
        parents_str = match.group(2)
        if parents_str:
            parents = [p.strip() for p in parents_str.split(',')]
            for parent in parents:
                edges.append((parent, child))

    return variables, edges
