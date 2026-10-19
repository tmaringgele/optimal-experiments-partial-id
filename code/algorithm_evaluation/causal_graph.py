"""Causal graph representation with district and response-type computations.

Implements the graphical operations needed for GetUselessExperiments:
- District computation (c-components)
- R_theta: query-relevant response-type variables
- R*_theta: district-hull response-types
- AllPathsBlocked (Algorithm 2 from the paper)
"""
import networkx as nx
from collections import deque


class CausalGraph:
    """A causal DAG with observed variables and latent confounders.

    Observed variables are regular nodes. Each latent confounder U connects
    exactly two observed variables via directed edges U -> V1, U -> V2,
    representing a bidirected edge V1 <-> V2.
    """

    def __init__(self, observed_vars, directed_edges, hidden_confounders=None):
        """
        Args:
            observed_vars: list of observed variable names
            directed_edges: list of (parent, child) tuples among observed vars
            hidden_confounders: list of (var1, var2) tuples for latent confounders
        """
        self.observed = list(observed_vars)
        self.observed_set = set(self.observed)
        self.directed_edges = list(directed_edges)
        self.hidden = []
        self.hidden_children = {}  # hidden_var_name -> (child1, child2)

        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.observed)
        self.graph.add_edges_from(self.directed_edges)

        if hidden_confounders:
            for v1, v2 in hidden_confounders:
                self.add_confounder(v1, v2)

    def add_confounder(self, v1, v2):
        """Add a latent confounder between two observed variables."""
        name = f"U_{v1}_{v2}"
        # Ensure unique naming
        idx = 0
        while name in self.graph:
            idx += 1
            name = f"U_{v1}_{v2}_{idx}"
        self.hidden.append(name)
        self.hidden_children[name] = (v1, v2)
        self.graph.add_node(name)
        self.graph.add_edge(name, v1)
        self.graph.add_edge(name, v2)

    def observed_parents(self, var):
        """Get observed parents of a variable (excluding hidden confounders)."""
        return [p for p in self.graph.predecessors(var) if p in self.observed_set]

    def observed_ancestors(self, variables):
        """Get all observed ancestors of a set of variables (excluding the variables themselves).

        Uses the directed graph to find all nodes with a directed path to any
        variable in the set, restricted to observed variables.

        Args:
            variables: iterable of variable names

        Returns:
            set of observed variable names that are ancestors
        """
        ancestors = set()
        for v in variables:
            ancestors |= nx.ancestors(self.graph, v)
        return (ancestors & self.observed_set) - set(variables)

    def get_districts(self):
        """Compute districts (c-components).

        A district is a maximal set of observed variables connected by
        bidirected edges, together with the hidden variables that create
        those bidirected edges. Observed variables not involved in any
        bidirected edge form singleton districts.

        Returns:
            list of frozensets, each containing variable names (observed + hidden)
        """
        # Build undirected graph of bidirected connections among observed vars
        bidir = nx.Graph()
        bidir.add_nodes_from(self.observed)
        for u_name, (v1, v2) in self.hidden_children.items():
            bidir.add_edge(v1, v2)

        districts = []
        for component in nx.connected_components(bidir):
            district = set(component)  # observed vars in this district
            # Add hidden vars whose children are in this district
            for u_name, (v1, v2) in self.hidden_children.items():
                if v1 in district or v2 in district:
                    district.add(u_name)
            districts.append(frozenset(district))
        return districts

    def get_R_theta(self, query_worlds):
        """Get R_theta: response-type variables relevant for query theta.

        R_theta = {R in hidden : exists world c, exists directed path
                   R -> Y^(c) not passing through X^(c)}

        Args:
            query_worlds: list of (Y_set, X_set) tuples, one per
                          counterfactual world

        Returns:
            set of hidden variable names
        """
        R_theta = set()
        for Y_set, X_set in query_worlds:
            reduced = self.graph.copy()
            reduced.remove_nodes_from(X_set)
            for r in self.hidden:
                if r in R_theta or r not in reduced:
                    continue
                for y in Y_set:
                    if y in reduced and nx.has_path(reduced, r, y):
                        R_theta.add(r)
                        break
        return R_theta

    def get_R_theta_star(self, query_worlds):
        """Get R*_theta: response-types in the district hull of R_theta.

        1. Compute R_theta (query-relevant response-types)
        2. Find all districts containing at least one R in R_theta
        3. R*_theta = all hidden vars in those districts

        Args:
            query_worlds: list of (Y_set, X_set) tuples

        Returns:
            set of hidden variable names
        """
        R_theta = self.get_R_theta(query_worlds)
        if not R_theta:
            return set()

        districts = self.get_districts()
        hidden_set = set(self.hidden)

        R_theta_star = set()
        for d in districts:
            if d & R_theta:
                for var in d:
                    if var in hidden_set:
                        R_theta_star.add(var)
        return R_theta_star

    def all_paths_blocked(self, R_theta_star, W, Z):
        """Check if all directed paths from R*_theta to W pass through Z.

        Implements Algorithm 2 (AllPathsBlocked) from the paper.
        Removes Z from the graph, then BFS from R*_theta. If any node
        in W is reached, returns False (unblocked path exists).

        Args:
            R_theta_star: set of hidden variable names (source nodes)
            W: set/frozenset of outcome variable names
            Z: set/frozenset of intervention variable names

        Returns:
            True if all paths are blocked, False otherwise
        """
        Z_set = set(Z)
        W_set = set(W)

        # Remove Z nodes from the graph
        reduced = self.graph.copy()
        reduced.remove_nodes_from(Z_set)

        # BFS from R*_theta nodes that are still in the reduced graph
        visited = set()
        queue = deque()
        for r in R_theta_star:
            if r in reduced:
                visited.add(r)
                queue.append(r)

        while queue:
            current = queue.popleft()
            if current in W_set:
                return False  # Found unblocked path
            for child in reduced.successors(current):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)

        return True  # All paths are blocked by Z
