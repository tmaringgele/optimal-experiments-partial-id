"""ID algorithm for causal effect identifiability.

Adapted from the Ananke causal-inference library.
(https://gitlab.com/causal/ananke)
Original paper: Shpitser & Pearl (2006), "Identification of Joint
Interventional Distributions in Recursive Semi-Markovian Causal Models"
https://www.aaai.org/Papers/AAAI/2006/AAAI06-191.pdf


"""
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
#  Support functions
# ---------------------------------------------------------------------------

def _condset(var, orderlist):
    """Return the conditional set of var based on topological order."""
    pos = int(np.where(orderlist == var)[0][0])
    if pos == 0:
        return np.array([])
    return orderlist[0:pos]


def _toposort(varsarray, order):
    return np.array([orderi for orderi in order if orderi in varsarray])


def _removeedge(x, G, hiddvar):
    """Remove nodes x and any connected hidden variables from graph G."""
    G_del = G.copy()
    for edge in x:
        neighbors = list(G.predecessors(edge)) + list(G.successors(edge))
        delete = [u for u in hiddvar.tolist() if u in neighbors] + [edge]
        G_del.remove_nodes_from(delete)
    return G_del


def _incomingedgedel(nodes, G):
    """Remove all incoming edges to the specified nodes (mutilated graph)."""
    adjmtx = nx.to_numpy_array(G)
    node_list = list(nx.nodes(G))
    for nodei in nodes.tolist():
        adjmtx[:, node_list.index(nodei)] = 0
    G_bar_x = nx.from_numpy_array(adjmtx, create_using=nx.DiGraph)
    G_ = nx.relabel_nodes(G_bar_x, {i: j for i, j in enumerate(node_list)})
    return G_


def _outcomeedgedel(nodes, G):
    """Remove all outgoing edges from the specified nodes."""
    adjmtx = nx.to_numpy_array(G)
    node_list = list(nx.nodes(G))
    for nodei in nodes.tolist():
        adjmtx[node_list.index(nodei), :] = 0
    G_bar_x = nx.from_numpy_array(adjmtx, create_using=nx.DiGraph)
    G_ = nx.relabel_nodes(G_bar_x, {i: j for i, j in enumerate(node_list)})
    return G_


def _ispresent(x, graph):
    return np.array([node for node in x.tolist() if node in list(nx.nodes(graph))])


def _C_components(graph, hiddvar, orderlist):
    """Compute C-components (districts) of the graph."""
    adj_mtx = np.zeros((len(list(nx.nodes(graph))), len(list(nx.nodes(graph)))))
    listnode = list(nx.nodes(graph))

    for u in hiddvar:
        if u in listnode:
            neighbors = list(graph.neighbors(u))
            if len(neighbors) >= 2:
                adj_mtx[listnode.index(neighbors[0]), listnode.index(neighbors[1])] += 1
                adj_mtx[listnode.index(neighbors[1]), listnode.index(neighbors[0])] += 1

    # Delete non-observable nodes from adjacency matrix
    hidden_indices = [listnode.index(u) for u in hiddvar if u in listnode]
    adj_mtx2 = np.delete(np.delete(adj_mtx, hidden_indices, axis=1), hidden_indices, axis=0)

    for u in hiddvar:
        if u in listnode:
            listnode.remove(u)

    Gsub = nx.relabel_nodes(
        nx.from_numpy_array(adj_mtx2), {i: j for i, j in enumerate(listnode)}
    )

    C_comp = [np.array(list(component_i)) for component_i in nx.connected_components(Gsub)]

    # Order c-components by topological order
    C_topo = []
    for ii in range(len(C_comp)):
        posmax = [orderlist.tolist().index(Sub_C_i) for Sub_C_i in C_comp[ii]]
        C_topo.append(max(posmax))

    C_topo_sort = sorted(C_topo)
    newind = [C_topo_sort.index(ii) for ii in C_topo]
    C_comp = [C_comp[idx] for idx in newind]
    C_comp = [_toposort(C_comp[ii], orderlist) for ii in range(len(C_comp))]

    return C_comp


def _probabilitystr():
    return {
        "01_var": np.array([]),
        "02_cond": np.array([]),
        "03_sumvar": np.array([]),
        "04_numerator": [],
        "05_fraction": False,
        "06_denominator": [],
        "07_sum": False,
        "08_prod": False,
        "09_branch": [],
    }


def _an(x, G, orderlist):
    """Return ancestors of x (including x itself), topologically ordered."""
    anc = []
    for xx in x.tolist():
        anc.extend(list(nx.ancestors(G, xx)))
        anc.extend(list(x))
    anc = np.array([ordi for ordi in orderlist if ordi in anc])
    return np.array(anc)


def _complex_prob(P, var, condset, v, orderlist):
    if len(condset) == 0:
        P_complex = P.copy()
        if P_complex["07_sum"]:
            P_complex["03_sumvar"] = _toposort(
                np.union1d(P_complex["03_sumvar"], np.setdiff1d(v, np.union1d(var, condset))),
                orderlist,
            )
        else:
            P_complex["07_sum"] = True
            P_complex["03_sumvar"] = _toposort(
                np.setdiff1d(v, np.union1d(var, condset)), orderlist
            )
    else:
        P_complex = _probabilitystr()
        P_complex["05_fraction"] = True
        P_complex["04_numerator"] = P.copy()
        P_complex["06_denominator"] = P.copy()
        if P["07_sum"]:
            P_complex["04_numerator"]["03_sumvar"] = _toposort(
                np.union1d(P["03_sumvar"], np.setdiff1d(v, np.union1d(var, condset))),
                orderlist,
            )
            P_complex["06_denominator"]["03_sumvar"] = _toposort(
                np.union1d(P["03_sumvar"], np.setdiff1d(v, condset)), orderlist
            )
        else:
            P_complex["04_numerator"]["03_sumvar"] = _toposort(
                np.setdiff1d(v, np.union1d(var, condset)), orderlist
            )
            P_complex["06_denominator"]["03_sumvar"] = _toposort(
                np.setdiff1d(v, condset), orderlist
            )
    return P_complex


# ---------------------------------------------------------------------------
#  Main ID algorithm
# ---------------------------------------------------------------------------

def _ID(y, x, P, G, orderlist, hiddvar):
    """ID algorithm (Shpitser & Pearl 2006).

    Raises ValueError if the causal effect is not identifiable (hedge found).
    """
    P_out = P.copy()

    # Line 0: get observed graph
    present_hidden = _ispresent(hiddvar, G)
    if len(present_hidden) > 0:
        G_obs = _removeedge(present_hidden, G, present_hidden)
        v = np.array(list(nx.nodes(G_obs)))
        v = np.array([ordi for ordi in orderlist if ordi in v])
    else:
        G_obs = G.copy()
        v = np.array(list(nx.nodes(G_obs)))
        v = np.array([ordi for ordi in orderlist if ordi in v])

    # Line 1: if x is empty
    if len(x) == 0:
        if P_out["08_prod"] or P_out["05_fraction"]:
            P_out["03_sumvar"] = _toposort(
                np.union1d(np.setdiff1d(v, y), P_out["03_sumvar"]), orderlist
            )
            P_out["07_sum"] = True
        else:
            if len(v) == 1:
                P_out["07_sum"] = True
                P_out["03_sumvar"] = _toposort(
                    np.union1d(np.setdiff1d(v, y), P_out["03_sumvar"]), orderlist
                )
            P_out["01_var"] = y
        return P_out

    # Line 2: ancestor restriction
    anc = _an(y, G_obs, orderlist)
    if len(np.setdiff1d(v, anc)) != 0:
        edgedel = np.array([u for u in list(nx.nodes(G_obs)) if u not in anc])
        G_an = _removeedge(edgedel, G, _ispresent(hiddvar, G))

        if P_out["08_prod"] or P_out["05_fraction"]:
            P_out["03_sumvar"] = _toposort(
                np.union1d(np.setdiff1d(v, anc), P_out["03_sumvar"]), orderlist
            )
            P_out["07_sum"] = True
        else:
            P_out["01_var"] = anc

        P_iter = _ID(y, np.intersect1d(x, anc), P_out, G_an, orderlist, hiddvar)
        return P_iter

    # Line 3: W computation
    G_bar_x = _incomingedgedel(x, G_obs)
    an_bar_x = _an(y, G_bar_x, orderlist)
    W = np.setdiff1d(np.setdiff1d(v, x), an_bar_x)
    if len(W) > 0:
        P_iter = _ID(y, np.union1d(x, W), P_out, G, orderlist, hiddvar)
        return P_iter

    G_less_x = _removeedge(x, G, hiddvar)
    C_G_less_x = _C_components(G_less_x, hiddvar, orderlist)

    # Line 4: multiple c-components
    if len(C_G_less_x) > 1:
        P_iter = []
        for p_iter in range(len(C_G_less_x)):
            P_iter_i = _ID(
                C_G_less_x[p_iter],
                np.setdiff1d(v, C_G_less_x[p_iter]),
                P_out, G, orderlist, hiddvar,
            )
            P_iter.append(P_iter_i)

        if len(np.setdiff1d(v, np.union1d(y, x))) != 0:
            P_out["07_sum"] = True
        P_out["08_prod"] = True
        P_out["03_sumvar"] = _toposort(np.setdiff1d(v, np.union1d(y, x)), orderlist)
        P_out["09_branch"] = P_iter
        return P_out

    if len(C_G_less_x) == 1:
        C_G = _C_components(G, hiddvar, orderlist)

        # Line 5: hedge (not identifiable)
        if len(C_G) == 1:
            if len(np.setdiff1d(v, C_G)) == 0:
                raise ValueError(
                    "Graph forms a hedge: causal effect is not identifiable"
                )

        # Line 6: S' in C(G)
        if any([C_G_less_x[0].tolist() == C_Gi.tolist() for C_Gi in C_G]):
            P_prod_b = []
            for ii in range(len(C_G_less_x[0])):
                if P_out["08_prod"]:
                    cond_list = _condset(C_G_less_x[0][ii], v)
                    P_prod_i = _complex_prob(P_out, C_G_less_x[0][ii], cond_list, v, orderlist)
                else:
                    P_prod_i = P_out.copy()
                    P_prod_i["01_var"] = np.array(C_G_less_x[0][ii])
                    cond_list = _condset(C_G_less_x[0][ii], v)
                    P_prod_i["02_cond"] = cond_list
                P_prod_b.append(P_prod_i)

            if len(C_G_less_x[0]) > 1:
                P_6 = _probabilitystr()
                P_6["03_sumvar"] = _toposort(np.setdiff1d(v, np.union1d(y, x)), orderlist)
                P_6["09_branch"] = P_prod_b.copy()
                P_6["08_prod"] = True
            else:
                P_6 = P_prod_b[0].copy()
                if P_6["08_prod"] or P_6["05_fraction"]:
                    P_6["03_sumvar"] = np.union1d(
                        P_6["03_sumvar"], np.setdiff1d(C_G_less_x[0], y)
                    )
                else:
                    P_6["01_var"] = np.setdiff1d(
                        P_6["01_var"],
                        np.union1d(P_6["03_sumvar"], np.setdiff1d(C_G_less_x[0], y)),
                    )
            return P_6

        # Line 7: S' subset of some C_Gi
        if any([len(np.setdiff1d(C_G_less_x, C_Gi)) == 0 for C_Gi in C_G]):
            S = [C_Gi for C_Gi in C_G if len(np.setdiff1d(C_G_less_x, C_Gi)) == 0][0]
            P_ = _probabilitystr()
            P_prod_b = []

            for iii in range(len(S)):
                if P_out["08_prod"]:
                    cond_list = _condset(S[iii], v)
                    P_prod_i = _complex_prob(P_out, S[iii], cond_list, v, orderlist)
                else:
                    P_prod_i = P_out.copy()
                    P_prod_i["01_var"] = np.array(S[iii])
                    cond_list = _condset(S[iii], v)
                    P_prod_i["02_cond"] = cond_list
                P_prod_b.append(P_prod_i)

            if len(S) > 1:
                P_["08_prod"] = True
                P_["09_branch"] = P_prod_b.copy()
            else:
                P_ = P_prod_b[0].copy()

            edgedel = [u for u in list(nx.nodes(G_obs)) if u not in S]
            G_s = _removeedge(edgedel, G, hiddvar)
            P_iter = _ID(y, np.intersect1d(x, S), P_, G_s, orderlist, hiddvar)
            return P_iter


# ---------------------------------------------------------------------------
#  Public wrapper
# ---------------------------------------------------------------------------

def is_identifiable(causal_graph, y_vars, x_vars):
    """Check if P(y_vars | do(x_vars)) is identifiable in the causal graph.

    Uses the ID algorithm (Shpitser & Pearl 2006). Returns True if identifiable,
    False if a hedge is found (non-identifiable).

    Args:
        causal_graph: CausalGraph object
        y_vars: list/set of outcome variable names
        x_vars: list/set of intervention variable names

    Returns:
        True if identifiable, False otherwise
    """
    G = causal_graph.graph.copy()
    hiddvar = np.array(causal_graph.hidden) if causal_graph.hidden else np.array([])

    # Get topological order from observed graph
    G_obs = G.copy()
    if len(causal_graph.hidden) > 0:
        G_obs.remove_nodes_from(causal_graph.hidden)
    orderlist = np.array(list(nx.topological_sort(G_obs)))

    y = np.array(list(y_vars))
    x = np.array(list(x_vars))
    P = _probabilitystr()

    try:
        _ID(y, x, P, G, orderlist, hiddvar)
        return True
    except ValueError:
        return False
