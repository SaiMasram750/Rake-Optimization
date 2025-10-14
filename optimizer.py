from pulp import LpProblem, LpMinimize, LpVariable, lpSum

def optimize_rake_plan(orders, wagons):
    prob = LpProblem("RakeFormation", LpMinimize)

    assign = LpVariable.dicts("Assign", [(o, w) for o in orders for w in wagons], cat='Binary')

    prob += lpSum([assign[o, w] * (wagons[w] - orders[o]) for o in orders for w in wagons])

    for o in orders:
        prob += lpSum([assign[o, w] for w in wagons]) == 1

    for w in wagons:
        prob += lpSum([assign[o, w] * orders[o] for o in orders]) <= wagons[w]

    prob.solve()

    result = []
    for o in orders:
        for w in wagons:
            if assign[o, w].varValue == 1:
                result.append((o, w))
    return result
