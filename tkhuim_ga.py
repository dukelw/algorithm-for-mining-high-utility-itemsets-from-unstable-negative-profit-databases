import random


def get_dataset():
    data = []
    with open("4thdataset.txt", "r", encoding="utf-8") as dataset:
        next(dataset)
        for line in dataset:
            parts = line.split()
            transaction_id = parts[0]
            items = [str(u) for u in parts[1].split(",")]
            quantity = [int(u) for u in parts[2].split(",")]
            profit = [int(u) for u in parts[3].split(",")]
            data.append([[transaction_id], items, quantity, profit])
    return data


def get_utility(data):
    unit = set()
    for trans in data:
        for item in trans[1]:
            unit.add(item)
    utility = []
    for item in unit:
        utility.append([item, 0])
    return utility


def calculate_utility(data, utility):
    dataset = []
    for i in range(len(data)):
        dataset.append([data[i][0], data[i][1], data[i][2]])
        for j in range(len(data[i][1])):
            dataset[i][2][j] = data[i][2][j] * data[i][3][j]
    return dataset


data = get_dataset()
utility = list(get_utility(data))
dataset = calculate_utility(data, utility)


def utility_itemset(dataset, utility):
    utility_trans = utility
    for k in range(len(utility_trans)):
        utility_trans[k][1] = 0
    for i in range(len(dataset)):
        for j in range(len(dataset[i][1])):
            for k in range(len(utility_trans)):
                if dataset[i][1][j] == utility_trans[k][0]:
                    utility_trans[k][1] += dataset[i][2][j]
    return utility_trans


u = utility_itemset(dataset, utility)


def TU(transaction):
    return sum(transaction[2])


def get_top_m_items(transaction, m):
    items_with_utilities = list(zip(transaction[1], transaction[2]))
    items_with_utilities.sort(key=lambda x: x[1], reverse=True)
    top_m_items = items_with_utilities[:m]
    top_items = [item[0] for item in top_m_items]
    top_utilities = [item[1] for item in top_m_items]
    return [transaction[0], top_items, top_utilities]


def initial_solutions(dataset, n, m):
    trans_P = []
    P = []
    for Ty in dataset:
        u = TU(Ty)
        X = get_top_m_items(Ty, m)
        if len(P) < n:
            trans_P.append(X)
            P.append(X[1])
        else:
            min_utility = min(trans_P, key=TU)
            if u > TU(min_utility):
                trans_P.remove(min_utility)
                P.remove(min_utility[1])
                trans_P.append(X)
                P.append(X[1])
    return P


def F(X):
    sum = 0
    if X is None:
        return sum
    for i in range(len(dataset)):
        if set(X).issubset(set(dataset[i][1])):
            for j in range(len(dataset[i][1])):
                if dataset[i][1][j] in X:
                    sum += dataset[i][2][j]
    return sum


def roullete_wheel(utility_itemset):
    elements = []
    weights = []
    sum = 0
    for item in utility_itemset:
        elements.append(item[0])
        weights.append(item[1])
        sum += item[1]
    for i in range(len(weights)):
        weights[i] = weights[i] / sum
    return random.choices(elements, weights, k=1)


def genetic_operators(S, a, b):
    P = set()
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            Xi = S[i]
            Xj = S[j]
            if a > random.uniform(0, 1):
                x = ""
                y = ""
                if F(Xi) > F(Xj):
                    minXi = 0
                    maxXj = 0
                    for item in utility_itemset(dataset, utility):
                        if Xi is not None:
                            if item[0] in Xi:
                                minXi = item[1]
                                x = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xi is not None:
                            if item[0] in Xi:
                                if item[1] < minXi:
                                    minXi = item[1]
                                    x = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xj is not None:
                            if item[0] in Xj:
                                maxXj = item[1]
                                y = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xj is not None:
                            if item[0] in Xj:
                                if item[1] > maxXj:
                                    maxXj = item[1]
                                    y = item[0]
                else:
                    minXj = 0
                    maxXi = 0
                    for item in utility_itemset(dataset, utility):
                        if Xj is not None:
                            if item[0] in Xj:
                                minXj = item[1]
                                y = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xj is not None:
                            if item[0] in Xj:
                                if item[1] < minXj:
                                    minXj = item[1]
                                    y = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xi is not None:
                            if item[0] in Xi:
                                maxXi = item[1]
                                x = item[0]
                    for item in utility_itemset(dataset, utility):
                        if Xi is not None:
                            if item[0] in Xi:
                                if item[1] > maxXi:
                                    maxXi = item[1]
                                    x = item[0]

                if Xi is None:
                    Xi = set()
                else:
                    Xi = set(Xi)

                Xi = Xi - {x}
                Xi = Xi | {y}

                if Xj is None:
                    Xj = set()
                else:
                    Xj = set(Xj)

                Xj = Xj - {y}
                Xj = Xj | {x}

            for X in [Xi, Xj]:
                if b > random.uniform(0, 1):
                    x = ""
                    if 0.5 > random.uniform(0, 1):
                        minX = 0
                        for item in utility_itemset(dataset, utility):
                            if X is not None:
                                if item[0] in X:
                                    minX = item[1]
                                    x = item[0]
                        for item in utility_itemset(dataset, utility):
                            if X is not None:
                                if item[0] in X:
                                    if item[1] < minX:
                                        minX = item[1]
                                        x = item[0]
                        if X is None:
                            X = set()
                        else:
                            X = set(X)
                        X = set(X) - {x}
                    else:
                        x = roullete_wheel(utility_itemset(dataset, utility))
                        if X is None:
                            X = set()
                        else:
                            X = set(X)
                        if x is None:
                            x = set()
                        else:
                            x = set(x)
                        X = set(X) | set(x)
                if X is not None and len(X) != 0:
                    P.add("".join(X))
    return P


def contains_same_characters(E, target):
    target_set = set(target)

    for item in E:
        if set(item) == target_set:
            return True

    return False


def TKHUIM_GA(dataset, n, m, e):
    HUP = {}
    P = []
    E = []
    u = utility_itemset(dataset, utility)
    u.sort(key=lambda x: x[1], reverse=True)

    top_m_items = u[:e]
    top_items = [item[0] for item in top_m_items]
    E = top_items

    exit = False
    P = initial_solutions(dataset, n, m)
    a = 0.5
    b = 0.5
    while True:
        S = tournament_selection(P, len(P) - 1, n)

        P = genetic_operators(S, a, b)

        new_E = []
        for item in P:
            if not contains_same_characters(new_E, item):
                new_E.append(item)
            if not contains_same_characters(HUP.keys(), item):
                HUP[item] = F(list(item))
        for item in E:
            if not contains_same_characters(new_E, item):
                new_E.append(item)
            if not contains_same_characters(HUP.keys(), item):
                HUP[item] = F(list(item))

        new_E.sort(key=F, reverse=True)
        new_E = list(set(new_E))
        HUP = dict(sorted(HUP.items(), key=lambda item: item[1], reverse=True))

        if new_E != E:
            a = a + 0.05
            b = b - 0.05
            E = new_E
        else:
            a = a - 0.05
            b = b + 0.05

        if round(b, 2) == 1.00:
            exit = True

        if exit:
            break
    return dict(list(HUP.items())[:e])


def tournament(T, k):
    # INPUT
    #   T = a list of individuals randomly selected from a population.
    #   k = the tournament size. In other words, the number of elements in T.
    # OUTPUT
    #   the fittest individual.
    if k == 0:
        return
    best = T[0]
    for i in range(1, k):
        next = T[i]
        if F(next) > F(best):
            best = next
    return best


def tournament_selection(P, k, n):
    # INPUT
    #   P = the population as a set of individuals.
    #   k = the tournament size, such that 1 ≤ k ≤ the number of individuals in P.
    #   n = the total number of individuals we wish to select.
    # OUTPUT
    #   the pool of individuals selected in the tournaments.

    P_list = list(P)
    B = [None] * n

    for i in range(n):
        T = random.choices(P_list, k=k)  # Picks with replacement
        B[i] = tournament(T, len(T))

    return B


E = TKHUIM_GA(dataset, 4, 5, 20)
for item in E:
    print(item, "-", E[item])
