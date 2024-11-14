# Demo
def read_data(file_name="ehmintable.txt"):
    with open(file_name, "r") as file:
        data = file.read()

    dataset = eval(data)
    return dataset


dataset = read_data()

# For unique printing
printed_itemsets = set()


class TransactionInfo:
    """A class to represent a transaction info <TID, U, PRU>."""

    def __init__(self, tid, utility, pru):
        self.tid = tid
        self.utility = utility
        self.pru = pru

    def __repr__(self):
        return f"<TID: {self.tid}, U: {self.utility}, PRU: {self.pru}>"


class TIVector:
    """A class to represent the TI-vector, a container for <TID, U, PRU> structures."""

    def __init__(self):
        self.transactions = []

    def add_transaction(self, tid, utility, pru):
        self.transactions.append(TransactionInfo(tid, utility, pru))

    def __repr__(self):
        return f"TI-Vector({self.transactions})"


class EHMINItem:
    """A class to represent an item in the EHMIN-list with its utility, PRU, and TI-vector."""

    def __init__(self, item_name=None, utility=0, pru=0):
        self.item_name = item_name
        self.utility = utility
        self.pru = pru
        self.ti_vector = TIVector() if item_name is not None else None

    def add_transaction_info(self, tid, utility, pru):
        self.ti_vector.add_transaction(tid, utility, pru)

    def set_ti_vector(self, ti_vector):
        self.ti_vector = ti_vector

    def __repr__(self):
        return (
            f"EHMINItem(Item: {self.item_name}, U: {self.utility}, PRU: {self.pru}, "
            f"{self.ti_vector})"
        )


class EHMINList:
    """A class to manage a global EHMIN-list."""

    def __init__(self):
        self.items = {}

    def find_or_create(self, item_name, utility=0, pru=0):
        """Finds an item by name or creates a new one if it doesn't exist."""
        if item_name not in self.items:
            self.items[item_name] = EHMINItem(item_name, utility, pru)
        return self.items[item_name]

    def increase_pru(self, item_name, pru):
        self.items[item_name].pru += pru

    def __repr__(self):
        return f"EHMINList({list(self.items.values())})"


def utility_of_itemset(itemset, transaction):
    """
    Calculate the utility of an itemset in a given transaction.
    :param itemset: List of items in the itemset.
    :param transaction: A dictionary with transaction data.
    :return: Total utility of the itemset in the transaction.
    """
    utility = 0
    for item in itemset:
        if item in transaction["items"]:
            idx = transaction["items"].index(item)
            utility += transaction["quantities"][idx] * transaction["profit"][idx]
    return utility


def transaction_utility(transaction):
    """
    Calculate the Transaction Utility (TU) for a given transaction.
    :param transaction: A dictionary with transaction data.
    :return: Total utility of the transaction.
    """
    utility = 0
    for i in range(len(transaction["items"])):
        utility += transaction["quantities"][i] * transaction["profit"][i]
    return utility


def redefine_transaction_utility(transaction):
    """
    Calculate the PTU /Redefined Transaction Utility (RTU) for a given transaction.
    :param transaction: A dictionary with transaction data.
    :return: Total Reduced Utility of the transaction.
    """
    RTU = 0
    for i in range(len(transaction["items"])):
        if transaction["profit"][i] > 0:
            RTU += transaction["quantities"][i] * transaction["profit"][i]
    return RTU


def calculate_rtwu(itemset, dataset):
    """
    Calculate the Redefined Transactional Weighted Utility (RTWU) for a given itemset across the dataset.
    :param itemset: A list of items defining the base itemset (e.g., ['a', 'b']).
    :param dataset: A list of transactions.
    :return: RTWU value for the itemset across the dataset.
    """
    RTWU = 0
    for transaction in dataset:
        # Check if all items in the itemset are in the transaction
        if all(item in transaction["items"] for item in itemset):
            # Calculate RTU for this transaction and add it to RTWU
            RTWU += redefine_transaction_utility(transaction)
    return RTWU


def transaction_weight_utility(itemset, dataset):
    """
    Calculate the Transaction Weight Utility (TWU) for an itemset in the dataset.
    :param itemset: List of items in the itemset.
    :param dataset: List of transactions.
    :return: Total TWU of the itemset across all transactions in the database.
    """
    TWU = 0
    for transaction in dataset:
        # Check if all items in itemset are present in the transaction
        if all(item in transaction["items"] for item in itemset):
            TWU += transaction_utility(transaction)
    return TWU


def redefine_transaction_weight_utility(itemset, dataset):
    """
    Calculate the Redefined Transaction Weight Utility (RTWU) for an itemset in the dataset.
    :param itemset: List of items in the itemset.
    :param dataset: List of transactions.
    :return: Total RTWU of the itemset across all transactions in the database.
    """
    RTWU = 0
    for transaction in dataset:
        # Check if all items in itemset are present in the transaction
        if all(item in transaction["items"] for item in itemset):
            RTWU += redefine_transaction_utility(transaction)
    return RTWU


def remaining_utility(itemset, transaction):
    """
    Calculate the remaining utility (ru) of an itemset in a transaction.
    :param itemset: List of items in the itemset.
    :param transaction: A dictionary with transaction data.
    :return: Remaining utility of the itemset in the transaction.
    """
    ru = 0
    # Get the last index in the sorted transaction where itemset items appear
    max_idx = max(
        transaction["items"].index(item)
        for item in itemset
        if item in transaction["items"]
    )

    # Sum the utility of items appearing after the itemset in the transaction
    for i in range(max_idx + 1, len(transaction["items"])):
        quantity = transaction["quantities"][i]
        profit = transaction["profit"][i]
        ru += quantity * profit

    return ru


def redefined_remaining_utility(itemset, transaction):
    """
    Calculate the redefined remaining utility (rru) of an itemset in a transaction.
    :param itemset: List of items in the itemset.
    :param transaction: A dictionary with transaction data.
    :return: Redefined remaining utility of the itemset in the transaction.
    """
    rru = 0

    # Lọc ra những item trong itemset mà có trong transaction["items"]
    valid_items = [item for item in itemset if item in transaction["items"]]

    # In case item set is an empty set, rru is an empty set
    if len(itemset) == 0:
        for i in range(0, len(transaction["items"])):
            quantity = transaction["quantities"][i]
            profit = transaction["profit"][i]
            if profit > 0:  # Only consider items with positive profit
                rru += quantity * profit
        return rru

    # Nếu không có item nào hợp lệ, trả về 0
    if not valid_items:
        return rru

    # Get the last index in the sorted transaction where itemset items appear
    max_idx = max(transaction["items"].index(item) for item in valid_items)
    # Sum the utility of items appearing after the itemset with positive profit
    for i in range(max_idx + 1, len(transaction["items"])):
        quantity = transaction["quantities"][i]
        profit = transaction["profit"][i]
        if profit > 0:  # Only consider items with positive profit
            rru += quantity * profit

    return rru


def categorize_items(dataset):
    """
    Calculate item utilities and classify items into positive, negative, and mixed utility sets.
    :param dataset: List of transactions.
    :return: Tuple of (positive_items, negative_items, mixed_items)
    """
    item_utilities = {}
    for transaction in dataset:
        for idx, item in enumerate(transaction["items"]):
            utility = transaction["quantities"][idx] * transaction["profit"][idx]
            if item not in item_utilities:
                item_utilities[item] = []
            item_utilities[item].append(utility)

    positive_items = set()
    negative_items = set()

    for item, utilities in item_utilities.items():
        if all(u > 0 for u in utilities):
            positive_items.add(item)
        elif all(u < 0 for u in utilities):
            negative_items.add(item)

    return positive_items, negative_items


def calculate_ptwus(dataset):
    """
    Calculate the Positive Transactional Weighted Utility (PTWU) and
    Redefined Transactional Weighted Utility (RTWU) for each item across the dataset.

    PTWU and RTWU measure the utility contribution and frequency of each item in the dataset.

    Parameters:
    dataset (list): A list of dictionaries, where each dictionary represents a transaction with:
        - 'items' (list of str): List of item names in the transaction.

    Returns:
    tuple: A tuple containing:
        - rtwus (dict): A dictionary where keys are items and values are their RTWU across transactions.
        - supports (dict): A dictionary where keys are items and values are their occurrence counts.
    """
    rtwus = {}
    supports = {}

    for transaction in dataset:
        for item in transaction["items"]:
            if item in supports:
                supports[item] += 1
            else:
                supports[item] = 1

            if item not in rtwus:
                itemset = [item]
                rtwus[item] = calculate_rtwu(itemset, dataset)

    return rtwus, supports


def get_items_order(itemset, positive_items, negative_items, rtwus, supports):
    """
    Sort items in a transaction according to the processing order:
    (i) PI items are sorted by RTWU (ascending), (ii) NI items are sorted by support (ascending).
    """
    itemset = dict(sorted(itemset.items()))

    def sort_key(item):
        # If item is in positive_items (PI), sort by RTWU (ascending)
        if item in positive_items:
            return (1, rtwus.get(item, float("inf")))
        # If item is in negative_items (NI), sort by support (ascending)
        elif item in negative_items:
            return (2, supports.get(item, float("inf")))
        # Default priority for items not in PI or NI
        return (3, float("inf"))

    # Sort the items in itemset using the custom sort_key
    sorted_items = sorted(itemset, key=sort_key)
    return sorted_items


def calculate_utility_and_dataset(itemset, dataset):
    """
    Calculate the utility of the given itemset and create the corresponding dataset Dβ.

    :param itemset: The itemset for which utility is to be calculated.
    :param dataset: The dataset containing transactions.
    :return: A tuple containing the total utility and the filtered dataset.
    """

    utility = 0

    for transaction in dataset:
        # Convert the itemset to a set for easier subset checking
        itemset_set = set(itemset)

        # Check if the itemset is a subset of the transaction's items
        if itemset_set.issubset(set(transaction["items"])):
            # Calculate utility for this transaction
            transaction_utility = 0

            # Calculate utility based on profit and quantities
            for item, quantity in zip(transaction["items"], transaction["quantities"]):
                if item in itemset_set:
                    index = transaction["items"].index(item)
                    profit = transaction["profit"][index]
                    transaction_utility += profit * quantity

            utility += transaction_utility

    return utility


def calculate_pu(pattern, transaction, positive_items):
    """
    Calculate the Positive Utility (PU) of a given pattern in a transaction.

    The positive utility of a pattern, X, in a transaction, Tk, is the sum of the utilities
    of items in the pattern that are also in the positive items list (PI) in the transaction.

    Parameters:
    - pattern: Set of items (list of item names) representing the pattern X.
    - transaction: Dictionary representing a single transaction with keys:
        - 'TID': Transaction ID.
        - 'items': List of item names in the transaction.
        - 'quantities': List of item quantities in the transaction, corresponding to 'items'.
        - 'profit': List of profit per item in the transaction, corresponding to 'items'.
    - positive_items: Set of items considered as positive (PI).

    Returns:
    - Positive utility (PU) of the pattern in the given transaction (integer).

    Example:
    - For pattern {D, E} in transaction T5 from the dataset, if only D is in PI,
    PU({D, E}, T5) = U(D, T5).
    """
    pu = 0  # Initialize positive utility

    # Loop through items in the transaction and calculate utility for items in both pattern and positive_items
    for item, quantity, profit in zip(
        transaction["items"], transaction["quantities"], transaction["profit"]
    ):
        if item in pattern and item in positive_items:
            utility = quantity * profit
            pu += (
                utility  # Add to positive utility if item is in both the pattern and PI
            )

    return pu


def build_eucs(order):
    eucs = [[0 for _ in range(len(order))] for _ in range(len(order))]

    for i in range(1, len(order)):
        eucs[0][i] = order[i - 1]
        eucs[i][0] = order[i]

    for i in range(0, len(order)):
        row_item = order[i]
        for j in range(i + 1, len(order)):
            col_item = order[j]
            tmp_set = {row_item, col_item}
            eucs[j][i + 1] = calculate_rtwu(tmp_set, dataset)

    return eucs


def calculate_pru(itemset, dataset):
    # Initialize PRU value
    pru = 0

    # Iterate through each transaction in the dataset
    for transaction in dataset:
        items = transaction["items"]
        profits = transaction["profit"]
        quantities = transaction["quantities"]

        # Check if itemset is a subset of the transaction's items
        if set(itemset).issubset(set(items)):
            # Calculate PRU(X, T_k) for items after the last item in the itemset
            pru_x_tk = sum(
                profits[i] * quantities[i]
                for i in range(0, len(items))
                if profits[i] > 0 and not set(items[i]).issubset(set(itemset))
            )

            # Add PRU(X, T_k) to the total PRU(X)
            pru += pru_x_tk

    return pru


def ehmin_combine(Uk, Ul, pfutils, minU):
    """
    Combine two EHMIN-lists (Uk, Ul) and create a new conditional EHMIN-list.
    Args:
        Uk: EHMIN-list for item Uk
        Ul: EHMIN-list for item Ul
        pfutils: Prefix utility map containing transaction IDs and their utilities
        minU: Minimum utility threshold for pruning

    Returns:
        A new EHMIN-list C if conditions are met, otherwise None
    """
    # Initialize the conditional utility pattern list C
    C = EHMINItem(Ul.item_name, 0, Ul.pru)
    # C.set_ti_vector(Ul.ti_vector)  # Start with Ul's transaction info
    x = Uk.utility + Uk.pru
    y = Uk.utility

    # Initialize iterators for the utility vectors of Uk and Ul
    current_k = 0
    current_l = 0

    # Get the lengths of utility vectors
    length_k = len(Uk.ti_vector.transactions)
    length_l = len(Ul.ti_vector.transactions)

    while current_k < length_k and current_l < length_l:
        sk = Uk.ti_vector.transactions[current_k]
        sl = Ul.ti_vector.transactions[current_l]

        if sk.tid == sl.tid:
            # Retrieve the prefix utility for the shared transaction ID
            pfutil = pfutils.get(sk.tid, 0)

            # Calculate the combined utility and remaining utility
            util = sk.utility + sl.utility - pfutil
            rutil = min(sk.pru, sl.pru)

            # Add the combined transaction to C
            C.add_transaction_info(sk.tid, utility=util, pru=rutil)
            C.utility += util

            # Update the `y` value for pruning
            y += sl.utility - pfutil

            # N-Prune condition
            if sk.pru == 0 and y < minU:
                return None

            # Move both iterators forward
            current_k += 1
            current_l += 1
        elif sk.tid > sl.tid:
            # Move iterator for Ul forward
            current_l += 1
        else:
            # LA-Prune condition: check if further processing is beneficial
            x -= sk.utility + sk.pru
            if x < minU:
                return None

            # Move iterator for Uk forward
            current_k += 1

    if len(C.ti_vector.transactions) == 0:
        return None

    return C


def ehmin_mine(P, UL, pref, eucs, minU, sorted_item):
    # Initialize the prefix utility map
    pfutils = {}
    if P.item_name != None:
        for s in P.ti_vector.transactions:
            pfutils[s.tid] = s.utility

    # Iterate over each Uk in UL
    for Uk in UL.items.values():
        # First pruning condition (U ≥ minUtil)
        if Uk.utility >= minU:
            tmp = pref.union({Uk.item_name})
            HUP["".join(tmp)] = Uk.utility

        # Second pruning condition (U + PRU ≥ minUtil)
        if Uk.utility + Uk.pru >= minU:
            # Initialize the conditional EHMIN-lists, CL
            CL = EHMINList()

            # Iterate over each Ul in UL where l > k
            for Ul in UL.items.values():
                k = sorted_item.index(Uk.item_name) + 1
                l = sorted_item.index(Ul.item_name)
                if k <= l:  # Ensure l > k
                    # EUCS pruning condition
                    if eucs[l][k] >= minU:
                        C = ehmin_combine(Uk, Ul, pfutils, minU)
                        if C:
                            CL.items[C.item_name] = C

            # Recursive call to EHMIN_Mine if CL is non-empty
            if len(CL.items) > 0:
                ehmin_mine(Uk, CL, pref | {Uk.item_name}, eucs, minU, sorted_item)


def ehmin(δ):
    # Step 1: 1st Database Scan
    # Calculate PTWU (RTWU)
    ptwus, supports = calculate_ptwus(dataset)
    ptus = {}
    print("ptwus", ptwus)
    print("supports", supports)
    # Calculate PTU (RTU) of each transaction
    for transaction in dataset:
        ptu = redefine_transaction_utility(transaction)
        ptus[transaction["TID"]] = ptu
    print("ptus", ptus)
    # Calculate minU
    minU = sum(value * δ for value in ptus.values())
    print("minU", minU)
    # Get EHMN-list for 1-itemsets
    positive_items, negative_items = categorize_items(dataset)
    list_item = {item: ptwu for item, ptwu in ptwus.items() if ptwu >= minU}
    sorted_item = get_items_order(
        list_item, positive_items, negative_items, ptwus, supports
    )
    print("sorted_item", sorted_item)
    # Index EHMIN-list sorted item
    # Calculate utility
    for item in sorted_item:
        utility = calculate_utility_and_dataset(item, dataset)
        # Index EHMIN-list with utility and pru = 0
        ehmin_item = ehmin_list.find_or_create(item, utility)
    print("Ehmin", ehmin_list)

    # Step 2: 2nd Database Scan
    for transaction in dataset:
        ptu_k = 0  # Recompute PTU(Tk) and initialize it to 0

        # Step 1: Calculate PTU for each transaction
        for item in transaction["items"]:
            # Check PTWU(i) condition for pruning
            if ptwus[item] > minU:
                ptu_k += calculate_pu(set(item), transaction, positive_items)

        # Initialize a temporary map
        tmp = {}

        # # Step 2: Insert items into tmp and calculate PTWU if necessary
        for item, quantity, profit in zip(
            transaction["items"], transaction["quantities"], transaction["profit"]
        ):
            tmp[item] = quantity * profit  # Store internal utility and external utility

            # PTWU condition to recompute PTWU
            if ptwus[item] > minU:
                new_PTWIU = calculate_rtwu(set(item), dataset)
                ptwus[item] = new_PTWIU + ptu_k

        rutil = 0  # Initialize rutil
        # Sort to calculate PRU (inportant)
        tmp_list_item = {item: ptwu for item, ptwu in tmp.items()}
        tmp = get_items_order(
            tmp_list_item, positive_items, negative_items, ptwus, supports
        )
        tmp = {item: tmp_list_item[item] for item in tmp}
        # # Process each item in reverse order
        for item, utility in reversed(list(tmp.items())):
            # Find or create the item in the EHMIN-list
            ehmin_item = ehmin_list.find_or_create(item, utility, pru=0)
            # Insert values into Ui.Tk vector
            ehmin_item.add_transaction_info(
                transaction["TID"], utility=utility, pru=rutil
            )

            # Update rutil if U(i) > 0
            # This make PRU wrong value!
            if utility > 0:
                ehmin_list.increase_pru(item, rutil)
                rutil += utility

        # Calculate EUCS[v_ik, v_jk] with PTU_k
        eucs = build_eucs(sorted_item)
        # for line in eucs:
        #     print(line)
        # return

    print("After 2nd scan", ehmin_list)

    # Step 3: Mining
    ehmin_mine(EHMINItem(), ehmin_list, set(), eucs, minU, sorted_item)
    for item in HUP:
        print(item, "-", HUP[item])
    return HUP


# Create an empty EHMINList
HUP = {}
ehmin_list = EHMINList()
ehmin(0.2)
