# Demo
def read_data(file_name = "ehmintable.txt"):
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
    def __init__(self, item_name, utility=0, pru=0):
        self.item_name = item_name
        self.utility = utility
        self.pru = pru
        self.ti_vector = TIVector()

    def add_transaction_info(self, tid, utility, pru):
        self.ti_vector.add_transaction(tid, utility, pru)

    def __repr__(self):
        return (f"EHMINItem(Item: {self.item_name}, U: {self.utility}, PRU: {self.pru}, "
                f"{self.ti_vector})")

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
    max_idx = max(transaction["items"].index(item) for item in itemset if item in transaction["items"])
    
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
    if (len(itemset) == 0):
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
            return (1, rtwus.get(item, float('inf')))
        # If item is in negative_items (NI), sort by support (ascending)
        elif item in negative_items:
            return (2, supports.get(item, float('inf')))
        # Default priority for items not in PI or NI
        return (3, float('inf'))

    # Sort the items in itemset using the custom sort_key
    sorted_items = sorted(itemset, key=sort_key)
    return sorted_items


def rlu(X, z, dataset):
    """
    Calculate the Redefined Local Utility (RLU) for an itemset X and an item z.
    :param X: The base itemset.
    :param z: The item to be considered for RLU with X.
    :param dataset: The dataset containing transactions.
    :return: The RLU value.
    """
    rlu_value = 0

    # X union {z}
    extended_itemset = set(X).union({z})
    for transaction in dataset:
        # Check if the transaction contains all items in X ∪ {z}
        if extended_itemset.issubset(set(transaction["items"])):
            # Calculate u(X, T_k) and rru(X, T_k)
            # Use extended_itemset for the value of the transaction instead of X because of the definition
            u_X = utility_of_itemset(X, transaction)
            rru_X = redefined_remaining_utility(X, transaction)
            # Sum them up
            rlu_value += u_X + rru_X

    return rlu_value

def rsu(X, z, dataset):
    """
    Calculate the Redefined Sibling Utility (RSU) for an itemset X and an item z.
    :param X: The base itemset.
    :param z: The item to be considered for RSU with X.
    :param dataset: The dataset containing transactions.
    :return: The RSU value.
    """
    rsu_value = 0

    # X union {z}
    extended_itemset = set(X).union({z})

    for transaction in dataset:
        # Check if the transaction contains all items in X ∪ {z}
        if extended_itemset.issubset(set(transaction["items"])):
            # Calculate u(X, T_k), u(z, T_k), and rru(z, T_k)
            u_X = utility_of_itemset(X, transaction)
            u_z = utility_of_itemset([z], transaction)
            rru_z = redefined_remaining_utility(extended_itemset, transaction)
            rsu_value += u_X + u_z + rru_z

    return rsu_value

def process_and_remove_database(dataset, secondaryUnionη):
    # Process the dataset
    for transaction in dataset:
        # Filter items, quantities, and profit based on secondaryUnionη
        filtered_data = [
            (item, qty, prof)
            for item, qty, prof in zip(transaction["items"], transaction["quantities"], transaction["profit"])
            if item in secondaryUnionη
        ]
        
        # Unzip the filtered data back into items, quantities, and profit lists
        transaction["items"], transaction["quantities"], transaction["profit"] = map(list, zip(*filtered_data)) if filtered_data else ([], [], [])

    return dataset

def sort_transaction_items(order, dataset):
    priority = {item: i for i, item in enumerate(order)}

    # Function to sort items in each transaction based on the remaining order
    def process_transaction(transaction):
        # Zip items with quantities and profits, then sort based on item priority
        sorted_items = sorted(
            zip(transaction["items"], transaction["quantities"], transaction["profit"]),
            key=lambda x: priority.get(x[0], float('inf'))  # Use infinity if item is not in priority list
        )
        
        # Separate and calculate profits as profit * quantity
        sorted_items, quantities, profits = zip(*sorted_items)

        # Update the transaction with the sorted items and calculated profits
        return {
            "TID": transaction["TID"],
            "items": list(sorted_items),
            "profit": list(profits),
            "quantities": list(quantities)
        }

    # Process each transaction in the dataset and return the results
    return [process_transaction(transaction) for transaction in dataset]

def sort_transactions(transactions):
    # Create a sorting key that processes the items in reverse order for comparison
    def sort_key(transaction):
        # Create a tuple of the items in reverse order for ASCII comparison
        return tuple(reversed(transaction['items']))

    # Sort the transactions based on the created sorting key
    sorted_transactions = sorted(transactions, key=sort_key)
    return sorted_transactions[::-1]

def transaction_projection(transaction, itemset):
    """
    Project the given transaction using the specified itemset.
    
    :param transaction: A single transaction containing items and their quantities/profits.
    :param itemset: The itemset used for the projection.
    :return: A list of items that are in the transaction and come after the itemset, or an empty list if not all items are present.
    """
    projected_items = []
    projected_quantity = []
    projected_profit = []
    itemset_items = set(itemset)  # Convert itemset to a set for quick lookups

    # Check if all items in the itemset are present in the transaction
    if itemset_items.issubset(set(transaction['items'])):
        # Find the last index of the items in the itemset
        last_index = -1
        for item in transaction['items']:
            if item in itemset_items:
                last_index = transaction['items'].index(item)
        
        # Collect items after the last index of the itemset in the transaction
        if last_index != -1:
            projected_items = transaction['items'][last_index + 1:]
            projected_quantity = transaction['quantities'][last_index + 1:]
            projected_profit = transaction['profit'][last_index + 1:]

    return projected_items, projected_quantity, projected_profit


def database_projection(dataset, itemset):
    """
    Project the entire dataset using the specified itemset.
    
    :param dataset: The dataset containing all transactions.
    :param itemset: The itemset used for projecting the database.
    :return: A list of transactions projected by the itemset.
    """
    projected_dataset = []

    for transaction in dataset:
        projected_items, projected_quantity, projected_profit = transaction_projection(transaction, itemset)
        if projected_items:  # Only add non-empty projections
            projected_dataset.append({
                'TID': transaction['TID'],  # Keep transaction ID
                'items': projected_items,
                'quantities': projected_quantity,  # Optionally keep quantities or modify
                'profit': projected_profit,  # Optionally keep profit or modify
            })

    return projected_dataset

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
        if itemset_set.issubset(set(transaction['items'])):
            # Calculate utility for this transaction
            transaction_utility = 0
            
            # Calculate utility based on profit and quantities
            for item, quantity in zip(transaction['items'], transaction['quantities']):
                if item in itemset_set:
                    index = transaction['items'].index(item)
                    profit = transaction['profit'][index]
                    transaction_utility += profit * quantity
            
            utility += transaction_utility
    
    return utility


def searchN(negative_items, itemset, dataset, minU, sorted_dataset):
    """
    Search for high utility itemsets by appending items with negative utility to the given itemset.
    
    :param negative_items: Set of items with negative utility.
    :param itemset: The current itemset being evaluated.
    :param dataset: The dataset containing transactions.
    :param minU: The minimum utility threshold.
    """
    # Step 1: Iterate through each item in the set of negative items
    for i in negative_items:
        # Step 2: Create a new itemset β by adding the current negative item
        beta = itemset.union({i})
        
        # Step 3: Scan the dataset to calculate u(β) and create Dβ
        utility_beta = calculate_utility_and_dataset(beta, sorted_dataset)
        D_beta = database_projection(dataset, list(beta))

        # Step 4: Check if utility of β is greater than or equal to minU
        if utility_beta >= minU:
            # Step 5: Output the β itemset
            beta_str = ''.join(sorted(beta))
            if beta_str not in printed_itemsets:
                print(f"High utility itemset found (Negative): {beta_str}, Utility: {utility_beta}")
                printed_itemsets.add(beta_str)

        # Step 7: Calculate RSU(β, z) for all z ∈ η after i
        primary_beta = {z for z in negative_items if rsu(beta, z, D_beta) >= minU}

        # Step 10: Recursively call SearchN with updated primary items
        if primary_beta:
            searchN(primary_beta, beta, D_beta, minU, sorted_dataset)

def search(negative_items, itemset, dataset, primary_items, secondary_items, minU, sorted_dataset):
    """
    Search for high utility itemsets by appending positive utility items to the given itemset.
    
    :param negative_items: Set of items with negative utility.
    :param itemset: The current itemset being evaluated.
    :param dataset: The dataset containing transactions.
    :param primary_items: The primary items available for extension.
    :param secondary_items: The secondary items for RLU and RSU calculations.
    :param minU: The minimum utility threshold.
    """
    # Step 1: Iterate through each item in Primary(X)
    for i in primary_items:
        # Step 2: Create a new itemset β by adding the current primary item
        beta = set(itemset).union({i})
        
        # Step 3: Scan the dataset to calculate u(β) and create Dβ
        utility_beta = calculate_utility_and_dataset(beta, dataset)
        D_beta = database_projection(sorted_dataset, list(beta))

        # Step 4: Check if utility of β is greater than or equal to minU
        if utility_beta >= minU:
            # Step 5: Output the β itemset
            beta_str = ''.join(sorted(beta))
            if beta_str not in printed_itemsets:
                print(f"High utility itemset found (Negative): {beta_str}, Utility: {utility_beta}")
                printed_itemsets.add(beta_str)

        # Step 7: If utility of β is greater than minU, proceed with SearchN
        if utility_beta > minU:
            searchN(negative_items, beta, D_beta, minU, sorted_dataset)

        # # Step 10: Calculate RSU(β, z) and RLU(β, z) for all z ∈ Secondary(X)
        primary_beta = set()
        secondary_beta = set()
        for z in secondary_items:
            if z == i:
                continue
            rsu_value = rsu(beta, z, sorted_dataset)
            rlu_value = rlu(beta, z, sorted_dataset)

            # Step 11: Update Primary(β) based on RSU threshold
            if rsu_value >= minU:
                primary_beta = primary_beta.union({z})

            # Step 12: Update Secondary(β) based on RLU threshold
            if rlu_value >= minU:
                secondary_beta = secondary_beta.union({z})        

        # # Step 13: Recursive search call with updated β, dataset Dβ, primary and secondary items
        search(negative_items, beta, sorted_dataset, primary_beta, secondary_beta, minU, sorted_dataset)

def emhun(dataset, minU):
    X = []

    # Step 2-4: Identify p, s, and n
    ρ, η, δ = categorize_items(dataset)

    # Display results for the sets
    print("Positive Utility Only Items (ρ):", ρ)
    print("Negative Utility Only Items (η):", η)
    print("Mixed Utility Items (δ):", δ)
    
    # Step 5: Scan D to calculate RLU(X, i) for all item i ∈ ( ∪ ), using UA;
    secondary = set()
    rtwus = {}
    rlus = {}
    ρδunion = ρ | δ
    for i in ρδunion:
        rlu_value = rlu(X, i, dataset)
        iset = []
        iset.append(i)
        rtwu_value = calculate_rtwu(iset, dataset)
        rtwus[i] = rtwu_value
        if rlu_value >= minU:
            rlus[i] = rlu_value
            secondary.add(i)
    print("Secondary", secondary)
    
    # Step 7: The algorithm then sorts the elements into the order defined in Definition 7
    secondaryUnionη = secondary | η
    print("Secondary union η", secondaryUnionη)
    sorted_secondaryUnionη = get_items_order(secondaryUnionη, ρ, δ, η)
    print("Sorted secondary union η", sorted_secondaryUnionη)

    # Step 8: Scan D to remove item x not in (Secondary(X) ∪ η);
    removed_dataset = process_and_remove_database(dataset, secondaryUnionη)

    # Step 9: Sort the items in the remaining transactions in the order of items with positive utility only, items with both negative and positive utility, items with negative utility only;
    p, n, s = categorize_items(removed_dataset)
    remaining_transaction_sort_order = get_items_order(secondaryUnionη, p, s, n)
    print("Remaining transactions sort order: ", remaining_transaction_sort_order)
    sorted_item_dataset = sort_transaction_items(remaining_transaction_sort_order, dataset)
    print(sorted_item_dataset)
    
    # Step 10: Sort transactions in the database D
    # Sort the transactions based on the given rules
    sorted_dataset = sort_transactions(sorted_item_dataset)
    for transaction in sorted_dataset:
        print(transaction)
    
    # Step 11 and 12: Calculate RSU and Primary(X)
    primary = set()
    rsus = {}
    for i in secondary:
        rsu_value = rsu(X, i, sorted_dataset)
        iset = []
        iset.append(i)
        rsus[i] = rsu_value
        if rsu_value >= minU:
            primary.add(i)
    primary = get_items_order(primary, ρ, δ, η)
    print("Primary", primary)
    print("RSU", rsus)
    search(n, X, dataset, primary, secondary, minU, sorted_dataset)

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
    for item, quantity, profit in zip(transaction['items'], transaction['quantities'], transaction['profit']):
        if item in pattern and item in positive_items:
            utility = quantity * profit
            pu += utility  # Add to positive utility if item is in both the pattern and PI
    
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
            pru_x_tk = sum(profits[i] * quantities[i] for i in range(0, len(items)) if profits[i] > 0 and not set(items[i]).issubset(set(itemset)))
            
            # Add PRU(X, T_k) to the total PRU(X)
            pru += pru_x_tk

    return pru

def ehmin(δ):
    # Step 1: 1st Database Scan
    print(dataset)
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
    sorted_item = get_items_order(list_item, positive_items, negative_items, ptwus, supports)
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
        for item in transaction['items']:
            # Check PTWU(i) condition for pruning
            if ptwus[item] > minU:
                ptu_k += calculate_pu(set(item), transaction, positive_items)
        
        # Initialize a temporary map
        tmp = {}

        # # Step 2: Insert items into tmp and calculate PTWU if necessary
        for item, quantity, profit in zip(transaction['items'], transaction['quantities'], transaction['profit']):
            tmp[item] = quantity * profit  # Store internal utility and external utility

            
        # PTWU condition to recompute PTWU
            if ptwus[item] > minU:
                new_PTWIU = calculate_rtwu(set(item), dataset)
                ptwus[item] = new_PTWIU + ptu_k

        rutil = 0  # Initialize rutil
        # Sort to calculate PRU (inportant)
        tmp_list_item = {item: ptwu for item, ptwu in tmp.items()}
        tmp = get_items_order(tmp_list_item, positive_items, negative_items, ptwus, supports)
        tmp = {item: tmp_list_item[item] for item in tmp}
        # # Process each item in reverse order
        for item, utility in reversed(list(tmp.items())):
            # Find or create the item in the EHMIN-list
            ehmin_item = ehmin_list.find_or_create(item, utility, pru = 0)
            # Insert values into Ui.Tk vector
            ehmin_item.add_transaction_info(transaction["TID"], utility=utility, pru = rutil)
            
            # Update rutil if U(i) > 0
            # This make PRU wrong value!
            if utility > 0:
                ehmin_list.increase_pru(item, rutil)
                rutil += utility

        # Calculate EUCS[v_ik, v_jk] with PTU_k
        eucs = build_eucs(sorted_item)

    print("After 2nd scan", ehmin_list)
    # Step 3: Mining
    HUP = {}

# Create an empty EHMINList
ehmin_list = EHMINList()
ehmin(0.2)