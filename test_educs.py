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

# Example dataset
dataset = [
    {
        "TID": "T1",
        "items": ["a", "b", "d", "h"],
        "quantities": [2, 3, 1, 1],
        "profit": [2, 1, 3, -1],
    },
    {"TID": "T2", "items": ["a", "c", "e", "h"], "quantities": [2, 4, 2, 3], "profit": [2, 1, -1, -1]},
    {
        "TID": "T3",
        "items": ["b", "c", "d", "e", "f"],
        "quantities": [6, 3, 1, 3, 2],
        "profit": [1, 1, 3, -1, 5],
    },

    {
        "TID": "T4",
        "items": ["a", "b", "c", "g"],
        "quantities": [4, 3, 3, 2],
        "profit": [2, 1, 1, -1],
    },
    {"TID": "T5", "items": ["b", "d", "e", "g" ,"h"], "quantities": [4, 4, 1, 2, 1], "profit": [1, 3, -1, -1, -1]},
]

# Example itemset
itemset = ["a", "c"]

# Calculate PRU for the itemset in the dataset
pru_result = calculate_pru(itemset, dataset)
print("PRU(", itemset, ") =", pru_result)
