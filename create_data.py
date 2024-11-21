import random

# Define possible items
items_list = ["a", "b", "c", "d", "e", "f", "g"]
profits = {"a": 2, "b": -3, "c": 4, "d": 2, "e": 6, "f": 7, "g": -5}

# Generate 500 transactions
dataset = []
for i in range(1, 10):
    # Each transaction has a unique ID
    transaction_id = f"T{i}"

    # Randomly choose between 2 and 7 items from the items_list
    items = random.sample(items_list, k=random.randint(2, 7))

    # Generate random quantities (between 1 and 5) and profits (between -2 and 4) for each item
    quantities = [random.randint(1, 5) for _ in items]
    profit = [profits[item] for item in items]
    print("items", items, profit)

    # Create the transaction and add it to the dataset
    transaction = {
        "TID": transaction_id,
        "items": items,
        "quantities": quantities,
        "profit": profit,
    }
    dataset.append(transaction)

# Display the first 5 transactions to check
print(dataset)
