import random

# Define possible items
items_list = ["a", "b", "c", "d", "e", "f", "g"]

# Generate 500 transactions
dataset = []
for i in range(1, 501):
    # Each transaction has a unique ID
    transaction_id = f"T{i}"
    
    # Randomly choose between 2 and 7 items from the items_list
    items = random.sample(items_list, k=random.randint(2, 7))
    
    # Generate random quantities (between 1 and 5) and profits (between -2 and 4) for each item
    quantities = [random.randint(1, 5) for _ in items]
    profit = [random.randint(-2, 4) for _ in items]
    
    # Create the transaction and add it to the dataset
    transaction = {
        "TID": transaction_id,
        "items": items,
        "quantities": quantities,
        "profit": profit
    }
    dataset.append(transaction)

# Display the first 5 transactions to check
print(dataset)
