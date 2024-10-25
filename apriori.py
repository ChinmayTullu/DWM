import pandas as pd
from itertools import combinations

# Function to read dataset from CSV file
def read_dataset(file_path):
    return pd.read_csv(file_path)

# Function to convert the dataset into a list of transactions
def get_transactions(dataset):
    # Each row is treated as a transaction, and each column value is treated as an item.
    return dataset.apply(lambda row: list(row.dropna().astype(str)), axis=1).tolist()

# Function to generate candidate itemsets from previous itemsets
def generate_candidates(prev_itemsets, k):
    candidates = []
    n = len(prev_itemsets)
    for i in range(n):
        for j in range(i + 1, n):
            itemset1 = prev_itemsets[i]
            itemset2 = prev_itemsets[j]
            
            # Join only if the first (k-2) items are the same, to ensure candidates of size k
            if itemset1[:k-2] == itemset2[:k-2]:
                candidate = sorted(list(set(itemset1).union(set(itemset2))))
                if candidate not in candidates:
                    candidates.append(candidate)
    return candidates

# Function to prune itemsets that do not meet the minimum support threshold
def prune_itemsets(itemsets, transactions, min_support):
    pruned_itemsets = []
    item_counts = {}
    
    # Count occurrences of each itemset
    for itemset in itemsets:
        for transaction in transactions:
            if set(itemset).issubset(set(transaction)):
                item_counts[tuple(itemset)] = item_counts.get(tuple(itemset), 0) + 1

    # Calculate support and prune itemsets that do not meet the min_support
    total_transactions = len(transactions)
    for itemset, count in item_counts.items():
        support = count / total_transactions
        if support >= min_support:
            pruned_itemsets.append(list(itemset))
    
    return pruned_itemsets

# Main Apriori function to find frequent itemsets
def apriori(dataset, min_support):
    transactions = get_transactions(dataset)
    # Start with single-item itemsets
    itemsets = [[item] for item in set(item for transaction in transactions for item in transaction)]
    k = 2
    frequent_itemsets = []    
    
    # Iteratively find frequent itemsets of increasing size
    while itemsets:
        pruned_itemsets = prune_itemsets(itemsets, transactions, min_support)
        if not pruned_itemsets:
            break
        frequent_itemsets.extend(pruned_itemsets)
        # Generate next-level candidates
        itemsets = generate_candidates(pruned_itemsets, k)
        k += 1
    
    return frequent_itemsets

# Function to generate association rules from frequent itemsets
def generate_rules(frequent_itemsets, transactions, min_confidence):
    rules = []
    itemset_support = {}
    
    # Calculate support for each itemset to use in confidence calculation
    for itemset in frequent_itemsets:
        support = sum(1 for transaction in transactions if set(itemset).issubset(set(transaction)))
        itemset_support[frozenset(itemset)] = support / len(transactions)
    
    # Generate rules
    for itemset in frequent_itemsets:
        if len(itemset) > 1:
            itemset = frozenset(itemset)
            for antecedent_size in range(1, len(itemset)):
                for antecedent in combinations(itemset, antecedent_size):
                    antecedent = frozenset(antecedent)
                    consequent = itemset - antecedent
                    
                    if antecedent in itemset_support:
                        confidence = itemset_support[itemset] / itemset_support[antecedent]
                        if confidence >= min_confidence:
                            rules.append((set(antecedent), set(consequent), confidence))
    
    return rules

# Convert percentage input to fraction
def percentage_to_fraction(percentage):
    return percentage / 100.0

# Example usage
file_path = "weka-compatible-csv.csv"  # Replace with your CSV file path
dataset = read_dataset(file_path)
min_support = percentage_to_fraction(float(input("Enter the minimum support (as a percentage): ")))
min_confidence = percentage_to_fraction(float(input("Enter the minimum confidence (as a percentage): ")))

# Run Apriori to find frequent itemsets and generate rules
frequent_itemsets = apriori(dataset, min_support)
association_rules = generate_rules(frequent_itemsets, get_transactions(dataset), min_confidence)

# Display the results
print("\nFrequent Itemsets:")
for itemset in frequent_itemsets:
    print(itemset)

print("\nAssociation Rules:")
for rule in association_rules:
    antecedent, consequent, confidence = rule
    print(f"{antecedent} => {consequent}, Confidence: {confidence:.2%}")
