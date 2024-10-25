# Training Data (Categorical) - Represented as tuples
data = [
    (1, 'Sunny', 'Hot', 'High', 'Weak', 'No'),
    (2, 'Sunny', 'Hot', 'High', 'Strong', 'No'),
    (3, 'Overcast', 'Hot', 'High', 'Weak', 'Yes'),
    (4, 'Rain', 'Mild', 'High', 'Weak', 'Yes'),
    (5, 'Rain', 'Cool', 'Normal', 'Weak', 'Yes'),
    (6, 'Rain', 'Cool', 'Normal', 'Strong', 'No'),
    (7, 'Overcast', 'Cool', 'Normal', 'Strong', 'Yes'),
    (8, 'Sunny', 'Mild', 'High', 'Weak', 'No'),
    (9, 'Sunny', 'Cool', 'Normal', 'Weak', 'Yes'),
    (10, 'Rain', 'Mild', 'Normal', 'Weak', 'Yes'),
    (11, 'Sunny', 'Mild', 'Normal', 'Strong', 'Yes'),
    (12, 'Overcast', 'Mild', 'High', 'Strong', 'Yes'),
    (13, 'Overcast', 'Hot', 'Normal', 'Weak', 'Yes'),
    (14, 'Rain', 'Mild', 'High', 'Strong', 'No')
]

# Extract Features and Labels Separately
features = [row[1:-1] for row in data]  # Exclude ID and Label
labels = [row[-1] for row in data]      # Only Labels ('Yes'/'No')

# Step 1: Calculate Priors P(Yes) and P(No)
total_samples = len(labels)
yes_count = labels.count('Yes')
no_count = labels.count('No')

P_yes = yes_count / total_samples
P_no = no_count / total_samples

print(f"P(Yes) = {P_yes}")
print(f"P(No) = {P_no}")

# Step 2: Define a function to calculate likelihoods P(Feature | Label)
def calculate_likelihood(feature_idx, feature_value, label_value):
    """Count occurrences of feature_value for a given label (Yes/No)."""
    count = sum(1 for i in range(total_samples) 
                if features[i][feature_idx] == feature_value and labels[i] == label_value)
    return count / labels.count(label_value)

# Step 3: Predict for a New Tuple
new_tuple = ('Sunny', 'Cool', 'High', 'Strong')  # Input to be predicted

# Calculate Posterior for 'Yes'
posterior_yes = P_yes
for idx, value in enumerate(new_tuple):
    posterior_yes *= calculate_likelihood(idx, value, 'Yes')

# Calculate Posterior for 'No'
posterior_no = P_no
for idx, value in enumerate(new_tuple):
    posterior_no *= calculate_likelihood(idx, value, 'No')

print(f"Posterior Probability for Yes = {posterior_yes}")
print(f"Posterior Probability for No = {posterior_no}")

# Step 4: Final Prediction
if posterior_yes > posterior_no:
    print("Prediction: Yes")
else:
    print("Prediction: No")
