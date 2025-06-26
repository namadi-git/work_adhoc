#sample selection
from collections import defaultdict, Counter

# Step 1: Build mappings
id_to_topics = df_call_rcs.groupby('call_id')['call_rcs'].apply(set).to_dict()
topic_to_ids = defaultdict(set)

for call_id, topics in id_to_topics.items():
    for topic in topics:
        topic_to_ids[topic].add(call_id)

# Step 2: Initialize counters
topic_cover_count = Counter()  # counts how many call_ids cover each topic
selected_ids = set()

# Step 3: Greedy selection
while any(c < 2 for c in topic_cover_count.values() if c < 2 or topic_to_ids):

    # Score each call_id by how many *uncovered or under-covered* topics it helps
    id_scores = {}
    for call_id, topics in id_to_topics.items():
        if call_id in selected_ids:
            continue
        score = sum(1 for t in topics if topic_cover_count[t] < 2)
        if score > 0:
            id_scores[call_id] = score

    if not id_scores:
        break  # All topics are already covered twice

    # Pick call_id with highest score
    best_id = max(id_scores, key=id_scores.get)
    selected_ids.add(best_id)

    # Update cover count
    for t in id_to_topics[best_id]:
        topic_cover_count[t] += 1

# Final result
df_sample = df_call_rcs[df_call_rcs['call_id'].isin(selected_ids)].copy()




## version 2

import pandas as pd

# Assume df_call_rcs is already defined as per the user's description.
# For demonstration and testing, I'll create a sample df_call_rcs similar to the previous thought.
# In a real scenario, this step would be skipped if df_call_rcs is already provided.
data = {
    'call_id': [101, 101, 102, 101, 103, 103, 104, 105, 106, 106, 107, 108, 108, 109, 110, 110, 111, 111, 112, 112, 113, 114],
    'call_rcs': ['Billing', 'Support', 'Technical', 'Sales', 'Billing', 'Support', 'Technical', 'Sales', 'Billing', 'Support', 'Sales', 'Billing', 'Technical', 'Support', 'Sales', 'Billing', 'Sales', 'Support', 'Technical', 'Billing', 'Sales', 'Technical']
}
df_call_rcs = pd.DataFrame(data)

# Step 1: Get unique call topics
unique_topics = df_call_rcs['call_rcs'].unique()

# Step 2: Create a mapping of call_id to the topics it covers
# This gives us a dictionary where keys are call_ids and values are lists of topics
call_id_to_topics = df_call_rcs.groupby('call_id')['call_rcs'].apply(list).to_dict()

# Initialize tracking variables
sampled_call_ids = set()
# topic_coverage_count: keeps track of how many *distinct* sampled call_ids cover each topic
topic_coverage_count = {topic: 0 for topic in unique_topics}

# Required coverage per topic
required_coverage = 2

# Iteratively select call IDs using a greedy approach
# The loop continues as long as there's at least one topic that hasn't met the required coverage.
while any(count < required_coverage for count in topic_coverage_count.values()):
    best_call_id_candidate = None
    max_score = -1

    # Iterate through all call_ids that have not been sampled yet
    for current_call_id, covered_topics in call_id_to_topics.items():
        if current_call_id in sampled_call_ids:
            continue # Skip if already sampled

        current_score = 0
        topics_this_id_helps = []

        # Calculate a score for the current call_id
        # The score is based on how many "needed" coverages it provides.
        # We prioritize topics that are further from their required_coverage.
        for topic in covered_topics:
            if topic_coverage_count[topic] < required_coverage:
                # Add to score. The amount added could be 1, or (required_coverage - topic_coverage_count[topic])
                # Using 1 for now, as it effectively counts how many new 'slots' it fills.
                # A more advanced heuristic might weight by (required_coverage - current_count)
                current_score += 1
                topics_this_id_helps.append(topic)

        # If this call_id helps meet *any* requirements, consider it as a candidate
        if current_score > 0:
            # Selection criteria:
            # 1. Maximize the number of distinct topics it helps cover (current_score)
            # 2. If scores are equal, prioritize call_id that covers more total unique topics (len(covered_topics)) - this is a secondary heuristic for minimizing
            # 3. If still equal, prioritize by call_id itself (e.g., smaller ID first, using -current_call_id for descending order with max)
            if current_score > max_score:
                max_score = current_score
                best_call_id_candidate = current_call_id
            elif current_score == max_score:
                # Tie-breaking for secondary minimization: prefer IDs that cover more topics overall
                if len(covered_topics) > len(call_id_to_topics.get(best_call_id_candidate, [])):
                    best_call_id_candidate = current_call_id
                elif len(covered_topics) == len(call_id_to_topics.get(best_call_id_candidate, [])):
                    # Final tie-breaking: deterministic selection
                    if best_call_id_candidate is None or current_call_id < best_call_id_candidate:
                        best_call_id_candidate = current_call_id

    if best_call_id_candidate is None:
        # No more call_ids can help meet the remaining requirements.
        # This means it's impossible to satisfy the condition with the given data,
        # or a very specific edge case where only already selected IDs can further contribute
        # (which shouldn't happen with the `if current_call_id in sampled_call_ids: continue` check)
        break

    # Add the chosen call_id to the sample
    sampled_call_ids.add(best_call_id_candidate)

    # Update coverage counts for all topics this selected call_id covers
    for topic in call_id_to_topics[best_call_id_candidate]:
        # Only increment if the topic still needs coverage
        if topic_coverage_count[topic] < required_coverage:
            topic_coverage_count[topic] += 1

# Convert the set of sampled call_ids to a list for easier viewing/use
sampled_call_ids_list = list(sampled_call_ids)

# Filter the original DataFrame to get the sample based on the selected call_ids
sampled_df = df_call_rcs[df_call_rcs['call_id'].isin(sampled_call_ids_list)].copy()

# --- Verification Step (Optional, for internal check) ---
# Check if all topics meet the minimum coverage in the sampled_df
verification_counts = sampled_df.groupby('call_rcs')['call_id'].nunique()
all_topics_covered_as_required = True
missing_coverage_topics = {}

for topic in unique_topics:
    count = verification_counts.get(topic, 0)
    if count < required_coverage:
        all_topics_covered_as_required = False
        missing_coverage_topics[topic] = count

if not all_topics_covered_as_required:
    print("Warning: The sampling could not achieve the required coverage for some topics due to data limitations or greedy algorithm limitations.")
    print("Topics with insufficient coverage:", missing_coverage_topics)
    print("Attempted to sample, but some topics have less than 2 distinct call_ids in the result.")

print("Sampled Call IDs:", sampled_call_ids_list)
print("\nSampled DataFrame (first 5 rows):")
print(sampled_df.head())
print("\nSampled DataFrame info:")
print(sampled_df.info())

sampled_df.to_csv("sampled_call_rcs.csv", index=False)
print("\nSampled DataFrame saved to sampled_call_rcs.csv")
