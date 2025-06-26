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
