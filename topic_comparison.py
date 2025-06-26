import pandas as pd
from collections import defaultdict
from sentence_transformers import SentenceTransformer, util
import torch

# --- Simulate DataFrames (as described previously) ---
data_model = {
    'call_id': ['C001', 'C001', 'C001', 'C002', 'C002', 'C003', 'C003', 'C004', 'C004'],
    'chunk_idx': [0, 0, 1, 0, 1, 0, 0, 0, 1],
    'model_topic': ['Claim Status Inquiry', 'Benefit Clarification', 'General Inquiry',
                    'Prior Authorization', 'Claim Status Inquiry', 'General Inquiry',
                    'Complaint', 'Technical Support', 'General Inquiry']
}
df_model_topics = pd.DataFrame(data_model)

data_gemini = {
    'call_id': ['C001', 'C001', 'C001', 'C002', 'C002', 'C003', 'C003', 'C004', 'C004'],
    'chunk_idx': [0, 0, 1, 0, 1, 0, 0, 0, 1],
    'gemini_topic': ['Patient Eligibility Verification', 'Out-of-Pocket Cost Inquiry', 'Provider Credentialing Status Check',
                     'New Prior Authorization Submission', 'Claim Denial Reason Explanation', 'Billing Error Resolution',
                     'EDI Connectivity Issue', 'Provider Portal Login Assistance', 'General Administrative Question']
}
df_gemini_topics = pd.DataFrame(data_gemini)


# --- Load a pre-trained Sentence Transformer model ---
# 'all-MiniLM-L6-v2' is a good balance of speed and performance.
# You might consider larger models like 'all-mpnet-base-v2' for higher accuracy
# if computational resources allow.
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Helper function to get embeddings for a list of topics ---
def get_embeddings(topics):
    if not topics:
        return []
    # Ensure topics are unique before embedding to avoid redundant computation
    unique_topics = list(set(topics))
    embeddings = model.encode(unique_topics, convert_to_tensor=True)
    return {topic: embedding for topic, embedding in zip(unique_topics, embeddings)}

# --- Aggregate topics into sets for each chunk ---
def aggregate_topics_and_get_embeddings(df, topic_col_name):
    aggregated_topics = defaultdict(set)
    for _, row in df.iterrows():
        key = (row['call_id'], row['chunk_idx'])
        aggregated_topics[key].add(row[topic_col_name])

    rows = []
    for (call_id, chunk_idx), topics_set in aggregated_topics.items():
        topics_list = list(topics_set)
        embeddings_dict = get_embeddings(topics_list)
        rows.append({
            'call_id': call_id,
            'chunk_idx': chunk_idx,
            f'{topic_col_name}_list': topics_list, # Keep as list for iteration
            f'{topic_col_name}_embeddings_dict': embeddings_dict # Store dict of topic:embedding
        })
    return pd.DataFrame(rows)

df_model_agg = aggregate_topics_and_get_embeddings(df_model_topics, 'model_topic')
df_gemini_agg = aggregate_topics_and_get_embeddings(df_gemini_topics, 'gemini_topic')

# Merge the aggregated dataframes
df_combined_sem = pd.merge(df_model_agg, df_gemini_agg, on=['call_id', 'chunk_idx'], how='outer')

# Fill NaN values for topics/embeddings if a chunk only exists in one source
df_combined_sem['model_topic_list'].fillna('', inplace=True)
df_combined_sem['gemini_topic_list'].fillna('', inplace=True)
df_combined_sem['model_topic_embeddings_dict'].fillna({}, inplace=True)
df_combined_sem['gemini_topic_embeddings_dict'].fillna({}, inplace=True)

# Ensure empty lists/dicts if NaN occurred after fillna
df_combined_sem['model_topic_list'] = df_combined_sem['model_topic_list'].apply(lambda x: x if isinstance(x, list) else [])
df_combined_sem['gemini_topic_list'] = df_combined_sem['gemini_topic_list'].apply(lambda x: x if isinstance(x, list) else [])
df_combined_sem['model_topic_embeddings_dict'] = df_combined_sem['model_topic_embeddings_dict'].apply(lambda x: x if isinstance(x, dict) else {})
df_combined_sem['gemini_topic_embeddings_dict'] = df_combined_sem['gemini_topic_embeddings_dict'].apply(lambda x: x if isinstance(x, dict) else {})


print("--- Combined DataFrame with Aggregated Topics and Embeddings ---")
print(df_combined_sem.head())
print("\n" + "="*80 + "\n")




# --- Semantic Similarity Threshold ---
# This is a critical hyperparameter. You'll need to tune this!
# A good starting point is usually between 0.6 and 0.8.
SEMANTIC_SIMILARITY_THRESHOLD = 0.75

# --- Function to find semantic matches between two sets of topics ---
def find_semantic_matches(model_topics_dict, gemini_topics_dict, threshold=SEMANTIC_SIMILARITY_THRESHOLD):
    matched_pairs = []
    model_matched = set()
    gemini_matched = set()

    for m_topic, m_embed in model_topics_dict.items():
        for g_topic, g_embed in gemini_topics_dict.items():
            # Calculate cosine similarity between embeddings
            similarity = util.cos_sim(m_embed, g_embed).item()
            if similarity >= threshold:
                matched_pairs.append((m_topic, g_topic, similarity))
                model_matched.add(m_topic)
                gemini_matched.add(g_topic)
    
    # Sort matches by similarity in descending order
    matched_pairs.sort(key=lambda x: x[2], reverse=True)
    return matched_pairs, model_matched, gemini_matched

# --- Apply semantic matching to each chunk ---
semantic_matches_data = []
for index, row in df_combined_sem.iterrows():
    model_topics_dict = row['model_topic_embeddings_dict']
    gemini_topics_dict = row['gemini_topic_embeddings_dict']

    matches, model_matched_set, gemini_matched_set = \
        find_semantic_matches(model_topics_dict, gemini_topics_dict, SEMANTIC_SIMILARITY_THRESHOLD)

    # These are topics the model predicted that had *no* semantic match with any Gemini topic
    model_only_sem = [t for t in row['model_topic_list'] if t not in model_matched_set]
    # These are topics Gemini identified that had *no* semantic match with any model topic
    gemini_only_sem = [t for t in row['gemini_topic_list'] if t not in gemini_matched_set]
    
    semantic_matches_data.append({
        'call_id': row['call_id'],
        'chunk_idx': row['chunk_idx'],
        'model_topics': row['model_topic_list'],
        'gemini_topics': row['gemini_topic_list'],
        'semantic_matched_pairs': matches,
        'model_only_semantic_topics': model_only_sem,
        'gemini_only_semantic_topics': gemini_only_sem,
        'num_semantic_matches': len(matches)
    })

df_semantic_comparison = pd.DataFrame(semantic_matches_data)

print("--- Semantic Comparison DataFrame (with example for C001, chunk 0) ---")
print(df_semantic_comparison.loc[(df_semantic_comparison['call_id'] == 'C001') & (df_semantic_comparison['chunk_idx'] == 0)])
print("\n" + "="*80 + "\n")



from collections import Counter

# --- Analyze Semantically Missing Topics (Gemini Only) ---
print("--- Analysis of Topics Present ONLY in Gemini's Output (Semantically Missing from Model) ---")
all_gemini_only_sem_topics = [topic for sublist in df_semantic_comparison['gemini_only_semantic_topics'] for topic in sublist]

if all_gemini_only_sem_topics:
    missing_sem_topics_counts = Counter(all_gemini_only_sem_topics)
    print("Most frequent topics identified by Gemini but NOT semantically matched by the current model:")
    for topic, count in missing_sem_topics_counts.most_common(10): # Show top 10
        print(f"- '{topic}': {count} occurrences")
else:
    print("No unique topics found in Gemini's output that were not semantically matched by the model.")

print("\n" + "="*80 + "\n")

# --- Analyze Model's Overly General Topics (or Mismatches) ---
print("--- Analysis of Model's Overly General Topics or Semantic Mismatches ---")

# Strategy: Find chunks where the number of semantic matches is low, relative to total topics,
# and where the model outputted a broad topic.
# Reuse or redefine your 'broad_model_topics' based on your model's labels.
broad_model_topics_list = ['Claim Status Inquiry', 'Benefit Clarification', 'General Inquiry', 'Complaint', 'Technical Support']

# Filter chunks where:
# 1. There are few or no semantic matches
# 2. Gemini has identified topics
# 3. Model also has topics
# 4. At least one model topic is considered "broad"
df_semantic_mismatches = df_semantic_comparison[
    (df_semantic_comparison['num_semantic_matches'] < 1) & # Few/no direct semantic matches
    (df_semantic_comparison['gemini_topics'].apply(lambda x: len(x) > 0)) & # Gemini has topics
    (df_semantic_comparison['model_topics'].apply(lambda x: len(x) > 0)) & # Model has topics
    (df_semantic_comparison['model_topics'].apply(lambda x: any(t in broad_model_topics_list for t in x)))
].copy()

if not df_semantic_mismatches.empty:
    print("Chunks where model topics are likely too broad or misclassified (low semantic match):")
    for index, row in df_semantic_mismatches.head(5).iterrows():
        print(f"\nCall ID: {row['call_id']}, Chunk ID: {row['chunk_idx']}")
        print(f"  Model Topics: {row['model_topics']}")
        print(f"  Gemini Topics: {row['gemini_topics']}")
        print(f"  Semantic Matched Pairs: {row['semantic_matched_pairs']}")
        print(f"  Model Only Topics (Sem.): {row['model_only_semantic_topics']}")
        print(f"  Gemini Only Topics (Sem.): {row['gemini_only_semantic_topics']}")
    if len(df_semantic_mismatches) > 5:
        print(f"\n... (and {len(df_semantic_mismatches) - 5} more instances)")

    # Aggregate which broad model topics are most frequently associated with semantic mismatches
    all_model_topics_in_semantic_disagreement = []
    for index, row in df_semantic_mismatches.iterrows():
        all_model_topics_in_semantic_disagreement.extend(
            [t for t in row['model_topics'] if t in broad_model_topics_list]
        )
    if all_model_topics_in_semantic_disagreement:
        model_disagreement_counts = Counter(all_model_topics_in_semantic_disagreement)
        print("\nBroad model topics most frequently associated with semantic disagreements:")
        for topic, count in model_disagreement_counts.most_common(5):
            print(f"- '{topic}': {count} occurrences")
else
