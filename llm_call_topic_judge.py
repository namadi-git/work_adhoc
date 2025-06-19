# --- Adapted Step 1: Formalize & Refine Label Definitions ---
# We use the LLM to help us refine the definitions, making them more robust.

CALL_TOPIC_LABELS_HIGH_LEVEL = [
    "Billing Inquiry",
    "Claim Status",
    "Policy Coverage Question",
    "Provider Network Inquiry",
    "Medication Refill Request",
    "Prior Authorization Request",
    "Appointment Scheduling",
    "Member Portal Support",
    "Benefit Explanation",
    "Complaint/Feedback",
    "Enrollment/Disenrollment"
]

refine_definitions_prompt_template = PromptTemplate(
    template="""
You are an expert in health insurance customer service and data annotation.
Your task is to provide extremely clear, concise, and unambiguous definitions for the following call topic labels.
For each definition, explicitly state what *should* be included and what *should NOT* be included to avoid common misclassifications, especially for similar-sounding topics.
Think about the core intent of a customer calling for that specific topic.

Labels to Define:
{labels_list}

Provide the definitions in a JSON object where keys are the label names and values are their detailed definitions, including 'MUST INCLUDE' and 'MUST NOT INCLUDE' criteria.

Example Output Structure:
{{
  "LabelName": "Definition text. MUST INCLUDE: [keywords/intents]. MUST NOT INCLUDE: [keywords/intents for similar topics]."
}}
""",
    input_variables=["labels_list"]
)

labels_list_str = "\n".join([f"- {label}" for label in CALL_TOPIC_LABELS_HIGH_LEVEL])
formatted_prompt_definitions = refine_definitions_prompt_template.format(labels_list=labels_list_str)

print("--- Adapted Step 1: Refining Label Definitions via LLM ---")
print("Sending request to Gemini to get refined definitions...")

try:
    response = model.generate_content(formatted_prompt_definitions)
    refined_definitions_text = response.text.strip()
    # Attempt to parse the JSON output from Gemini
    json_start = refined_definitions_text.find('{')
    json_end = refined_definitions_text.rfind('}') + 1
    if json_start != -1 and json_end != -1:
        json_str = refined_definitions_text[json_start:json_end]
        LLM_REFINED_DEFINITIONS = json.loads(json_str)
        print("\nLLM-Refined Label Definitions:")
        for label, definition in LLM_REFINED_DEFINITIONS.items():
            print(f"- {label}: {definition}\n")
    else:
        print(f"Warning: Could not parse JSON from LLM response. Raw response: {refined_definitions_text[:500]}...")
        LLM_REFINED_DEFINITIONS = {} # Fallback
except Exception as e:
    print(f"Error refining definitions with LLM: {e}")
    LLM_REFINED_DEFINITIONS = {} # Fallback

# Use these LLM_REFINED_DEFINITIONS in subsequent steps where LABEL_DEFINITIONS were used
# If the LLM failed, fallback to the manual definitions
if not LLM_REFINED_DEFINITIONS:
    print("Falling back to pre-defined LABEL_DEFINITIONS for subsequent steps due to LLM failure.")
    LABEL_DEFINITIONS = {
        "Billing Inquiry": "A 'Billing Inquiry' label is correct if the call transcript predominantly discusses charges, invoices, payment history, deductibles, co-pays, premium amounts, or any discrepancies related to financial transactions concerning the insurance policy or medical services. MUST NOT INCLUDE: General claim processing questions without specific financial discrepancy.",
        "Claim Status": "A 'Claim Status' label is correct if the call transcript primarily focuses on inquiring about the current processing state of a submitted claim, whether it's pending, approved, denied, or paid, or requesting details about a specific claim number. MUST NOT INCLUDE: Questions about coverage or why a claim was denied (that's coverage), or direct billing issues.",
        "Policy Coverage Question": "A 'Policy Coverage Question' label is correct if the call transcript involves inquiries about what specific medical procedures, services, medications, or conditions are covered by the member's insurance plan, including benefit limits or exclusions. MUST NOT INCLUDE: General benefit explanations or claim processing status.",
        "Provider Network Inquiry": "A 'Provider Network Inquiry' label is correct if the call transcript is about finding in-network doctors, specialists, hospitals, or clinics, verifying a provider's network status, or understanding rules about out-of-network care. MUST NOT INCLUDE: Appointment scheduling for a known provider or general information about a type of doctor.",
        "Medication Refill Request": "A 'Medication Refill Request' label is correct if the customer is specifically asking to get a prescription refilled, enquiring about the process for refills, or checking the status of a refill order. MUST NOT INCLUDE: Questions about coverage for a *new* medication or its approval status (that's prior authorization).",
        "Prior Authorization Request": "A 'Prior Authorization Request' label is correct if the customer is asking about the need for, process of, or status of an authorization required before receiving certain medical services or medications. MUST NOT INCLUDE: General medication refills or coverage for already approved medications.",
        "Appointment Scheduling": "An 'Appointment Scheduling' label is correct if the call transcript involves the customer attempting to set up, reschedule, or cancel a medical appointment, or asking for assistance in doing so. MUST NOT INCLUDE: Finding a provider without intent to schedule, or general medical advice.",
        "Member Portal Support": "A 'Member Portal Support' label is correct if the customer is seeking help with technical issues or questions related to the online member portal, such as login problems, navigating features, or accessing documents. MUST NOT INCLUDE: General policy questions that happen to be asked while logged in.",
        "Benefit Explanation": "A 'Benefit Explanation' label is correct if the call transcript involves the customer requesting a general overview or detailed explanation of their insurance plan's benefits, such as annual maximums, out-of-pocket limits, or how different services are covered. MUST NOT INCLUDE: Specific policy coverage for a medical procedure (more specific 'Policy Coverage Question').",
        "Complaint/Feedback": "A 'Complaint/Feedback' label is correct if the customer expresses dissatisfaction, lodges a formal complaint about a service, process, or experience, or provides general feedback (positive or negative) about the insurance company. MUST NOT INCLUDE: Simple frustration during a transaction or inquiry, if the primary intent is not to complain about service itself.",
        "Enrollment/Disenrollment": "An 'Enrollment/Disenrollment' label is correct if the call transcript focuses on enrolling in a new insurance plan, changing plans, or canceling an existing insurance policy. MUST NOT INCLUDE: Questions about current plan benefits or features."
    }
else:
    LABEL_DEFINITIONS = LLM_REFINED_DEFINITIONS




# --- Adapted Step 2: Generate "Difficult Case" Examples (Synthetic Few-Shot) ---

generate_examples_prompt_template = PromptTemplate(
    template="""
You are an expert in health insurance customer service. Based on the provided label definitions, create synthetic, short call transcript snippets that are challenging for automated classification.
For each example, provide:
1.  A short, realistic 'transcript_text'.
2.  The 'model_predicted_label' (which should be a common False Positive for this scenario).
3.  The 'human_ground_truth_label' (what it *should* be).
4.  A concise 'reasoning' explaining *why* the model's prediction is a False Positive, referencing the provided definitions.

Focus on distinguishing between closely related labels or identifying subtle nuances.

**Call Topic Label Definitions:**
{label_definitions}

Generate 3 diverse examples. Provide the output as a JSON array of objects.

Example Output Structure for one object:
{{
  "transcript_text": "...",
  "model_predicted_label": "...",
  "human_ground_truth_label": "...",
  "reasoning": "..."
}}
""",
    input_variables=[],
    partial_variables={"label_definitions": "\n".join([f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items()])}
)

print("\n--- Adapted Step 2: Generating Synthetic Difficult Examples via LLM ---")
print("Sending request to Gemini to get synthetic examples...")

SYNTHETIC_DIFFICULT_EXAMPLES = []
try:
    response = model.generate_content(generate_examples_prompt_template.format())
    synthetic_examples_text = response.text.strip()
    json_start = synthetic_examples_text.find('[')
    json_end = synthetic_examples_text.rfind(']') + 1
    if json_start != -1 and json_end != -1:
        json_str = synthetic_examples_text[json_start:json_end]
        SYNTHETIC_DIFFICULT_EXAMPLES = json.loads(json_str)
        print("\nLLM-Generated Synthetic Difficult Examples (for LLM's own learning):")
        for ex in SYNTHETIC_DIFFICULT_EXAMPLES:
            print(f"  Transcript: '{ex['transcript_text']}'")
            print(f"    Predicted (FP): '{ex['model_predicted_label']}'")
            print(f"    Truth: '{ex['human_ground_truth_label']}'")
            print(f"    Reasoning: '{ex['reasoning']}'\n")
    else:
        print(f"Warning: Could not parse JSON from synthetic examples response. Raw response: {synthetic_examples_text[:500]}...")
except Exception as e:
    print(f"Error generating synthetic examples: {e}")

# This list of SYNTHETIC_DIFFICULT_EXAMPLES will now be used as "few-shot examples" in subsequent prompts.


# --- Adapted Step 3: Prompt Engineering for Targeted LLM Analysis ---

output_schema_str = """
{
  "transcript_id": "string",
  "predicted_labels_assessment": [
    {
      "label": "string",
      "is_correct": "boolean", // true if the predicted label is a True Positive, false if a False Positive
      "reasoning": "string" // Justification for the decision
    }
  ]
}
"""

# Format synthetic examples for the prompt
formatted_synthetic_examples_for_prompt = ""
if SYNTHETIC_DIFFICULT_EXAMPLES:
    for i, ex in enumerate(SYNTHETIC_DIFFICULT_EXAMPLES):
        formatted_synthetic_examples_for_prompt += f"""
--- Example {i+1} ---
Transcript Text: "{ex['transcript_text']}"
Model Predicted Label: "{ex['model_predicted_label']}"
Expected Correct Label: "{ex['human_ground_truth_label']}"
Explanation for why Predicted Label is a False Positive: "{ex['reasoning']}"
"""

targeted_analysis_prompt_template = PromptTemplate(
    template="""
You are an experienced healthcare insurance expert and quality assurance specialist. Your task is to meticulously evaluate the accuracy of an automated call topic label for a *specific* customer service transcript.

**Evaluation Goal:** For the provided *single* predicted label, determine if it is a 'True Positive' (truly applicable and correct for the transcript) or a 'False Positive' (incorrectly assigned).

**Call Topic Label Definitions:**
{label_definitions}

**Learning Examples (Study these patterns for common errors):**
{formatted_synthetic_examples}

**Instructions for Your Current Task:**
1.  Read the provided call transcript and the *single* predicted label carefully.
2.  Decide if this predicted label is truly supported by the content and main intent of the transcript, based on the definitions above.
    * If it is truly applicable, set "is_correct": true.
    * If it is not applicable or irrelevant to the transcript, set "is_correct": false.
3.  Provide a concise 'reasoning' for your decision (why it's correct or incorrect), referencing specific parts of the transcript if possible.
4.  Your output MUST be a valid JSON object, strictly following the schema below. Do NOT include any other text or formatting outside the JSON.

**JSON Output Schema:**
{output_schema}

**Call Transcript to Assess:**
Transcript ID: {transcript_id}
Transcript Text: {transcript_text}

**AI Model's Predicted Label to Assess:**
- {predicted_label_to_assess}

**Your Assessment (JSON ONLY):**
""",
    input_variables=["transcript_id", "transcript_text", "predicted_label_to_assess"],
    partial_variables={
        "label_definitions": "\n".join([f"- {k}: {v}" for k, v in LABEL_DEFINITIONS.items()]),
        "output_schema": output_schema_str,
        "formatted_synthetic_examples": formatted_synthetic_examples_for_prompt
    }
)

print("\n--- Adapted Step 3: Targeted Prompt Template Defined ---")
# Example usage (not executed here, but for illustration)
# sample_prompt_for_single_case = targeted_analysis_prompt_template.format(
#     transcript_id="T_Problem_001",
#     transcript_text="Customer: I got a bill for $500 that says 'denied' for my MRI. I thought I had 80% coverage after deductible.",
#     predicted_label_to_assess="Claim Status" # Example of a potentially incorrect prediction
# )
# print(sample_prompt_for_single_case[:1000]) # Display part of it


# --- Adapted Step 4: Ad-hoc Qualitative Review of Actual Problematic Cases ---

print("\n--- Adapted Step 4: Performing Ad-hoc Qualitative Review (Manual Process in Practice) ---")
print("In a real scenario, you'd identify actual False Positive cases from your XLNet model.")

# Hypothetical problematic case (imagine this is a real FP from your XLNet model)
problematic_transcript_id = "P001"
problematic_transcript_text = "Hello, I just need to confirm if my visit to the chiropractor last week is covered under my plan. I know you cover physical therapy, but I'm not sure about chiro."
model_predicted_label_for_problematic_case = "Benefit Explanation" # XLNet model predicted this
# Human ground truth would be "Policy Coverage Question" (more specific than general benefit explanation)

print(f"\nAssessing a hypothetical problematic case (ID: {problematic_transcript_id}):")
print(f"  Transcript: {problematic_transcript_text}")
print(f"  XLNet Model Predicted: '{model_predicted_label_for_problematic_case}' (Suspected False Positive)")

llm_individual_judgment = {}
try:
    formatted_prompt_for_problem_case = targeted_analysis_prompt_template.format(
        transcript_id=problematic_transcript_id,
        transcript_text=problematic_transcript_text,
        predicted_label_to_assess=model_predicted_label_for_problematic_case
    )
    response = model.generate_content(formatted_prompt_for_problem_case)
    response_text = response.text.strip()
    json_start = response_text.find('{')
    json_end = response_text.rfind('}') + 1
    if json_start != -1 and json_end != -1:
        json_str = response_text[json_start:json_end]
        llm_individual_judgment = json.loads(json_str)
        print("\nLLM's Targeted Judgment:")
        print(json.dumps(llm_individual_judgment, indent=2))
    else:
        print(f"Warning: Could not parse JSON for problematic case. Raw response: {response_text[:500]}...")
except Exception as e:
    print(f"Error assessing problematic case with LLM: {e}")


# --- Adapted Step 5: No Direct Batch Precision Calculation ---

print("\n--- Adapted Step 5: No Batch Precision Calculation (Focus on Qualitative) ---")
print("In this context-limited approach, we are NOT calculating full precision scores directly from the LLM.")
print("The LLM is used for deep qualitative analysis of *specific* problematic instances (False Positives).")
print("Quantitative precision metrics (Precision, Recall, F1) should still be calculated on a smaller, fixed human-labeled test set.")



# --- Adapted Step 6: Analyze and Interpret Results (Focus on Insights) ---

print("\n--- Adapted Step 6: Analyzing and Interpreting Results (Focus on Insights) ---")

print("\n### Quantitative Context (from your internal fixed test set) ###")
print("Assume your internal metrics show, for example:")
print("- Overall Macro-Precision: 0.85")
print("- Precision for 'Benefit Explanation': 0.70 (indicating a problem area)")
print("- Precision for 'Prior Authorization Request': 0.72 (another problem area)")
print("\nOur LLM-based analysis focuses on understanding *why* these precision scores are what they are, by drilling into specific False Positives.")


print("\n### Qualitative Analysis from Targeted LLM Judgments ###")

if llm_individual_judgment and "error" not in llm_individual_judgment:
    print("\nDetailed Analysis of Problematic Case (P001):")
    for assessment in llm_individual_judgment['predicted_labels_assessment']:
        print(f"  Predicted Label: '{assessment['label']}'")
        print(f"  LLM Judgment: {'True Positive' if assessment['is_correct'] else 'False Positive'}")
        print(f"  LLM Reasoning: {assessment['reasoning']}")

        # Compare LLM judgment to our manual expectation for this problematic case
        if assessment['label'] == model_predicted_label_for_problematic_case:
            if not assessment['is_correct']:
                print(f"  **ACTIONABLE INSIGHT:** The LLM confirmed our suspicion: '{model_predicted_label_for_problematic_case}' is indeed a False Positive for this transcript.")
                print(f"  **Specific Reason from LLM:** The LLM's reasoning highlights that the question is about 'coverage under my plan' for a specific service ('chiropractor'), which points more directly to 'Policy Coverage Question' rather than a general 'Benefit Explanation'. Our XLNet model likely missed this specificity.")
            else:
                print(f"  **SURPRISING INSIGHT:** The LLM actually deemed '{model_predicted_label_for_problematic_case}' to be correct. This suggests a potential disagreement with our initial human assessment or a nuance we missed. We need to review this particular case carefully with a human expert and potentially refine label definitions.")
else:
    print("\nNo specific problematic case judgment to analyze or an error occurred.")

print("\n### Actionable Insights and Next Steps (General, informed by LLM's 'consultation') ###")
print("Based on the refined label definitions and synthetic examples from the LLM, as well as any ad-hoc problematic case analysis:")

print("\n1. **Refine XLNet Model Training Data (Crucial):**")
print("   - **Leverage LLM-Refined Definitions:** Use the precise 'MUST INCLUDE'/'MUST NOT INCLUDE' criteria from Step 1 to guide your human data annotators for future labeling efforts. This ensures higher consistency and accuracy in your ground truth.")
print("   - **Synthesize More Training Data:** The `SYNTHETIC_DIFFICULT_EXAMPLES` (from Step 2) provide patterns of common errors. Use these patterns to either: (a) Manually create more real-world-like training examples for these tricky distinctions, or (b) Use the LLM *itself* (with careful prompting and validation) to generate diverse training examples that emphasize these distinctions.")
print("   - **Focus on Ambiguous Cases:** Target collecting more real transcripts that fall into the nuanced categories where XLNet currently underperforms (e.g., distinguishing 'Benefit Explanation' from 'Policy Coverage Question', or 'Medication Refill' from 'Prior Authorization').")

print("\n2. **Improve XLNet Model Architecture/Fine-tuning:**")
print("   - If patterns of misclassification persist even with better data, consider deeper analysis of XLNet's feature extraction. Are there specific tokens or contextual clues it's missing for these nuanced labels?")
print("   - Experiment with different loss functions or weighting strategies during XLNet fine-tuning to penalize misclassifications of critical or frequently confused labels more heavily.")

print("\n3. **Post-processing and Rule-based Enhancements:**")
print("   - For recurring and highly specific False Positives (like the 'chiropractor coverage' vs 'general benefit' example), consider implementing small, confidence-based rules post-XLNet prediction. E.g., if 'Benefit Explanation' is predicted with high confidence but phrases like 'is covered for' or 'denied for' are present, cross-check against 'Policy Coverage Question'.")

print("\n4. **Continuous Monitoring & Iterative Improvement:**")
print("   - Maintain a small, high-quality, human-labeled test set to track quantitative precision metrics over time. This is your true north for overall performance.")
print("   - Periodically revisit the LLM for 'consultation': Ask it to generate *new* synthetic difficult examples, or analyze new types of recurring False Positives identified by your internal metrics. This allows the LLM to remain a dynamic source of insight without constantly running large evaluation sets.")
print("   - Re-evaluate the LLM as a judge: Ensure its judgments align with human expert consensus on a small validation set to prevent drift in the LLM's understanding.")

print("\nBy using the LLM as a sophisticated consultant rather than a bulk evaluator, we can glean critical insights for model improvement while staying within context and cost limits.")

