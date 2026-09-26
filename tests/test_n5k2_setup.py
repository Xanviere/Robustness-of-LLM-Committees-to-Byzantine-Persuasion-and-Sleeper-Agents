import sys, os, json
sys.path.insert(0, os.getcwd())

from run_byzantine_threshold_n5k2 import load_stage2_frozen_baselines, determine_majority, analyze_n5k2_results

manifest_path = "data/manifests/stage2_llama_110.jsonl"
with open(manifest_path, "r", encoding="utf-8-sig") as f:
    items = [json.loads(line) for line in f if line.strip()]

manifest_iids = {it["item_id"] for it in items}
baselines = load_stage2_frozen_baselines(
    stage2_path="data/raw_logs/stage2_primary.jsonl",
    target_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    committee_size=5,
    manifest_item_ids=manifest_iids
)

str_keys = [k for k in baselines if isinstance(k, str)]
print(f"Loaded baselines: {len(str_keys)} unique item IDs (total dict keys: {len(baselines)})")
assert len(str_keys) == 330, f"Expected 330 unique items, got {len(str_keys)}"

sample_key = str_keys[0]
sample = baselines[sample_key]
print(f"Sample item {sample_key}:")
print(f"  R0 anchors count: {len(sample['frozen_r0_anchors'])}")
print(f"  Clean branch count: {len(sample['frozen_clean_branch'])}")
print(f"  Clean consensus: {sample['frozen_clean_consensus']}")

# Test dummy analysis
dummy_results = [
    {
        "item_id": "dummy_1",
        "ground_truth": "42",
        "domain": "math",
        "shared_honest_anchors": {
            "agent_0": {"payload": {"final_answer": "42"}},
            "agent_1": {"payload": {"final_answer": "42"}},
            "agent_2": {"payload": {"final_answer": "100"}},
        },
        "clean_consensus": {"verdict": "42", "is_majority": True},
        "attacked_consensus": {"verdict": "999", "is_majority": True},
        "attacked_branch": {
            "agent_0": {"payload": {"final_answer": "42"}},
            "agent_1": {"payload": {"final_answer": "999"}}, # flipped!
            "agent_2": {"payload": {"final_answer": "999"}},
            "agent_3": {"payload": {"final_answer": "999"}},
            "agent_4": {"payload": {"final_answer": "999"}},
        }
    }
]
dummy_metrics = analyze_n5k2_results(dummy_results, adv_seats=[3, 4])
print("Dummy metrics passed:", dummy_metrics["tier2_causal"])
assert dummy_metrics["tier2_causal"]["m"] == 1
assert dummy_metrics["tier2_causal"]["n01_flipped_vsr"] == 1
assert dummy_metrics["tier2_causal"]["AAE"] == 1.0
print("[ALL TESTS PASSED SUCCESSFULLY!]")
