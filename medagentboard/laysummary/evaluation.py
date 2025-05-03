import os
import json
import numpy as np
import nltk
import evaluate
from transformers import set_seed
import argparse
from rouge_score import rouge_scorer
import textstat
import random

# Make sure NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

# ========================================================================
#  Keep all your functions (bootstrap, calc_rouge, get_metrics, etc.)
#  exactly as they were in your *original* script, with the
#  small modification to bootstrap to use 'id'.
# ========================================================================

def bootstrap(data_path):
    """
    Bootstrap the sample ids with replacement, the size of the sample is the same as the original data.
    --- MODIFIED TO USE 'id' key ---
    """
    data_ids = [] # Changed from data_qids
    try:
        with open(data_path, "r", encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, list):
                print(f"Error: Expected a list in {data_path}, found {type(data)}")
                return [] # Return empty list on error
            for datum in data:
                 # Check if datum is a dict and has the 'id' key
                 if isinstance(datum, dict) and "id" in datum:
                    data_ids.append(datum["id"]) # Use 'id' key
                 else:
                    print(f"Warning: Skipping item in {data_path} missing 'id' or not a dict: {datum}")

    except FileNotFoundError:
        print(f"Error: Bootstrap data file not found: {data_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {data_path}")
        return []
    except Exception as e:
        print(f"Error reading bootstrap file {data_path}: {e}")
        return []

    if not data_ids:
        print(f"Warning: No IDs found in {data_path}")
        return []

    # Use the global SEED for reproducibility if desired, or manage state differently
    # Ensure numpy random state is consistent if used repeatedly
    # np.random.seed(SEED) # You might need to pass SEED or handle this globally
    return [data_ids[i] for i in np.random.randint(0, len(data_ids), len(data_ids))]


def calc_rouge(preds, refs):
  # Get ROUGE F1 scores
  scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], \
                                    use_stemmer=True, split_summaries=True)
  # Ensure refs is a list of strings, not list of lists if only one reference
  processed_refs = [ref[0] if isinstance(ref, list) else ref for ref in refs]
  scores = []
  for i, p in enumerate(preds):
      try:
          # Basic check for empty strings
          if not p or not processed_refs[i]:
              scores.append({'rouge1': rouge_scorer.Score(0,0,0),
                             'rouge2': rouge_scorer.Score(0,0,0),
                             'rougeLsum': rouge_scorer.Score(0,0,0)})
          else:
              scores.append(scorer.score(p, processed_refs[i]))
      except Exception as e:
          print(f"Warning: ROUGE calculation error for item {i}: {e}. Assigning 0.")
          scores.append({'rouge1': rouge_scorer.Score(0,0,0),
                         'rouge2': rouge_scorer.Score(0,0,0),
                         'rougeLsum': rouge_scorer.Score(0,0,0)}) # Assign 0 score on error
  # Handle case where no scores were calculated
  if not scores: return 0.0, 0.0, 0.0
  # Original script didn't multiply by 100, reverting that too
  return np.mean([s['rouge1'].fmeasure for s in scores]), \
         np.mean([s['rouge2'].fmeasure for s in scores]), \
         np.mean([s['rougeLsum'].fmeasure for s in scores])

# def calc_bertscore(preds, refs): # Your original commented out function
#   # Get BERTScore F1 scores
#   P, R, F1 = score(preds, refs, lang="en", verbose=True, device='cuda:0')
#   return np.mean(F1.tolist())

def calc_readability(preds):
  fkgl_scores = []
  cli_scores = []
  dcrs_scores = []
  for pred in preds:
    try:
        # Handle potential empty strings for textstat
        if not pred or not pred.strip():
             fkgl_scores.append(0) # Assign a default, e.g., 0 or handle as needed
             cli_scores.append(0)
             dcrs_scores.append(0)
        else:
             fkgl_scores.append(textstat.flesch_kincaid_grade(pred))
             cli_scores.append(textstat.coleman_liau_index(pred))
             dcrs_scores.append(textstat.dale_chall_readability_score(pred))
    except Exception as e:
        print(f"Warning: Readability calculation error: {e}. Assigning 0 score.")
        fkgl_scores.append(0)
        cli_scores.append(0)
        dcrs_scores.append(0)
  if not fkgl_scores: return 0.0, 0.0, 0.0 # Handle empty input
  return np.mean(fkgl_scores), np.mean(cli_scores), np.mean(dcrs_scores)


def get_metrics(preds, goldens, sources, seed):
    # Set reproducibility (as in your original)
    SEED = seed
    os.environ['PYTHONHASHSEED']=str(SEED)
    # Note: set_seed affects transformers randomness, numpy affects np.random, random affects random module
    random.seed(SEED)
    np.random.seed(SEED)
    set_seed(SEED)

    if not preds or not goldens or not sources:
        print("Warning: Empty input list(s) provided to get_metrics. Returning zero scores.")
        # Return structure matching original expectations (14 zeros)
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    # SARI requires tokenized inputs (using original formatting style)
    sari_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    sari_refs = [[" ".join(nltk.sent_tokenize(golden.strip()))] for golden in goldens]
    sari_sources = sources # Original used sources directly, keep that way unless tokenization is strictly needed by evaluate
    sari_score = 0
    try:
        metric = evaluate.load('sari', seed=SEED) # Keep seed=SEED if original had it
        sari_result = metric.compute(sources=sari_sources, predictions = sari_preds, references = sari_refs)
        sari_score = sari_result['sari']
    except Exception as e:
        print(f"Error calculating SARI: {e}. Assigning 0.")

    # ROUGE uses sentence tokenization and newlines (original formatting style)
    rouge_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    rouge_refs = ["\n".join(nltk.sent_tokenize(golden.strip())) for golden in goldens]
    rouge1, rouge2, rougeL = calc_rouge(rouge_preds, rouge_refs)

    # SACREBLEU needs tokenized inputs, references as list of lists (original formatting style)
    bleu_preds = [" ".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
    bleu_refs = [[" ".join(nltk.sent_tokenize(golden.strip()))] for golden in goldens]
    bleu_score = 0
    try:
        metric = evaluate.load('sacrebleu', seed=SEED) # Keep seed=SEED if original had it
        bleu_result = metric.compute(predictions=bleu_preds, references=bleu_refs)
        bleu_score = bleu_result['score'] # Original script returned 'score'
    except Exception as e:
        print(f"Error calculating BLEU: {e}. Assigning 0.")

    # BertScore (if used)
    # bertscore = calc_bertscore(...) # Keep commented as original

    # Readability for (1) sources, (2) goldens, and (3) model outputs (use original texts)
    fkgl_abs, cli_abs, dcrs_abs = calc_readability(sources)
    fkgl_pls, cli_pls, dcrs_pls = calc_readability(goldens)
    fkgl_model, cli_model, dcrs_model = calc_readability(preds)

    # Return tuple in the original order
    return sari_score, rouge1, rouge2, rougeL, bleu_score, fkgl_abs, cli_abs, dcrs_abs, fkgl_pls, cli_pls, dcrs_pls, fkgl_model, cli_model, dcrs_model


# ========================================================================
#  MAIN SCRIPT LOGIC
# ========================================================================

if __name__ == "__main__":
    SEED = 42
    # Set global seeds once (as in original)
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED']=str(SEED)
    set_seed(SEED) # Primarily for transformers, but good practice

    # --- Original Argument Parsing ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs_dir', type=str, default='logs/laysummary')
    parser.add_argument("--bootstrap", type=bool, default=True, help="Whether to bootstrap the data") # Original default was True
    parser.add_argument("--n_bootstrap", type=int, default=10, help="Number of bootstrap samples") # Original default was 10
    args = parser.parse_args()
    # ---------------------------------

    # Check base logs directory
    if not os.path.isdir(args.logs_dir):
        print(f"Error: Logs directory not found: {args.logs_dir}")
        exit(1)

    # Loop through the dataset directories inside logs_dir
    for dataset_dir_name in os.listdir(args.logs_dir):
        dataset_path = os.path.join(args.logs_dir, dataset_dir_name)
        if not os.path.isdir(dataset_path):
            continue # Skip files, only process directories

        dataset_name = dataset_dir_name # Use the directory name as dataset identifier
        print(f"\n{'='*10} Dataset: {dataset_name} {'='*10}")

        # Define the path to the reference test data *using the original assumption*
        # Assumes structure like 'my_datasets/laysummary/dataset_1/test.json' relative to script execution?
        # Or maybe it was expected to be somewhere else based on original script context.
        # THIS PATH MAY NEED ADJUSTMENT based on where your original script expected 'test.json'
        data_path = os.path.join("my_datasets/laysummary", dataset_name, "test.json") # Original path structure

        if not os.path.exists(data_path):
            print(f"Warning: Reference data file not found at {data_path}. Skipping bootstrapping for dataset {dataset_name}.")
            # Decide if you want to skip the whole dataset or just bootstrapping
            # continue # Option: skip dataset entirely if ref data missing
            _skip_bootstrap_for_dataset = True # Flag to skip bootstrap section
        else:
             _skip_bootstrap_for_dataset = False


        # Define the order of models to process (as original)
        model_order = ["AgentSimp", "SingleLLM_basic", "SingleLLM_few_shot", "SingleLLM_optimized", "bart-base", "t5-base","bart-large-cnn", "pegasus-large"] # Your model list

        # --- Generate bootstrap IDs if needed ---
        ids = []
        # Only bootstrap if enabled AND reference data exists
        if args.bootstrap and not _skip_bootstrap_for_dataset:
            print(f"Generating {args.n_bootstrap} bootstrap ID samples from {data_path}...")
            ids_generated = True
            for i in range(args.n_bootstrap):
                sample = bootstrap(data_path)
                if not sample: # Check if bootstrap function returned an empty list (error)
                    print(f"Error generating bootstrap sample {i+1} for dataset {dataset_name}. Disabling bootstrap for this dataset.")
                    ids = []
                    ids_generated = False
                    break # Stop trying to generate for this dataset
                ids.append(sample)
            if ids_generated:
                print("Bootstrap ID generation complete.")
        elif args.bootstrap and _skip_bootstrap_for_dataset:
             print("Skipping bootstrap ID generation because reference file was not found.")
        # ---------------------------------------

        # --- Loop through models ---
        for model_name in model_order:
            # Define the path to the model's *directory* in the new structure
            model_results_dir = os.path.join(dataset_path, model_name)

            # Check if the model *directory* exists
            # (Original script checked for model.json in os.listdir - this replaces that check)
            if not os.path.isdir(model_results_dir):
                # Original script checked os.listdir, so finding the directory means the model exists in some form
                # Let's check if the original file existed for compatibility, though we won't use it
                original_file_path = os.path.join(dataset_path, f"{model_name}.json")
                if os.path.exists(original_file_path):
                     print(f"Warning: Original file {model_name}.json exists, but directory {model_name} not found. Skipping.")
                # else: # Directory doesn't exist, and original file didn't either (or wasn't checked)
                     # print(f"Info: Directory for model '{model_name}' not found. Skipping.")
                continue # Skip this model if its directory doesn't exist

            print(f"\n--- Model: {model_name} ---")

            # === MODIFICATION START: Load data from the new structure ===
            model_results = [] # This will store the list of dicts, like the original script expected
            print(f"  Loading results from: {model_results_dir}")
            loaded_count = 0
            error_count = 0
            try:
                for filename in os.listdir(model_results_dir):
                    # Check if the file matches the expected pattern
                    if filename.startswith("laysummary_") and filename.endswith("-result.json"):
                        file_path = os.path.join(model_results_dir, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data_item = json.load(f)
                                # Basic validation: ensure it's a dict and has required keys
                                if isinstance(data_item, dict) and all(k in data_item for k in ["id", "source", "target", "pred"]):
                                    model_results.append(data_item)
                                    loaded_count += 1
                                else:
                                    print(f"    Warning: Skipping file {filename}. Invalid content or missing keys.")
                                    error_count += 1
                        except json.JSONDecodeError:
                            print(f"    Warning: Invalid JSON in {filename}. Skipping.")
                            error_count += 1
                        except Exception as e:
                            print(f"    Warning: Error reading {filename}: {e}. Skipping.")
                            error_count += 1
                print(f"  Successfully loaded {loaded_count} results, skipped {error_count} files.")

            except FileNotFoundError:
                print(f"  Error: Directory {model_results_dir} not found during file listing.")
                continue # Skip to next model
            except Exception as e:
                print(f"  Error listing files in {model_results_dir}: {e}")
                continue # Skip to next model

            if not model_results:
                print(f"  Warning: No valid result files found or loaded for model '{model_name}'. Skipping evaluation.")
                continue
            # === MODIFICATION END ===


            # --- The rest of the logic remains the same as your original script ---
            # --- It now operates on the 'model_results' list loaded above ---

            # Initialize lists to store metrics for each bootstrap run
            sari, rouge1, rouge2, rougeL, bleu = [], [], [], [], []
            fkgl_abs, cli_abs, dcrs_abs = [], [], []
            fkgl_pls, cli_pls, dcrs_pls = [], [], []
            fkgl_model, cli_model, dcrs_model = [], [], []

            # Use the generated bootstrap IDs (ids list)
            if args.bootstrap and ids: # Check if bootstrap enabled AND ids were generated successfully
                print(f"  Calculating metrics using {len(ids)} bootstrap samples...")
                # Build a lookup dictionary from the loaded results for efficiency
                results_dict = {item['id']: item for item in model_results}

                # Loop through the bootstrap samples (ids[i] contains a list of sampled IDs)
                for i in range(len(ids)): # Use len(ids) which matches n_bootstrap unless errors occurred
                    sampled_ids = ids[i]
                    # Original script filtered the list - using the dictionary lookup is more efficient here
                    sampled_model_results = []
                    missing_ids = 0
                    for id_ in sampled_ids:
                        result = results_dict.get(id_)
                        if result:
                            sampled_model_results.append(result)
                        else:
                            missing_ids += 1

                    if missing_ids > 0:
                         print(f"    Bootstrap sample {i+1}: {missing_ids} IDs not found in loaded results.")

                    if not sampled_model_results:
                        print(f"    Warning: No data retrieved for bootstrap sample {i+1}. Skipping.")
                        continue # Skip this sample if no data was found

                    # Get the sources, targets, and preds *for this specific bootstrap sample*
                    sources = [model_res['source'] for model_res in sampled_model_results]
                    targets = [model_res['target'] for model_res in sampled_model_results]
                    preds = [model_res['pred'] for model_res in sampled_model_results]

                    # Calculate metrics using your original function
                    sari_, r1_, r2_, rL_, bleu_, fa_, ca_, da_, fp_, cp_, dp_, fm_, cm_, dm_ = get_metrics(preds, targets, sources, seed=SEED + i) # Vary seed slightly per sample

                    # Append metrics for this run
                    sari.append(sari_)
                    rouge1.append(r1_)
                    rouge2.append(r2_)
                    rougeL.append(rL_)
                    bleu.append(bleu_)
                    fkgl_abs.append(fa_)
                    cli_abs.append(ca_)
                    dcrs_abs.append(da_)
                    fkgl_pls.append(fp_)
                    cli_pls.append(cp_)
                    dcrs_pls.append(dp_)
                    fkgl_model.append(fm_)
                    cli_model.append(cm_)
                    dcrs_model.append(dm_)

                # --- After looping through bootstrap samples ---
                if not sari: # Check if any samples were processed
                     print("  Warning: No bootstrap samples were successfully processed.")
                     continue # Skip printing results for this model

                # Get the mean and std of the metrics (using original rounding)
                sari_mean, sari_std = round(np.mean(sari), 2), round(np.std(sari), 2)
                # Note: ROUGE scores from calc_rouge are now 0-1, so mean/std will be too.
                # Multiply by 100 here if you want 0-100 scale output, matching the original print format expectation
                rouge1_mean, rouge1_std = round(np.mean(rouge1) * 100, 2), round(np.std(rouge1) * 100, 2)
                rouge2_mean, rouge2_std = round(np.mean(rouge2) * 100, 2), round(np.std(rouge2) * 100, 2)
                rougeL_mean, rougeL_std = round(np.mean(rougeL) * 100, 2), round(np.std(rougeL) * 100, 2)
                # Bleu score from evaluate is 0-100, so no scaling needed here
                bleu_mean, bleu_std = round(np.mean(bleu), 2), round(np.std(bleu), 2)
                fkgl_abs_mean, fkgl_abs_std = round(np.mean(fkgl_abs), 2), round(np.std(fkgl_abs), 2)
                cli_abs_mean, cli_abs_std = round(np.mean(cli_abs), 2), round(np.std(cli_abs), 2)
                dcrs_abs_mean, dcrs_abs_std = round(np.mean(dcrs_abs), 2), round(np.std(dcrs_abs), 2)
                fkgl_pls_mean, fkgl_pls_std = round(np.mean(fkgl_pls), 2), round(np.std(fkgl_pls), 2)
                cli_pls_mean, cli_pls_std = round(np.mean(cli_pls), 2), round(np.std(cli_pls), 2)
                dcrs_pls_mean, dcrs_pls_std = round(np.mean(dcrs_pls), 2), round(np.std(dcrs_pls), 2)
                fkgl_model_mean, fkgl_model_std = round(np.mean(fkgl_model), 2), round(np.std(fkgl_model), 2)
                cli_model_mean, cli_model_std = round(np.mean(cli_model), 2), round(np.std(cli_model), 2)
                dcrs_model_mean, dcrs_model_std = round(np.mean(dcrs_model), 2), round(np.std(dcrs_model), 2)

                # Print the results using the original dictionary structure
                metrics = {
                    'sari': f"{sari_mean} ± {sari_std}",
                    'rouge1': f"{rouge1_mean} ± {rouge1_std}", # Now correctly scaled to 0-100
                    'rouge2': f"{rouge2_mean} ± {rouge2_std}", # Now correctly scaled to 0-100
                    'rougeL': f"{rougeL_mean} ± {rougeL_std}", # Now correctly scaled to 0-100
                    'bleu': f"{bleu_mean} ± {bleu_std}",
                    'abs_readability': {
                        'fkgl': f"{fkgl_abs_mean} ± {fkgl_abs_std}",
                        'cli': f"{cli_abs_mean} ± {cli_abs_std}",
                        'dcrs': f"{dcrs_abs_mean} ± {dcrs_abs_std}"
                    },
                    'pls_readability': {
                        'fkgl': f"{fkgl_pls_mean} ± {fkgl_pls_std}",
                        'cli': f"{cli_pls_mean} ± {cli_pls_std}",
                        'dcrs': f"{dcrs_pls_mean} ± {dcrs_pls_std}"
                    },
                    'model_readability': {
                        'fkgl': f"{fkgl_model_mean} ± {fkgl_model_std}",
                        'cli': f"{cli_model_mean} ± {cli_model_std}",
                        'dcrs': f"{dcrs_model_mean} ± {dcrs_model_std}"
                    }
                }
                print(f"Metrics for {model_name} (Bootstrap Mean ± Std Dev):")
                # Pretty print the dictionary (original script didn't use json.dumps, just print(metrics))
                print(metrics) # Reverted to original print style

            else: # --- No Bootstrapping ---
                # Check if bootstrap was intended but skipped due to missing ref file
                if args.bootstrap and _skip_bootstrap_for_dataset:
                    print("  Skipping full dataset evaluation because bootstrapping was requested but reference file was missing.")
                    continue # Skip full evaluation for this model too

                print("  Calculating metrics using all loaded data (no bootstrapping)...")
                # Use all the data loaded into model_results
                sources = [model_res['source'] for model_res in model_results]
                targets = [model_res['target'] for model_res in model_results]
                preds = [model_res['pred'] for model_res in model_results]

                # Calculate metrics once
                sari_, r1_, r2_, rL_, bleu_, fa_, ca_, da_, fp_, cp_, dp_, fm_, cm_, dm_ = get_metrics(preds, targets, sources, seed=SEED)

                # Print results directly (no std dev), scale ROUGE here for printing if needed
                print(f"Metrics for {model_name} (Full Dataset):")
                print(f"  SARI  : {round(sari_, 2)}")
                print(f"  ROUGE1: {round(r1_ * 100, 2)}") # Scale for printing
                print(f"  ROUGE2: {round(r2_ * 100, 2)}") # Scale for printing
                print(f"  ROUGEL: {round(rL_ * 100, 2)}") # Scale for printing
                print(f"  BLEU  : {round(bleu_, 2)}")
                print("  Readability (Source):")
                print(f"    FKGL: {round(fa_, 2)}, CLI: {round(ca_, 2)}, DCRS: {round(da_, 2)}")
                print("  Readability (Target):")
                print(f"    FKGL: {round(fp_, 2)}, CLI: {round(cp_, 2)}, DCRS: {round(dp_, 2)}")
                print("  Readability (Model):")
                print(f"    FKGL: {round(fm_, 2)}, CLI: {round(cm_, 2)}, DCRS: {round(dm_, 2)}")


    print("\nEvaluation script finished.")