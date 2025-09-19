from .constrained_decoding import generate_constrained_response_batch, classify_vocabulary, generate_probe_constrained_response_batch
from .check_answer import evaluate_answer, extract_after_boxed
import numpy as np
from transformers import DynamicCache
import torch
import os
import h5py
import pandas as pd
from tqdm import tqdm
import torch.nn as nn

def set_random_seed(seed_value):

    os.environ['PYTHONHASHSEED']=str(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataGenerator():
    """
    A class to generate data for the probing task.
    """

    def __init__(self, cfg, model, tokenizer, dataset):
        set_random_seed(cfg.seed)
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.layer_num = cfg.dataset.layer_num
        # add hook
        self.dataset = dataset

        # 3.54s saved
        vocab = [self.tokenizer.decode(i) for i in range(len(self.tokenizer))]
        vocab += [""] * (self.model.config.vocab_size - len(vocab))
        self.vocab = vocab
        
        
        token_class_np, ch_composite_tokens = classify_vocabulary(vocab, self.tokenizer)
        self.token_class = torch.from_numpy(token_class_np).to(self.model.device)
        self.ch_composite_tokens = ch_composite_tokens
        
        # Suppose you have one activation vector (1D numpy array) at a time
        # Example:
        # activations_np = np.random.randn(5120).astype(np.float32)
        
        # Create (or open) the .h5 file and dataset
        file_path = self.cfg.dataset.save_dir + "activations_" + self.cfg.dataset.name + '.h5'
        feature_dim = self.cfg.dataset.feature_dim
        
        # Create the file and resizable dataset if it doesn't exist
        with h5py.File(file_path, "a") as f:
            for layer in self.cfg.dataset.layer_num:
                if "activations_"+str(layer) not in f:
                    dset = f.create_dataset(
                        "activations_"+str(layer),
                        shape=(0, feature_dim),
                        maxshape=(None, feature_dim),  # Unlimited rows
                        dtype='float16',
                        chunks=True  # Enable resizing
                    )
            
        self.activation_file_path = file_path
        self.stats_file_path = self.cfg.dataset.save_dir + "stats_" + self.cfg.dataset.name + '.csv'
        
    # Append one activation at a time
    def append_activation_row(self, dset, activation_row):
        #assert activation_row.shape == (feature_dim,)
        current_size = dset.shape[0]
        dset.resize(current_size + 1, axis=0)
        dset[current_size] = activation_row
            
    # On Yihao's device, use this version
    def shorten_past_data(self, past_data, all_generated_ids, token_count, start_location, duplicate_num=1):
        """
        Shorten the past data to the given token counts. You may need to use the other version due to incompatibility.
        """
        new_data = {"key_values": DynamicCache(), "token_ids": None}


        for l in range(len(past_data["key_values"].key_cache)):
            new_key = past_data["key_values"].key_cache[l][:, :, :-token_count+start_location, :]
            new_value = past_data["key_values"].value_cache[l][:, :, :-token_count+start_location, :]
            if duplicate_num > 1:
                new_key = new_key.repeat(duplicate_num, 1, 1, 1)
                new_value = new_value.repeat(duplicate_num, 1, 1, 1)
            new_data["key_values"].key_cache.append(new_key)
            new_data["key_values"].value_cache.append(new_value)

        new_data["token_ids"] = torch.stack([torch.tensor(ids, dtype=torch.long) for ids in all_generated_ids for _ in range(duplicate_num)]).to(self.model.device) #[B, len]
        
        new_data["start_location"] = start_location
        
        return new_data
    
    
    '''
    # On Jiayi's device, use this version
    def shorten_past_data(self, past_data, all_generated_ids, token_count, start_location, duplicate_num=1):
        """
        Shorten the past key/value cache to the given token counts for Qwen/QwQ models.
        """
        # Store new key/value pairs as a list of tuples
        new_kv_pairs = []

        for key, value in past_data["key_values"]:
            new_key = key[:, :, :-token_count + start_location, :]
            new_value = value[:, :, :-token_count + start_location, :]

            if duplicate_num > 1:
                new_key = new_key.repeat(duplicate_num, 1, 1, 1)
                new_value = new_value.repeat(duplicate_num, 1, 1, 1)

            new_kv_pairs.append((new_key, new_value))

        # Convert to same format as original key_values (tuple of tuples)
        new_data = {
            "key_values": tuple(new_kv_pairs),
            "token_ids": torch.stack([
                torch.tensor(ids, dtype=torch.long)
                for ids in all_generated_ids
                for _ in range(duplicate_num)
            ]).to(self.model.device)
        }
        new_data["start_location"] = start_location
        return new_data
    
    '''
    
    

    def generate_sample(self):
        """
        Generate a sample for the probing task.
        """
        
        inputs = self.generate_batch_prompts(self.dataset["chin_query"].tolist()[0:2])
        generated_texts, token_counts, all_generated_ids, all_masks_np, _, _, past_data, _ = generate_constrained_response_batch(self.model, inputs, self.tokenizer, self.vocab, self.token_class, self.cfg.dataset.max_tokens, "switch_to_EN", [11,11], do_sample = self.cfg.dataset.do_sample, temperature = self.cfg.dataset.temperature, top_p = self.cfg.dataset.top_p, \
                                                                                                                                   past_data=None, save_past_data=True, ch_composite_tokens=self.ch_composite_tokens)
        
        #import pdb; pdb.set_trace()
        
        #print(generated_texts)
        
        past_data = self.shorten_past_data(past_data, all_generated_ids, token_counts[0], 5)
        generated_texts, token_counts, all_generated_ids, all_masks_np, _, _, past_data, activations_np = generate_constrained_response_batch(self.model, inputs, self.tokenizer, self.vocab, self.token_class, self.cfg.dataset.max_tokens, "none", [5,5], do_sample = self.cfg.dataset.do_sample, temperature = self.cfg.dataset.temperature, top_p = self.cfg.dataset.top_p, \
                                                                                                                                   past_data=past_data, save_past_data=False, ch_composite_tokens=self.ch_composite_tokens, layer_num=[47, 63])
        #print(generated_texts)
        
        

    def generate_batch_prompts(self, questions):
        # Create batched chat prompts
        messages_batch = [[{"role": "user", "content": q}] for q in questions]
        texts = [
            self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_batch
        ]
        
        # Tokenize the batch
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.model.device)

        return inputs
    
    def generate_sample_data(self, sample_idx=0, prompt_lang='ch', num_constrained_positions=1):
        if prompt_lang=='ch':
        # Step 0. Generate initial unconstrained response
            inputs = self.generate_batch_prompts([self.dataset["chin_query"].iloc[sample_idx]])
        else:
            inputs = self.generate_batch_prompts([self.dataset["query"].iloc[sample_idx]])
        problem_id = self.dataset["problem_id"].iloc[sample_idx]
        generated_texts, token_counts, all_generated_ids, all_masks_np, all_switches_np, all_heuristics, past_data, _ = generate_constrained_response_batch(self.model, inputs, self.tokenizer, self.vocab, self.token_class, self.cfg.dataset.max_tokens, "none", None, do_sample = self.cfg.dataset.do_sample, temperature = self.cfg.dataset.temperature, top_p = self.cfg.dataset.top_p, \
                                                                                                                                   past_data=None, save_past_data=True, heuristic_measure='lang_entropy', ch_composite_tokens=self.ch_composite_tokens)
        #print(generated_texts)
        #print(all_heuristics)
        
        
        answer0 = extract_after_boxed(generated_texts[0])
        gt_answer = self.dataset["answer"].iloc[sample_idx]
        score0 = evaluate_answer(gt_answer, answer0)
        stats0 = generate_node_stats(idx=None, if_switch=None, if_natural=None, if_en_to_ch=None, switch_position=0, score=score0, answer=answer0, token_count=token_counts[0], gt_answer=gt_answer, heuristic=None)
        node0 = Node(stats=stats0)
        all_nodes = [node0]
        #import pdb; pdb.set_trace()
        for n in range(num_constrained_positions):

            if n >= 1:
                all_generated_ids = all_generated_ids_node
                all_masks_np = all_masks_np_node
                all_switches_np = all_switches_np_node
                all_heuristics = all_heuristics_node
                past_data_node = {"key_values": None, "token_ids": torch.tensor(all_generated_ids_node).to(self.model.device), "start_location": 0}
                
                generated_texts, _, _, _, _, _, past_data, _ = generate_constrained_response_batch(self.model, inputs, self.tokenizer, self.vocab, self.token_class, self.cfg.dataset.max_tokens, "none", [self.cfg.dataset.max_tokens], do_sample = self.cfg.dataset.do_sample, temperature = self.cfg.dataset.temperature, top_p = self.cfg.dataset.top_p, \
                                                                                                                                   past_data=past_data_node, save_past_data=True, heuristic_measure='lang_entropy', ch_composite_tokens=self.ch_composite_tokens)
                #print(generated_texts)
                answer0 = extract_after_boxed(generated_texts_node[0])
                score0 = evaluate_answer(gt_answer, answer0)
                stats0 = generate_node_stats(idx=None, if_switch=None, if_natural=None, if_en_to_ch=None, switch_position=selected_location, score=score0, answer=answer0, token_count=token_counts_[0], gt_answer=gt_answer, heuristic=None)
                node0 = Node(stats=stats0)
            # Step 1: For all natural switches (excluding position 0 and position -1), apply unswitch constraints
            natural_switch_positions = np.nonzero(all_switches_np[0])[0].tolist()
            natural_switch_positions = [pos for pos in natural_switch_positions if pos > node0.stats["switch_position"] and pos<len(all_switches_np[0])-1]

            modes = []
            locations = []

            for pos in natural_switch_positions:
                switch_direction = all_switches_np[0][pos]
                if switch_direction == 1:
                    mode = "no_CH"  # prevent CH if EN->CH
                elif switch_direction == -1:
                    mode = "no_EN"  # prevent EN if CH->EN

                modes.append(mode)
                locations.append(pos)

            # Step 2: Add synthetic switches if there are not enough natural ones
            token_count = token_counts[0]
            num_natural_switches = len(locations)  # from step 1

            if num_natural_switches >= self.cfg.dataset.max_switches:
                # randomly select self.cfg.dataset.max_switches positions
                random_idx = np.random.choice(num_natural_switches, self.cfg.dataset.max_switches, replace=False)
                modes = [modes[i] for i in random_idx]
                locations = [locations[i] for i in random_idx]
                num_natural_switches = self.cfg.dataset.max_switches



            desired_total = int(self.cfg.dataset.data_ratio * token_count)
            num_synthetic = max(0, desired_total - num_natural_switches)
            
            if num_synthetic > 0:
                # Candidate positions: not math, not already switched, not position 0
                valid_positions = np.array([pos for pos in np.where(all_masks_np[0] == 0)[0]
                                            if pos not in locations])
                valid_positions = np.array([pos for pos in valid_positions if pos > node0.stats["switch_position"] and pos < len(all_switches_np[0])-1])
                

                if len(valid_positions) > 0:
                    
                    heuristics_at_valid = all_heuristics[0][valid_positions]  # shape: [#valid_positions]
                    
                    mask = heuristics_at_valid > self.cfg.dataset.heuristic_threshold

                    filtered_pos = valid_positions[mask]
                    sorted_idx = np.argsort(-heuristics_at_valid[mask])# descending order
                    selected_positions = filtered_pos[sorted_idx[:num_synthetic]]


                    # Final positions sorted left-to-right
                    synthetic_positions = sorted(selected_positions.tolist())
                    #print('syn', len(synthetic_positions))
                    for pos in synthetic_positions:
                        
                        prior_switches = [p for p in range(pos) if all_switches_np[0][p]!=0]
                            
                        if len(prior_switches) > 0:
                            last_pos = prior_switches[-1]
                            current_lang = "ch" if all_switches_np[0][last_pos] == 1 else "en"
                        else:
                            post_switches = [p for p in range(pos, len(all_switches_np[0])) if all_switches_np[0][p]!=0]
                            next_pos = post_switches[0]
                            current_lang = "ch" if all_switches_np[0][next_pos] == -1 else "en"

                        # Flip language
                        mode = "switch_to_EN" if current_lang == "ch" else "switch_to_CH"
                        modes.append(mode)
                        locations.append(pos)
            #print(n, locations)

            # Randomly select one as the parent node for the next iteration
            if len(locations) > 0:
                random_idx = np.random.randint(0, len(locations))
                selected_location = locations[random_idx]
                #print("selected_location", selected_location)
                
            # Step 3: Sort (mode, location) pairs and batchify
            batch_size = self.cfg.dataset.batch_size  # e.g., 2, 4, etc.
            paired = list(zip(modes, locations))
            paired.sort(key=lambda x: x[1])  # sort by location

            for i in range(0, len(paired), batch_size):
                batch = paired[i:i+batch_size]
                batch_modes, batch_locations = zip(*batch)
                #print(batch_modes, batch_locations)
                if selected_location in batch_locations:
                    save_past_data = True
                else:
                    save_past_data = False
                
                # Step 4: Generate responses per batch with shortened cache
                min_location = min(batch_locations)
                past_data_truncated = self.shorten_past_data(past_data, all_generated_ids, token_counts[0], min_location-1, duplicate_num=len(batch_modes))
                generated_texts, token_counts_, all_generated_ids_, all_masks_np_, all_switches_np_, all_heuristics_, past_data_, activations_np = generate_constrained_response_batch(self.model, inputs, self.tokenizer, self.vocab, self.token_class, self.cfg.dataset.max_tokens, batch_modes, batch_locations, do_sample = self.cfg.dataset.do_sample, temperature = self.cfg.dataset.temperature, top_p = self.cfg.dataset.top_p, \
                                                                                                                                    past_data=past_data_truncated, save_past_data=False, heuristic_measure="lang_entropy" if save_past_data else None, ch_composite_tokens=self.ch_composite_tokens, layer_num=self.cfg.dataset.layer_num)
                #print(batch_modes, batch_locations)
                #print(generated_texts)
                if selected_location in batch_locations:
                    generated_texts_node = [generated_texts[batch_locations.index(selected_location)]]
                    all_generated_ids_node = [all_generated_ids_[batch_locations.index(selected_location)]]
                    all_masks_np_node = [all_masks_np_[batch_locations.index(selected_location)]]
                    all_switches_np_node = [all_switches_np_[batch_locations.index(selected_location)]]
                    
                    all_heuristics_node = [all_heuristics_[batch_locations.index(selected_location)]]
                    
                #print(generated_texts)
                # Save activations line by line:
                with h5py.File(self.activation_file_path, "a") as f:
                    for layer in self.cfg.dataset.layer_num:
                        dset = f["activations_"+str(layer)]
                        start_idx = dset.shape[0]
                        assert len(activations_np[str(layer)]) == len(batch_modes)
                        for b in range(len(batch_modes)):
                            self.append_activation_row(dset, activations_np[str(layer)][b])
                
                                        
                for b in range(len(batch_modes)):
                    if batch_modes[b] == "no_CH" or batch_modes[b] == "no_EN":
                        if_switch = 0
                        if_natural = 1
                    else:
                        if_switch = 1
                        if_natural = 0
                    if batch_modes[b] == "no_CH" or batch_modes[b] == "switch_to_CH":
                        if_en_to_ch = 1
                    else:
                        if_en_to_ch = 0

                    answer = extract_after_boxed(generated_texts[b])
                    score = evaluate_answer(gt_answer, answer)
                    stats = generate_node_stats(idx=start_idx+b, if_switch=if_switch, if_natural=if_natural, if_en_to_ch=if_en_to_ch, switch_position=batch_locations[b], score=score, answer=answer, token_count=token_counts_[b], gt_answer=gt_answer, heuristic=all_heuristics[0][batch_locations[b]])
                    node = Node(stats=stats, parent=node0)
                    all_nodes.append(node)
                    node0.add_child(node)
            # Step 5: Generate final stats
            new_stats = []
            for child_node in node0.children:
                idx = child_node.stats["idx"]
                if child_node.stats["if_switch"] == 0:
                    without_switch_score = child_node.stats["score"]
                    without_switch_answer = child_node.stats["answer"]
                    without_switch_token_count = child_node.stats["token_count"]
                    with_switch_score = node0.stats["score"]
                    with_switch_answer = node0.stats["answer"]
                    with_switch_token_count = node0.stats["token_count"]
                else:
                    with_switch_score = child_node.stats["score"]
                    with_switch_answer = child_node.stats["answer"]
                    with_switch_token_count = child_node.stats["token_count"]
                    without_switch_score = node0.stats["score"]
                    without_switch_answer = node0.stats["answer"]
                    without_switch_token_count = node0.stats["token_count"]
                if_natural = child_node.stats["if_natural"]
                if_en_to_ch = child_node.stats["if_en_to_ch"]
                heuristic = child_node.stats["heuristic"]
                final_stats = generate_final_stats(idx=idx, if_natural=if_natural, if_en_to_ch=if_en_to_ch, without_switch_score=without_switch_score, without_switch_answer=without_switch_answer, without_switch_token_count=without_switch_token_count, with_switch_score=with_switch_score, with_switch_answer=with_switch_answer, with_switch_token_count=with_switch_token_count, gt_answer=gt_answer, heuristic=heuristic, problem_id=problem_id, prompt_lang=prompt_lang)
                new_stats.append(final_stats)

            df = pd.DataFrame(new_stats)

            # Append to file if exists, otherwise create a new one with header
            df.to_csv(self.stats_file_path, mode="a", header=not os.path.exists(self.stats_file_path), index=False)
            if len(locations) == 0:
                break

    def generate_all_data(self, data_range=[0,10], prompt_lang='ch', num_constrained_positions=1):
        for sample_idx in tqdm(range(data_range[0], data_range[1])):
            #try:
            self.generate_sample_data(sample_idx, prompt_lang, num_constrained_positions)
            #except:
            #    print("Error in sample", sample_idx)
            #    continue

    def generate_constrained_response_with_probe(self, sample_idxs=[0], prompt_lang='ch', probe=None):
        """
        Generate constrained responses with probe.
        """
        if prompt_lang=='ch':
        # Step 0. Generate initial unconstrained response
            inputs = self.generate_batch_prompts(self.dataset["chin_query"].iloc[sample_idxs].tolist())
        else:
            inputs = self.generate_batch_prompts(self.dataset["query"].iloc[sample_idxs].tolist())

        
        generated_texts, token_counts, all_generated_ids, all_masks_np, all_switches_np, all_heuristics_np, probe_logs = generate_probe_constrained_response_batch(self.model, inputs, probe, self.tokenizer, self.vocab, self.token_class, self.cfg.dataset.max_tokens, do_sample = self.cfg.dataset.do_sample, temperature = self.cfg.dataset.temperature, top_p = self.cfg.dataset.top_p, \
                                                                                                                                   heuristic_measure='lang_entropy', heuristic_threshold=self.cfg.dataset.heuristic_threshold, ch_composite_tokens=self.ch_composite_tokens, layer_num=self.cfg.dataset.layer_num)

        
        #print(generated_texts)
        
        return generated_texts, token_counts, probe_logs
        
        
        

def generate_final_stats(idx, if_natural, if_en_to_ch, without_switch_score, without_switch_answer, without_switch_token_count, with_switch_score, with_switch_answer, with_switch_token_count, gt_answer, heuristic, problem_id, prompt_lang):
    """
    Generate a statistics dictionary for the probing task.
    """
    stats_dict = {
        "idx": idx,
        "if_natural": if_natural,
        "if_en_to_ch": if_en_to_ch,
        "without_switch_score": without_switch_score,
        "without_switch_answer": without_switch_answer,
        "without_switch_token_count": without_switch_token_count,
        "with_switch_score": with_switch_score,
        "with_switch_answer": with_switch_answer,
        "with_switch_token_count": with_switch_token_count,
        "gt_answer": gt_answer,
        "heuristic": heuristic,
        "problem_id": problem_id,
        "prompt_lang": prompt_lang
    }
    return stats_dict

def generate_node_stats(idx, if_switch, if_natural, if_en_to_ch, switch_position, score, answer, token_count, gt_answer, heuristic):
    """
    Generate a statistics dictionary for the probing task.
    """
    stats_dict = {
        "idx": idx,
        "if_switch": if_switch,
        "if_natural": if_natural,
        "if_en_to_ch": if_en_to_ch,
        "switch_position": switch_position,
        "score": score,
        "answer": answer,
        "token_count": token_count,
        "gt_answer": gt_answer,
        "heuristic": heuristic
    }
    return stats_dict


class Node:
    def __init__(self, stats, parent=None):
        self.stats = stats # if_natural, if_en_to_ch, score, answer, gt_answer, token_count,
        self.parent = parent
        self.children = []

    def add_child(self, child_node):
        child_node.parent = self  # Set the parent of the child
        self.children.append(child_node)

    def remove_child(self, child_node):
        self.children = [child for child in self.children if child != child_node]
        child_node.parent = None  # Clear the parent reference

    def traverse(self, depth=0):
        print("  " * depth + str(self.value))
        for child in self.children:
            child.traverse(depth + 1)

    def backtrace(self):
        path = []
        current = self
        while current:
            path.append(current.value)
            current = current.parent
        return list(reversed(path))  # from root to current node
