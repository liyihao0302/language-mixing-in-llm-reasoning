import torch
import re
import numpy as np
from tqdm import tqdm
import itertools
import string
import torch.nn.functional as F

import matplotlib.pyplot as plt
import matplotlib as mpl

activations = {}

'''

def get_activations(name, sample_idx):
    def hook(model, input, output):
        if activations.get(name) is None:
            activations[name] = []
        import pdb; pdb.set_trace()
        activations[name].append(output.detach())
    return hook

for num_layer, child in self.model.model.layers.named_children():
    if int(num_layer) is in cfg.model.num_layer:
        child.register_forward_hook(get_activations(num_layer))
'''


def generate_constrained_response(model, inputs, tokenizer, vocab, token_class, max_token = 4096, mode='no_EN', location=None, do_sample=False, temperature=1.0, top_p=1.0, ch_composite_tokens=None):
    '''
    mode: 'no_EN', 'no_CH', 'switch_to_EN', 'switch_to_CH'

    For 'no_EN' or 'no_CH', the constraint is applied throughout the entire decoding process if location is None, otherwise, it is applied to all tokens preceding the specified position.
    For 'switch_to_EN' or 'switch_to_CH', a 'no_EN' or 'no_CH' constraint is applied before the specified position, and a 'switch_to_EN' or 'switch_to_CH' constraint is applied at that position.

    location: None or int

    RETURN:
    - generated_text: the generated text
    - token_count: the number of tokens generated
    - generated_ids: the generated token ids
    - mask: a list of 0s and 1s indicating whether the token is in math mode (1) or not (0)
    '''

        
    exception_indices = find_nontext_letters(vocab)


    def get_token_mode(mode, location, t):
        if mode == 'none':
            token_mode = 'none'
        elif mode == 'no_EN' or mode == 'no_CH':
            if location is not None:
                if t < location:
                    token_mode = mode
                elif t >= location:
                    token_mode = 'none'
            else:
                token_mode = mode
        elif mode == 'switch_to_EN':
            if t < location:
                token_mode = 'none'
            elif t == location:
                token_mode = 'switch_to_EN'
            else:
                token_mode = 'none'
        elif mode == 'switch_to_CH':
            if t < location:
                token_mode = 'none'
            elif t == location:
                token_mode = 'switch_to_CH'
            else:
                token_mode = 'none'
        return token_mode

    current_lang = 'none'
    mask = []
    switches = []
    with torch.no_grad():

        token_mode = get_token_mode(mode, location, 0)
        # Input the prompt
        output = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            use_cache=True, #kv cache
        )
        current_segment = ("", "none")
        if mode == 'no_EN' and location == None:
            current_segment = ("", "ch")
            current_lang = "ch"
        elif mode == 'no_CH' and location == None:
            current_segment = ("", "en")
            current_lang = "en"
        logits = output.logits
        past_key_values = output.past_key_values
        
        next_token, current_segment, current_lang, math_flag, switch_flag = constrained_decoding(logits[0, -1, :], token_class, vocab, current_segment, current_lang, exception_indices, mode=token_mode, do_sample = do_sample, temperature=temperature, top_p=top_p, tokenizer=tokenizer, ch_composite_tokens = ch_composite_tokens)
        switches.append(switch_flag)
        generated = [next_token.unsqueeze(0)]
        mask.append(math_flag)
        eos_token_id = tokenizer.eos_token_id
        #print(current_segment)
        #print(next_token)
        # Generation loop
        for t in range(1, max_token):
            
            token_mode = get_token_mode(mode, location, t)
            with torch.no_grad():
                output = model(
                    input_ids=next_token.reshape(1,-1),  # only feed in the new token
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                logits = output.logits
                past_key_values = output.past_key_values  # update cache
                next_token, current_segment, current_lang, math_flag, switch_flag = constrained_decoding(logits[0, -1, :], token_class, vocab, current_segment, current_lang, exception_indices, mode=token_mode, do_sample = do_sample, temperature=temperature, top_p=top_p, tokenizer=tokenizer, ch_composite_tokens = ch_composite_tokens)
                mask.append(math_flag)
                switches.append(switch_flag)
                #print(current_segment)
                #print(next_token)
                #import pdb; pdb.set_trace()
                generated.append(next_token.unsqueeze(0))

            

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break


    generated_ids = torch.cat(generated, dim=1)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    return generated_text, t+1, generated_ids, np.array(mask), np.array(switches)
def find_nontext_letters(vocab):
    corner_cases = {"sin", "cos", "tan", "cot", "arcsin", "arccos", "arctan", "mod", "log", "ln", "boxed"}

    # 1–2 letter combinations, any case
    letters = string.ascii_letters  # a-z, A-Z
    allowed_1_2 = {''.join(chars) for n in [1, 2] for chars in itertools.product(letters, repeat=n)}
    
    common_words = {
        "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
        "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we", "ve"
    }
    filtered = {
        word for word in allowed_1_2
        if word.lower() not in common_words and word != "I"
    }
    
    uppercase_letters = string.ascii_uppercase
    allowed_3_4_upper = {''.join(chars) for n in [3, 4] for chars in itertools.product(uppercase_letters, repeat=n)}

    allowed_cases = corner_cases | filtered | allowed_3_4_upper

    indices = []
    for i, token in enumerate(vocab):
        letter_segments = re.findall(r'[a-zA-Z]+', token)
        if all(seg in allowed_cases for seg in letter_segments):
            indices.append(i)
    return indices





def generate_probe_constrained_response_batch(
    model, 
    inputs,
    probe, 
    tokenizer,
    vocab,
    token_class, 
    max_token=4096, 
    do_sample=False, 
    temperature=1.0, 
    top_p=1.0,
    heuristic_measure=None,
    heuristic_threshold=0.2,
    ch_composite_tokens=None,
    layer_num=None,
):
    """
    The probe should take in activations={'layer_name': [B,5120], 'layer_name': [B,5120], ...}, and return [B]: 0-harmful, 1-neutral, 2-beneficial
    """

    device = model.device
    batch_size = inputs["input_ids"].shape[0]

    
    exception_indices = set(find_nontext_letters(vocab))

    # Build index lists for each category
    ch_indices = (token_class[:, 0] == 1).nonzero(as_tuple=True)[0]
    en_indices = set((token_class[:, 1] == 1).nonzero(as_tuple=True)[0]) - set(exception_indices)
    

    # Convert to tensors (for indexing into probs[i])
    ch_indices = torch.tensor(list(ch_indices), device=device)
    en_indices = torch.tensor(list(en_indices), device=device)
    global activations
    
    def get_activations(name):
        def hook(model, input, output):
            activations[name]=output[0][:,-1,:].clone() #[B,5120]
        return hook
    # -------------------------------------------------------------------------
    # 2) Utility: Determine constraint mode per time step
    # -------------------------------------------------------------------------
    # now use a default 'none' mode

    # -------------------------------------------------------------------------
    # 3) Initialize batch states
    # -------------------------------------------------------------------------
    # We'll store generation states for each example in the batch
    
    # Each example will track (current_text_segment, current_lang)
    # e.g. ("", "ch") or ("", "en") etc. after the first step
    # Start them all as ("", "none")
    current_segments = [("", "none") for _ in range(batch_size)]
    current_langs = ["none"] * batch_size

    # We'll store arrays of masks and token-ids for each example
    # (for the newly generated tokens, beyond the prompt).
    all_generated_ids = [[] for _ in range(batch_size)]
    all_masks = [[] for _ in range(batch_size)]
    all_heuristics = [[] for _ in range(batch_size)]
    all_switches = [[] for _ in range(batch_size)]
    probe_logs = [[] for _ in range(batch_size)]

    # We also track which sequences are "finished" (EOS).
    eos_token_id = tokenizer.eos_token_id
    finished = [False] * batch_size
    finished = torch.tensor(finished, dtype=torch.bool, device=inputs["input_ids"].device)
    finished_count = 0

    # -------------------------------------------------------------------------
    # 4) Forward pass for the entire batch on the initial prompt
    # -------------------------------------------------------------------------
    
    
    with torch.no_grad():
        # shape of logits: [batch_size, seq_len, vocab_size]
        
        hook_handles = []  # Keep track of all hook handles 
        
        
        if layer_num != None:
            for b in range(batch_size):
                for num_layer, child in model.model.layers.named_children():
                    if int(num_layer) in layer_num:
                        handle = child.register_forward_hook(get_activations(num_layer))
                        hook_handles.append(handle)
        current_locations = torch.tensor([0] * batch_size, device=device)
        

        output = model(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            use_cache=True
        )

        for handle in hook_handles:
            handle.remove()
        hook_handles.clear()

        logits = output.logits  # [batch_size, seq_len, vocab_size]
        past_key_values = output.past_key_values
        

    # -------------------------------------------------------------------------
    # 5) For each example in the batch, decode the first next token
    # -------------------------------------------------------------------------
    
    current_modes = ["none"] * batch_size

    # 1) Gather relevant rows of logits for the unfinished examples:
    logits_not_finished = logits[~finished, -1, :]  # shape: [num_not_finished, vocab_size]

    # 2) Gather states for the unfinished examples
    segments_not_finished = [current_segments[b] for b in torch.where(~finished)[0]]
    langs_not_finished = [current_langs[b] for b in torch.where(~finished)[0]]
    modes_not_finished = [current_modes[b] for b in torch.where(~finished)[0]]
    
    token_indices, updated_segments, updated_langs, updated_math_flags, updated_switch_flags, heuristics = constrained_decoding_batch(logits_not_finished, token_class, vocab, segments_not_finished, langs_not_finished, exception_indices, modes_not_finished, do_sample, temperature, top_p, tokenizer, heuristic_measure=heuristic_measure, ch_composite_tokens=ch_composite_tokens, ch_indices=ch_indices, en_indices=en_indices)
    
    meta_features = {'if_natural': torch.zeros(batch_size).to(device), 'if_en_to_ch': torch.zeros(batch_size).to(device), 'heuristic': torch.zeros(batch_size).to(device)}
    
    # Create meta feature tensors on the correct device
    meta_features = {
        'if_natural': torch.zeros(batch_size, device=device),
        'if_en_to_ch': torch.zeros(batch_size, device=device),
        'heuristic': torch.zeros(batch_size, device=device),
    }

    # Indices of unfinished examples
    unfinished_indices = torch.where(~finished)[0]

    # Update meta features for unfinished examples
    for i, b in enumerate(unfinished_indices):
        # Check for natural switch (based on updated_switch_flags[i] != 0)
        meta_features['if_natural'][b] = 1 if updated_switch_flags[i] != 0 else 0
        # Check for en -> ch switch
        meta_features['if_en_to_ch'][b] = 1 if updated_switch_flags[i] == 1 else 0
        # Record heuristic score
        meta_features['heuristic'][b] = heuristics[i]

    probe_class = probe(activations, meta_features) #[B] 0-harmful, 1-neutral, 2-beneficial

    

    # Check switch_flags and heuristics
    for i, b in enumerate(unfinished_indices):
        if updated_switch_flags[i] == 1 and probe_class[b] == 0: # natural en->ch, harmful
            current_modes[b] = 'no_CH'
            constrained_token = tokenizer.decode(token_indices[i])
            probe_logs[b].append('no_CH'+ ': ' + constrained_token)
        elif updated_switch_flags[i] == -1 and probe_class[b] == 0: # natural ch -> en, harmful
            current_modes[b] = 'no_EN'
            constrained_token = tokenizer.decode(token_indices[i])
            probe_logs[b].append('no_EN'+ ': ' + constrained_token)
        elif updated_switch_flags[i] == 0:
            if heuristics[i] >= heuristic_threshold and updated_math_flags[i] == 0 and probe_class[b]==2: # synthetic en -> ch, beneficial
                
                if current_langs[b] == 'ch': # synthetic ch -> en, beneficial
                    current_modes[b] = 'switch_to_EN'
                    constrain_token = tokenizer.decode(token_indices[i])
                    probe_logs[b].append('switch_to_EN'+ ': ' + constrain_token)
                elif current_langs[b] == 'en':
                    current_modes[b] = 'switch_to_CH'
                    constrain_token = tokenizer.decode(token_indices[i])
                    probe_logs[b].append('switch_to_CH'+ ': ' + constrain_token)
    modes_not_finished = [current_modes[b] for b in torch.where(~finished)[0]]
    
    activations = {} # reset activations
    token_indices, updated_segments, updated_langs, updated_math_flags, updated_switch_flags, heuristics = constrained_decoding_batch(logits_not_finished, token_class, vocab, segments_not_finished, langs_not_finished, exception_indices, modes_not_finished, do_sample, temperature, top_p, tokenizer, heuristic_measure=heuristic_measure, ch_composite_tokens=ch_composite_tokens, ch_indices=ch_indices, en_indices=en_indices)

    all_chosen = torch.full((batch_size,), fill_value=eos_token_id, dtype=torch.long, device=logits.device)
    # Overwrite those entries where not_finished_mask == True
    all_chosen[~finished] = token_indices  # shape: [batch_size]

    # 5) Update per-example states
    #    We still need a small loop because these are Python lists of states
    unfinished_indices = torch.where(~finished)[0]  # e.g. tensor([0, 2, 5]) 
    for i, b in enumerate(unfinished_indices):
        # Update the segment/lang
        current_segments[b] = updated_segments[i]
        current_langs[b] = updated_langs[i]
        # Append the chosen token ID to that example's generation history
        all_generated_ids[b].append(int(all_chosen[b].item()))
        all_heuristics[b].append(heuristics[i])
        all_masks[b].append(updated_math_flags[i])  # e.g. 0 or 1
        all_switches[b].append(updated_switch_flags[i])  # e.g. 0 or 1

    # 6) Build the final next_tokens of shape [batch_size, 1]
    next_tokens = all_chosen.unsqueeze(-1)  # [batch_size, 1]

    # 7) Check for newly reached EOS in the just-chosen tokens

    # Update current_locations and finished mask
    current_locations[~finished] += 1
    finished |= (all_chosen == eos_token_id) | (current_locations >= max_token)
    finished_count = finished.sum().item()
            

        
        
        # -------------------------------------------------------------------------
        # 6) Iterative generation loop
        # -------------------------------------------------------------------------
        # We'll generate up to max_token new tokens in total.
        # If all sequences finish early, we break.
    while 1:

        if finished_count == batch_size:
            # all are done
            break

        hook_handles = []  # Keep track of all hook handles 
        
        
        if layer_num != None:
            for b in range(batch_size):
                for num_layer, child in model.model.layers.named_children():
                    if int(num_layer) in layer_num:
                        handle = child.register_forward_hook(get_activations(num_layer))
                        hook_handles.append(handle)
            
            
        with torch.no_grad():
            # Single forward pass for the entire batch with last token
            output = model(
                input_ids=next_tokens, 
                past_key_values=past_key_values,
                use_cache=True
            )
            
            for handle in hook_handles:
                handle.remove()
            hook_handles.clear()

            logits = output.logits  # [batch_size, 1, vocab_size]
            past_key_values = output.past_key_values

        current_modes = ["none"] * batch_size

        # 1) Gather relevant rows of logits for the unfinished examples:
        logits_not_finished = logits[~finished, -1, :]  # shape: [num_not_finished, vocab_size]

        # 2) Gather states for the unfinished examples
        segments_not_finished = [current_segments[b] for b in torch.where(~finished)[0]]
        langs_not_finished = [current_langs[b] for b in torch.where(~finished)[0]]
        modes_not_finished = [current_modes[b] for b in torch.where(~finished)[0]]
        token_indices, updated_segments, updated_langs, updated_math_flags, updated_switch_flags, heuristics = constrained_decoding_batch(logits_not_finished, token_class, vocab, segments_not_finished, langs_not_finished, exception_indices, modes_not_finished, do_sample, temperature, top_p, tokenizer, heuristic_measure=heuristic_measure, ch_composite_tokens=ch_composite_tokens, ch_indices=ch_indices, en_indices=en_indices)

        meta_features = {'if_natural': torch.zeros(batch_size).to(device), 'if_en_to_ch': torch.zeros(batch_size).to(device), 'heuristic': torch.zeros(batch_size).to(device)}
    
        # Create meta feature tensors on the correct device
        meta_features = {
            'if_natural': torch.zeros(batch_size, device=device),
            'if_en_to_ch': torch.zeros(batch_size, device=device),
            'heuristic': torch.zeros(batch_size, device=device),
        }

        # Indices of unfinished examples
        unfinished_indices = torch.where(~finished)[0]

        # Update meta features for unfinished examples
        for i, b in enumerate(unfinished_indices):
            # Check for natural switch (based on updated_switch_flags[i] != 0)
            meta_features['if_natural'][b] = 1 if updated_switch_flags[i] != 0 else 0
            # Check for en -> ch switch
            meta_features['if_en_to_ch'][b] = 1 if updated_switch_flags[i] == 1 else 0
            # Record heuristic score
            meta_features['heuristic'][b] = heuristics[i]
            
        probe_class = probe(activations, meta_features) #[B] 0-harmful, 1-neutral, 2-beneficial
        
        # Check switch_flags and heuristics
        for i, b in enumerate(unfinished_indices):
            if updated_switch_flags[i] == 1 and probe_class[b] == 0: # natural en->ch, harmful
                current_modes[b] = 'no_CH'
                constrained_token = tokenizer.decode(token_indices[i])
                probe_logs[b].append('no_CH'+ ': ' + constrained_token)
            elif updated_switch_flags[i] == -1 and probe_class[b] == 0: # natural ch -> en, harmful
                current_modes[b] = 'no_EN'
                constrained_token = tokenizer.decode(token_indices[i])
                probe_logs[b].append('no_EN'+ ': ' + constrained_token)
            elif updated_switch_flags[i] == 0:
                if heuristics[i] >= heuristic_threshold and updated_math_flags[i] == 0 and probe_class[b]==2: # synthetic en -> ch, beneficial
                    
                                    
                    if current_langs[b] == 'ch': # synthetic ch -> en, beneficial
                        current_modes[b] = 'switch_to_EN'
                        constrain_token = tokenizer.decode(token_indices[i])
                        probe_logs[b].append('switch_to_EN'+ ': ' + constrain_token)
                    elif current_langs[b] == 'en':
                        current_modes[b] = 'switch_to_CH'
                        constrain_token = tokenizer.decode(token_indices[i])
                        probe_logs[b].append('switch_to_CH'+ ': ' + constrain_token)
        modes_not_finished = [current_modes[b] for b in torch.where(~finished)[0]]
        
        activations = {} # reset activations
        token_indices, updated_segments, updated_langs, updated_math_flags, updated_switch_flags, heuristics = constrained_decoding_batch(logits_not_finished, token_class, vocab, segments_not_finished, langs_not_finished, exception_indices, modes_not_finished, do_sample, temperature, top_p, tokenizer, heuristic_measure=heuristic_measure, ch_composite_tokens=ch_composite_tokens, ch_indices=ch_indices, en_indices=en_indices)

        all_chosen = torch.full((batch_size,), fill_value=eos_token_id, dtype=torch.long, device=logits.device)
        # Overwrite those entries where not_finished_mask == True
        all_chosen[~finished] = token_indices  # shape: [batch_size]

        # 5) Update per-example states
        #    We still need a small loop because these are Python lists of states
        unfinished_indices = torch.where(~finished)[0]  # e.g. tensor([0, 2, 5]) 
        for i, b in enumerate(unfinished_indices):
            # Update the segment/lang
            current_segments[b] = updated_segments[i]
            current_langs[b] = updated_langs[i]
            # Append the chosen token ID to that example's generation history
            all_generated_ids[b].append(int(all_chosen[b].item()))
            all_heuristics[b].append(heuristics[i])
            all_masks[b].append(updated_math_flags[i])  # e.g. 0 or 1
            all_switches[b].append(updated_switch_flags[i])

        # 6) Build the final next_tokens of shape [batch_size, 1]
        next_tokens = all_chosen.unsqueeze(-1)  # [batch_size, 1]

        # 7) Check for newly reached EOS in the just-chosen tokens
        current_locations[~finished] += 1
        finished |= (all_chosen == eos_token_id) | (current_locations >= max_token)
        finished_count = finished.sum().item()

        #print(current_segments)
        #import pdb; pdb.set_trace()
    
    
    # -------------------------------------------------------------------------
    # 7) Convert generated IDs back to text
    # -------------------------------------------------------------------------
    generated_texts = []
    token_counts = []
    for b in range(batch_size):
        # The new tokens are in all_generated_ids[i].
        decoded = tokenizer.decode(all_generated_ids[b], skip_special_tokens=False)

        generated_texts.append(decoded)
        token_counts.append(len(all_generated_ids[b]))

    # Convert all_masks[i] to arrays
    all_masks_np = [np.array(m) for m in all_masks]
    all_switches_np = [np.array(s) for s in all_switches]
    all_heuristics_np = [np.array(h) for h in all_heuristics]

    
    for b in range(batch_size):
        
        if np.all(all_switches_np[b] == 0):
            # Set the first switch_flag based on current_lang
            
            if current_langs[b] == 'ch':
                all_switches_np[b][0] = 1 #EN->CH
            elif current_langs[b] == 'en':
                all_switches_np[b][0] = -1 #CH->EN
 
    
    return generated_texts, token_counts, all_generated_ids, all_masks_np, all_switches_np, all_heuristics_np, probe_logs


def generate_constrained_response_batch(
    model, 
    inputs, 
    tokenizer,
    vocab,
    token_class, 
    max_token=4096, 
    modes='no_EN', 
    locations=None, 
    do_sample=False, 
    temperature=1.0, 
    top_p=1.0,
    past_data = None,
    save_past_data=False,
    heuristic_measure=None,
    ch_composite_tokens=None,
    layer_num=None,
):
    """
    Batched version of constrained decoding. 
    Ensures that all samples in the batch go through the GPU at the same time.

    Parameters
    ----------
    model : PreTrainedModel
        The (HF) model used for generation.
    inputs : dict
        A dictionary of batched input_ids and attention_mask, both of shape [batch_size, seq_len].
    tokenizer : PreTrainedTokenizer
        The tokenizer corresponding to the model.
    max_token : int
        Maximum number of new tokens to generate.
    modes : list
        One of 'no_EN', 'no_CH', 'switch_to_EN', 'switch_to_CH'.
    locations : list
        Position where switching constraints apply (for switch modes). If None, constraint is constant.
    do_sample : bool
        Whether to use sampling (True) or greedy decoding (False).
    temperature : float
        Temperature for sampling.
    top_p : float
        Top-p for nucleus sampling.
    past_key_values: list
        Optional. Precomputed past key values to resume generation from a prior context.
    start_location: int
        Optional. Initial location (per example) to resume generation.
    next_tokens: list of ints
        Optional. Initial next tokens (per example) to resume generation.
    save_past_key_values: list of bools
        Optional. Whether to save past key values for each example.
    save_past_key_values_location: int
        Optional. Location to save past key values for each example.

    Returns
    -------
    generated_texts : list of str
        The generated text for each example in the batch.
    token_counts : list of int
        The number of tokens generated for each example.
    all_generated_ids : list of torch.LongTensor
        The generated token IDs for each example.
    all_masks : list of ndarray
        For each example, a mask array (0/1) of the same length as the generated tokens,
        indicating whether the token is in math mode (1) or not (0).
    """

    device = model.device
    if past_data is None:
        batch_size = inputs["input_ids"].shape[0]
    else:
        batch_size = past_data['token_ids'].shape[0]
        
    if isinstance(modes, str):
        modes = [modes] * batch_size
    if locations is None:
        locations = [None] * batch_size

    
    exception_indices = set(find_nontext_letters(vocab))

    # Build index lists for each category
    ch_indices = (token_class[:, 0] == 1).nonzero(as_tuple=True)[0]
    en_indices = set((token_class[:, 1] == 1).nonzero(as_tuple=True)[0]) - set(exception_indices)
    

    # Convert to tensors (for indexing into probs[i])
    ch_indices = torch.tensor(list(ch_indices), device=device)
    en_indices = torch.tensor(list(en_indices), device=device)
    
    
    def get_activations(name, sample_idx):
        def hook(model, input, output):
            if activations.get(name) is None:
                activations[name] = []
            activations[name].append(output[0][sample_idx].cpu()) #[1,5120]
        return hook
    # -------------------------------------------------------------------------
    # 2) Utility: Determine constraint mode per time step
    # -------------------------------------------------------------------------
    def get_token_mode(single_mode, single_location, t):
        """
        Returns which constraint mode applies at time t.
        """
        if single_mode == 'none':
            token_mode = 'none'
        elif single_mode in ['no_EN', 'no_CH']:
            
            # entire gen or until location
            if single_location is not None:
                if t == single_location:
                    token_mode = single_mode
                else:
                    token_mode = 'none'
            else:
                token_mode = single_mode
        elif single_mode == 'switch_to_EN':
            if t < single_location:
                token_mode = 'none'
            elif t == single_location:
                token_mode = 'switch_to_EN'
            else:
                token_mode = 'none'
        elif single_mode == 'switch_to_CH':
            if t < single_location:
                token_mode = 'none'
            elif t == single_location:
                token_mode = 'switch_to_CH'
            else:
                token_mode = 'none'
        else:
            token_mode = 'none'
        return token_mode

    # -------------------------------------------------------------------------
    # 3) Initialize batch states
    # -------------------------------------------------------------------------
    # We'll store generation states for each example in the batch
    
    # Each example will track (current_text_segment, current_lang)
    # e.g. ("", "ch") or ("", "en") etc. after the first step
    # Start them all as ("", "none")
    current_segments = [("", "none") for _ in range(batch_size)]
    current_langs = ["none"] * batch_size
    

    # If user set mode = 'no_EN' (w/o location), 
    # then we might want to start current_lang = 'ch' etc.
    for b in range(batch_size):
        if modes[b] == 'no_EN' and locations[b] is None:
            
            current_langs[b] = "ch"
            current_segments[b] = ("", "ch")
        elif modes[b] == 'no_CH' and locations[b] is None:
            
            current_langs[b] = "en"
            current_segments[b] = ("", "en")

    # We'll store arrays of masks and token-ids for each example
    # (for the newly generated tokens, beyond the prompt).
    all_generated_ids = [[] for _ in range(batch_size)]
    all_masks = [[] for _ in range(batch_size)]
    all_heuristics = [[] for _ in range(batch_size)]
    all_switches = [[] for _ in range(batch_size)]

    # We also track which sequences are "finished" (EOS).
    eos_token_id = tokenizer.eos_token_id
    finished = [False] * batch_size
    finished = torch.tensor(finished, dtype=torch.bool, device=inputs["input_ids"].device)
    finished_count = 0

    # -------------------------------------------------------------------------
    # 4) Forward pass for the entire batch on the initial prompt
    # -------------------------------------------------------------------------
    
    
    with torch.no_grad():
        # shape of logits: [batch_size, seq_len, vocab_size]
        if past_data is None:
            current_locations = torch.tensor([0] * batch_size, device=device)
            

            output = model(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
                use_cache=True
            )


            logits = output.logits  # [batch_size, seq_len, vocab_size]
            past_key_values = output.past_key_values
            

        # -------------------------------------------------------------------------
        # 5) For each example in the batch, decode the first next token
        # -------------------------------------------------------------------------
        
            current_modes = [get_token_mode(modes[b], locations[b], 0) for b in range(batch_size)]

            # 1) Gather relevant rows of logits for the unfinished examples:
            logits_not_finished = logits[~finished, -1, :]  # shape: [num_not_finished, vocab_size]

            # 2) Gather states for the unfinished examples
            segments_not_finished = [current_segments[b] for b in torch.where(~finished)[0]]
            langs_not_finished = [current_langs[b] for b in torch.where(~finished)[0]]
            modes_not_finished = [current_modes[b] for b in torch.where(~finished)[0]]
            
            token_indices, updated_segments, updated_langs, updated_math_flags, updated_switch_flags, heuristics = constrained_decoding_batch(logits_not_finished, token_class, vocab, segments_not_finished, langs_not_finished, exception_indices, modes_not_finished, do_sample, temperature, top_p, tokenizer, heuristic_measure=heuristic_measure, ch_composite_tokens=ch_composite_tokens, ch_indices=ch_indices, en_indices=en_indices)
            
            all_chosen = torch.full((batch_size,), fill_value=eos_token_id, dtype=torch.long, device=logits.device)
            # Overwrite those entries where not_finished_mask == True
            all_chosen[~finished] = token_indices  # shape: [batch_size]

            # 5) Update per-example states
            #    We still need a small loop because these are Python lists of states
            unfinished_indices = torch.where(~finished)[0]  # e.g. tensor([0, 2, 5]) 
            for i, b in enumerate(unfinished_indices):
                # Update the segment/lang
                current_segments[b] = updated_segments[i]
                current_langs[b] = updated_langs[i]
                # Append the chosen token ID to that example's generation history
                all_generated_ids[b].append(int(all_chosen[b].item()))
                all_heuristics[b].append(heuristics[i])
                all_masks[b].append(updated_math_flags[i])  # e.g. 0 or 1
                all_switches[b].append(updated_switch_flags[i])  # e.g. 0 or 1

            # 6) Build the final next_tokens of shape [batch_size, 1]
            next_tokens = all_chosen.unsqueeze(-1)  # [batch_size, 1]

            # 7) Check for newly reached EOS in the just-chosen tokens

            # Update current_locations and finished mask
            current_locations[~finished] += 1
            finished |= (all_chosen == eos_token_id) | (current_locations >= max_token)
            finished_count = finished.sum().item()
            

        else:
            past_key_values = past_data['key_values'] 
            past_token_ids = past_data['token_ids'] #[B, T]
            start_location = past_token_ids.shape[-1]
            
            current_locations = torch.tensor([start_location] * batch_size, device=device)
            for t in range(start_location):
                for b in range(batch_size):
                    chosen_str = vocab[past_token_ids[b,t].item()]

                    current_segments[b], current_langs[b], math_flag_i, switch_flag_i = update_current_segment(chosen_str, current_segments[b], current_langs[b], True if past_token_ids[b,t].item() in ch_composite_tokens else False)
     
                      
                    all_generated_ids[b].append(int(past_token_ids[b,t].item()))
                    all_heuristics[b].append(None)
                    all_masks[b].append(math_flag_i)  # e.g. 0 or 1
                    all_switches[b].append(switch_flag_i)  # e.g. 0 or 1
            next_tokens = past_token_ids[:,-1].unsqueeze(-1)  # [batch_size, 1]
        
        # -------------------------------------------------------------------------
        # 6) Iterative generation loop
        # -------------------------------------------------------------------------
        # We'll generate up to max_token new tokens in total.
        # If all sequences finish early, we break.
        while 1:

            if finished_count == batch_size:
                # all are done
                break

            hook_handles = []  # Keep track of all hook handles 
            
            
            if layer_num != None:
                for b in range(batch_size):
                    if current_locations[b] == locations[b]:
                        for num_layer, child in model.model.layers.named_children():
                            if int(num_layer) in layer_num:
                                handle = child.register_forward_hook(get_activations(num_layer, b))
                                hook_handles.append(handle)
            
            
            with torch.no_grad():
                # Single forward pass for the entire batch with last token
                output = model(
                    input_ids=next_tokens, 
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                for handle in hook_handles:
                    handle.remove()
                hook_handles.clear()

                logits = output.logits  # [batch_size, 1, vocab_size]
                past_key_values = output.past_key_values

            current_modes = [get_token_mode(modes[b], locations[b], current_locations[b]) for b in range(batch_size)]

            # 1) Gather relevant rows of logits for the unfinished examples:
            logits_not_finished = logits[~finished, -1, :]  # shape: [num_not_finished, vocab_size]

            # 2) Gather states for the unfinished examples
            segments_not_finished = [current_segments[b] for b in torch.where(~finished)[0]]
            langs_not_finished = [current_langs[b] for b in torch.where(~finished)[0]]
            modes_not_finished = [current_modes[b] for b in torch.where(~finished)[0]]
            token_indices, updated_segments, updated_langs, updated_math_flags, updated_switch_flags, heuristics = constrained_decoding_batch(logits_not_finished, token_class, vocab, segments_not_finished, langs_not_finished, exception_indices, modes_not_finished, do_sample, temperature, top_p, tokenizer, heuristic_measure=heuristic_measure, ch_composite_tokens=ch_composite_tokens, ch_indices=ch_indices, en_indices=en_indices)

            all_chosen = torch.full((batch_size,), fill_value=eos_token_id, dtype=torch.long, device=logits.device)
            # Overwrite those entries where not_finished_mask == True
            all_chosen[~finished] = token_indices  # shape: [batch_size]

            # 5) Update per-example states
            #    We still need a small loop because these are Python lists of states
            unfinished_indices = torch.where(~finished)[0]  # e.g. tensor([0, 2, 5]) 
            for i, b in enumerate(unfinished_indices):
                # Update the segment/lang
                current_segments[b] = updated_segments[i]
                current_langs[b] = updated_langs[i]
                # Append the chosen token ID to that example's generation history
                all_generated_ids[b].append(int(all_chosen[b].item()))
                all_heuristics[b].append(heuristics[i])
                all_masks[b].append(updated_math_flags[i])  # e.g. 0 or 1
                all_switches[b].append(updated_switch_flags[i])

            # 6) Build the final next_tokens of shape [batch_size, 1]
            next_tokens = all_chosen.unsqueeze(-1)  # [batch_size, 1]

            # 7) Check for newly reached EOS in the just-chosen tokens
            current_locations[~finished] += 1
            finished |= (all_chosen == eos_token_id) | (current_locations >= max_token)
            finished_count = finished.sum().item()

            #print(current_segments)
            #import pdb; pdb.set_trace()
    
    
    # -------------------------------------------------------------------------
    # 7) Convert generated IDs back to text
    # -------------------------------------------------------------------------
    generated_texts = []
    token_counts = []
    for b in range(batch_size):
        # The new tokens are in all_generated_ids[i].
        decoded = tokenizer.decode(all_generated_ids[b], skip_special_tokens=False)

        generated_texts.append(decoded)
        token_counts.append(len(all_generated_ids[b]))

    # Convert all_masks[i] to arrays
    all_masks_np = [np.array(m) for m in all_masks]
    all_switches_np = [np.array(s) for s in all_switches]
    all_heuristics_np = [np.array(h) for h in all_heuristics]

    
    for b in range(batch_size):
        
        if np.all(all_switches_np[b] == 0):
            # Set the first switch_flag based on current_lang
            
            if current_langs[b] == 'ch':
                all_switches_np[b][0] = 1 #EN->CH
            elif current_langs[b] == 'en':
                all_switches_np[b][0] = -1 #CH->EN
    
    
    if save_past_data:
        past_data = {
            'key_values': past_key_values
        }
    else:
        past_data = None

    global activations
    activations_np = {k: torch.cat(v, dim=0).numpy() for k, v in activations.items()}
    activations = {} # reset activations
 
    
    return generated_texts, token_counts, all_generated_ids, all_masks_np, all_switches_np, all_heuristics_np, past_data, activations_np


def constrained_decoding_batch(
    logits, 
    token_class, 
    vocab, 
    current_segments, 
    current_langs, 
    exception_indices, 
    modes='no_EN', 
    do_sample=False, 
    temperature=1.0, 
    top_p=1.0, 
    tokenizer=None,
    heuristic_measure=None,
    ch_composite_tokens=None,
    ch_indices=None,
    en_indices=None,
):
    """
    Batched version of constrained_decoding.

    Parameters
    ----------
    logits : torch.FloatTensor
        Shape [batch_size, vocab_size]. Logits for each example in the batch.
    token_class : torch.Tensor
        Shape [vocab_size, 4] (as in your single-example code), 
        storing classification (ch vs. en vs. math vs. etc).
    vocab : list of str
        The decoded string representation of each token ID (0..vocab_size-1).
    current_segments : list of tuple
        For each sample, the current segment: (some_string, 'en'/'ch'/'none'/'...').
    current_langs : list of str
        For each sample, the current language 'en'/'ch'/'none'/etc.
    exception_indices : set or list
        Indices of tokens allowed despite the usual constraints (like short letter combos).
    modes : str or list of str
        The constraint modes to apply ('no_EN', 'no_CH', 'switch_to_EN', 'switch_to_CH', 'none').
        If a single string is passed, it is broadcast to all examples in the batch.
    do_sample : bool
        Whether to sample from the distribution (True) or pick argmax (False).
    temperature : float
        Temperature scaling for sampling.
    top_p : float
        Top-p (nucleus) sampling cutoff.
    tokenizer : PreTrainedTokenizer or None
        (Optional) Tokenizer if needed for debug printing or logging.

    Returns
    -------
    token_indices : torch.LongTensor
        Shape [batch_size]. The selected next-token IDs for each sample.
    new_segments : list of tuple
        Updated segments for each example after selecting the next token.
    new_langs : list of str
        Updated languages for each example.
    math_flags : list of int
        0/1 flags for each example, indicating whether the selected token is in math mode.
    """
    device = logits.device
    batch_size, vocab_size = logits.shape
    
    # If modes is a single string, replicate it across the batch.
    if isinstance(modes, str):
        modes = [modes] * batch_size
    
    # We'll store output for each example in these lists:
    token_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
    new_segments = [None] * batch_size
    new_langs = [None] * batch_size
    math_flags = [None] * batch_size
    switch_flags = [None] * batch_size
    heuristics = [None] * batch_size

    
        

    # 1) Temperature scaling
    logits = logits / temperature
    #import pdb; pdb.set_trace()
    # 2) Softmax over row_logits
    probs = F.softmax(logits, dim=-1)
    
    # 3) Top-p nucleus filtering
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    if top_p < 1.0:
        # cumulative probabilities along each row
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # mask out everything after crossing top_p
        # "cumulative_mask" has shape [batch_size, vocab_size]
        cumulative_mask = (cumulative_probs <= top_p)

        # zero out any probability beyond the threshold
        sorted_probs = sorted_probs * cumulative_mask

        # renormalize each row
        row_sums = sorted_probs.sum(dim=-1, keepdim=True)
        # If a row has sum=0, it means that entire row is masked out => fallback
        # We'll handle fallback individually in the loop below.
        # For rows that sum > 0, do normal re-normalization:
        nonzero_mask = (row_sums > 0).squeeze(1)
        sorted_probs[nonzero_mask] = (
            sorted_probs[nonzero_mask] / row_sums[nonzero_mask]
        )
        
        
    for i in range(batch_size):
        row_mode = modes[i]
        row_segment = current_segments[i]
        row_lang = current_langs[i]
        # 4) Apply language constraints if needed
        if row_mode != 'none':
            #import pdb; pdb.set_trace()
            allowed_mask = torch.ones_like(sorted_probs[i], dtype=torch.bool, device=device)

            if row_mode == 'no_CH':

                if row_segment[1] == 'en' or row_segment[1] == 'none' or row_segment[1] == 'ch':
                # "no_CH" means no Chinese: we enforce token_class[:,0] == 0
                # token_class[x, 0] = 1 => "Chinese char"
                # so allowed is token_class[x, 0] == 0
                    allowed_mask = (token_class[sorted_indices[i], 0] == 0)

            elif row_mode == 'no_EN':
                
                if row_segment[1] == 'en' or row_segment[1] == 'none' or row_segment[1] == 'ch':
                # "no_EN" means no letter tokens => token_class[x,1] == 0
                    allowed_mask = (token_class[sorted_indices[i], 1] == 0)
    
                    # Re-allow exception tokens
                    exception_tensor = torch.tensor(list(exception_indices), device=device)
                    row_exception_mask = torch.isin(sorted_indices[i], exception_tensor)
                    allowed_mask[row_exception_mask] = 1

            elif row_mode == 'switch_to_EN':
                # Must be an English letter token => token_class[x,1] == 1
                allowed_mask = (token_class[sorted_indices[i], 1] == 1)
                

            elif row_mode == 'switch_to_CH':
                # Must be a Chinese token => token_class[x,0] == 1
                allowed_mask = (token_class[sorted_indices[i], 0] == 1)
            
            # Filter out disallowed tokens
            filtered_probs = sorted_probs[i] * allowed_mask
            if filtered_probs.sum() == 0:
                #import pdb; pdb.set_trace()
                # Fallback: revert to the *full* row_probs if everything is disallowed
                filtered_probs = probs[i]
                filtered_probs, sorted_indices[i] = torch.sort(filtered_probs, descending=True)
            else:
                filtered_probs = filtered_probs / filtered_probs.sum()

            # Overwrite sorted_probs with filtered version
            sorted_probs[i] = filtered_probs

        # 5) Sample or Greedy pick
        if do_sample:
            sampled_idx = torch.multinomial(sorted_probs[i], num_samples=1)
        else:
            sampled_idx = torch.argmax(sorted_probs[i])

        # The chosen token ID is sorted_indices[sampled_idx]
        chosen_token_id = sorted_indices[i, sampled_idx]
        if heuristic_measure=='surprisal': #-logP(w_t|w_1:t-1)
            heuristics[i] = -probs[i, chosen_token_id].item()
        elif heuristic_measure=='entropy':
            heuristics[i] = -torch.sum(probs[i] * torch.log2(probs[i])).item()

        elif heuristic_measure=='lang_entropy':

            # Get probabilities for each group
            p_ch = probs[i][ch_indices].sum().item()
            p_en = probs[i][en_indices].sum().item()

            # Compute binary entropy
            entropy = - (p_ch * np.log2(p_ch/(p_ch+p_en+1e-6)+1e-6) + p_en * np.log2(p_en/(p_ch+p_en+1e-6)+1e-6))
            heuristics[i] = entropy.item()
            #import pdb; pdb.set_trace()
        else:
            heuristics[i] = None
            

        token_indices[i] = chosen_token_id

        # 6) Update current_segment/current_lang
        #    Suppose 'update_current_segment' is the same helper from your original code:
        #         new_segment, new_lang, math_flag = update_current_segment(
        #             next_token_str, row_segment, row_lang
        #         )
        #    We'll decode the chosen token to get next_token_str
        chosen_str = vocab[chosen_token_id.item()]

        updated_seg, updated_lang, math_flag_i, switch_flag_i = update_current_segment(
                chosen_str, row_segment, row_lang, True if chosen_token_id.item() in ch_composite_tokens else False)

        
        new_segments[i] = updated_seg
        new_langs[i] = updated_lang
        math_flags[i] = math_flag_i
        

        switch_flags[i] = switch_flag_i

    

    return token_indices, new_segments, new_langs, math_flags, switch_flags, heuristics




def constrained_decoding(logits, token_class, vocab, current_segment, current_lang, exception_indices, mode='no_EN', do_sample=False, temperature=1.0, top_p=1.0, tokenizer=None, ch_composite_tokens = None):
    '''
    mode: 'no_EN', 'no_CH', 'switch_to_EN', 'switch_to_CH', 'none'

    location: None or int

    do_sample: whether to sample or take the top token
    temperature: temperature for sampling
    top_p: cumulative probability of top tokens to keep
    '''
    # Standard decoding
    logits = logits / temperature
    
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    if top_p != 1.0:
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p

        if torch.any(cutoff):
            # Keep first token where cum_prob exceeds top_p
            first_to_remove = torch.where(cutoff)[0][0]
            sorted_probs[first_to_remove:] = 0.0
        
        sorted_probs = sorted_probs / sorted_probs.sum()
    
    
    # Add language constraints
    
    if mode == 'none':
        pass
    else:
        allowed_mask = torch.ones(sorted_indices.size(), dtype=torch.bool).to(logits.device)
        if mode == 'no_CH':
            if current_segment[1] == 'en' or current_segment[1] == 'none' or current_segment[1] == 'ch':
                # As long as the next token is not Chinese, it's fine
                allowed_mask = token_class[sorted_indices, 0] == 0
            # Any token in math mode is allowed
            
        elif mode == 'no_EN':
            if current_segment[1] == 'en' or current_segment[1] == 'none' or current_segment[1] == 'ch':
                # a) As long as the next token is not letter, it's fine
                allowed_mask = token_class[sorted_indices, 1] == 0
                
                # b) If this token was disallowed purely because it's a letter token, check exceptions
                
                #exception_ranks = [rank for rank, token_id in enumerate(sorted_indices) if token_id.item() in exception_indices]
                #import pdb; pdb.set_trace()
                exception_tensor = torch.tensor(list(exception_indices), device=sorted_indices.device)
                exception_mask = torch.isin(sorted_indices, exception_tensor)

                # Set True at those positions
                allowed_mask[exception_mask] = 1
                
                # Any token in math mode is allowed
                
        elif mode == 'switch_to_EN':
            # The next token must be 'letter'
            allowed_mask = token_class[sorted_indices, 1] == 1
        elif mode == 'switch_to_CH':
            # The next token must be 'ch'
            allowed_mask = token_class[sorted_indices, 0] == 1
        
        sorted_probs = sorted_probs * allowed_mask
        if sorted_probs.sum() == 0:
            # Fall back to none if nothing left
            
            sorted_probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(sorted_probs, descending=True)

        else:
            sorted_probs = sorted_probs / sorted_probs.sum()

    if do_sample == False:
        sampled_index = torch.tensor([torch.argmax(sorted_probs)])

    else:
        sampled_index = torch.multinomial(sorted_probs, num_samples=1)
    
    token_index = sorted_indices[sampled_index]
    '''
    argmax_id = torch.argmax(probs).item()
    if token_index != argmax_id:
        ratio = probs[argmax_id]/probs[token_index]
    else:
        ratio = 1.0
    
    if token_index != torch.argmax(probs):
        top_k = 5  # change as needed
        
        argmax_id = torch.argmax(probs).item()
        argmax_prob = probs[argmax_id].item()
        argmax_token = tokenizer.decode([argmax_id])
    
        print(f"Argmax Token: {argmax_token}, Prob: {argmax_prob:.4f}")
        sorted_probs_, sorted_indices_ = torch.sort(sorted_probs, descending=True)

        # have to redo sorting on sorted_probs
        ...
        # Print out the top-k probs and corresponding token indices
        for i in range(min(top_k, sorted_probs_.size(0))):
            id1 = sorted_indices_[i]
            token_id = sorted_indices[id1].item()

            prob = probs[token_id].item()
            token_str = tokenizer.decode([token_id])
    
            print(f"Top {i} Token: {token_str}, Prob: {prob:.4f}")
    '''
    
    current_segment, current_lang, math_flag, switch_flag = update_current_segment(vocab[token_index[0]], current_segment, current_lang, True if token_index[0] in ch_composite_tokens else False)
        
    
    #print(vocab[token_index[0]])
    return token_index, current_segment, current_lang, math_flag, switch_flag

def classify_vocabulary(vocabulary, tokenizer):
    """
    Classifies each token into one or more of the following categories:
      - 'ch': Chinese chars (and basic CJK punctuation).
      - 'letter': ASCII letters [a-zA-Z].
      - 'special': special tokens with <>
      - 'other': digits, punctuation, spaces, newlines, etc. (tokens not matching any of the above)

    Returns a NumPy array of shape [vocab_size, 4], where each column
    corresponds to one category in the order [ch, letter, math, other].
    A value of 1 indicates the token belongs to that category.
    """
    
            
    # Regex patterns
    chinese_pattern = re.compile(r'[\u4E00-\u9FFF\u3000-\u303F]')  # Basic CJK range + punctuation
    letter_pattern = re.compile(r'[A-Za-z]')                      # ASCII letters
    # Expanded set of LaTeX math-related symbols (full and partial)
    special_pattern =  re.compile(r'<.*?>')

    # Prepare output array
    labels = np.zeros((len(vocabulary), 4), dtype=int)  # [vocab_size, 4]

    for i, token in enumerate(vocabulary):
        has_ch = bool(chinese_pattern.search(token))
        has_letter = bool(letter_pattern.search(token))
        has_special = bool(special_pattern.search(token))

        if has_ch:
            labels[i, 0] = 1
        if has_letter and not has_special:
            labels[i, 1] = 1
        if has_special:
            labels[i, 2] = 1
            labels[i, 3] = 1  # 'math' tokens can be 'other'

        # 'other' only if the token didn't match any of the above
        if not (has_ch or has_letter or has_special):
            labels[i, 3] = 1
    chinese_char = re.compile(r'[\u4E00-\u9FFF\u3000-\u303F]')
    hanzi_chars = [" "+chr(c) for c in range(0x4E00, 0x9FFF+1)]
    tok1s = []
    for ch in hanzi_chars:
        token_ids = tokenizer.encode(ch, add_special_tokens=False)
        if len(token_ids) == 2:
            tok1, tok2 = token_ids
            decoded1 = tokenizer.decode([tok1])
            decoded2 = tokenizer.decode([tok2])
            if not chinese_char.search(decoded1) and not chinese_char.search(decoded2):
                
                if chinese_char.search(tokenizer.decode([tok1,tok2])):
                    #print(f"{repr(decoded1)} + {repr(decoded2)} → {repr(ch)}")
                    labels[tok1, 0] = 1
                    labels[tok1, 1:] = 0
                    tok1s.append(tok1)
                    #labels[tok2, 0] = 1
                    #labels[tok2, 1:] = 0
    ch_composite_tokens = sorted(set(tok1s))
    return labels, ch_composite_tokens
    


def check_nontext_letters(token):
    # extract all consecutive letter segments
    letter_segments = re.findall(r'[a-zA-Z]+', token)

    corner_cases = {"sin", "cos", "tan", "cot", "arcsin", "arccos", "arctan", "mod", "log", "ln", ""}

    common_words = {
        "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
        "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we"
    }

    # If there are no letter segments, we consider it non-text-like → return False
    if not letter_segments:
        return False

    # Check that ALL letter segments satisfy one of the rules
    for seg in letter_segments:
        if (
            seg.lower() in corner_cases or
            (len(seg) <= 2 and seg.lower() not in common_words) or
            (seg.isupper() and len(seg) in [3, 4])
        ):
            continue  # this segment is OK
        else:
            return False  # one segment failed the rule

    return True  # all segments passed


def update_current_segment(token_str, current_segment, current_lang, is_ch_composite):
    """
    Updates the current segment based on the latest token.
    The current segment is a tuple (segment_text, segment_type).
    segment_type can be one of 'en', 'ch', 'math', or 'none'.
    """
    new_segment = current_segment[0] + token_str
    math_flag = 0
    switch_flag = 0 # 1: EN->CH, -1: CH->EN
    if current_segment[1] == 'none':
        
        # no letter and no chinese
        match = re.search(r'(\\\(|\\\[|\$)', new_segment)  # left LaTeX delimiters
        if match:
            new_type = 'math'
            # Keep only from the first bracket (or $) to the end
            new_segment = new_segment[match.start()-1:]
            current_segment = (new_segment, 'math')
            math_flag = 1
        elif re.search(r'[\u4E00-\u9FFF\u3000-\u303F]', token_str) or is_ch_composite:
            new_type = 'ch'
        elif re.search(r'[a-zA-Z]', token_str):
            if check_nontext_letters(token_str):
                new_type = 'none'
            else:
                new_type = 'en'
        else:
            new_type = 'none'
        
    elif current_segment[1] == 'en':
        match = re.search(r'(\\\(|\\\[|\$)', new_segment)  # left LaTeX delimiters
        if match:
            new_type = 'math'
            # Keep only from the first bracket (or $) to the end
            new_segment = new_segment[match.start()-1:]
            current_segment = (new_segment, 'math')
            math_flag = 1
            
        elif re.search(r'[\u4E00-\u9FFF\u3000-\u303F]', token_str) or is_ch_composite:
            new_type = 'ch'
            new_segment = token_str  # Switch to 'ch'
            switch_flag = 1
        else:
            new_type = 'en'
            new_segment = current_segment[0] + token_str

    elif current_segment[1] == 'ch':
        
        match = re.search(r'(\\\(|\\\[|\$)', new_segment)  # left LaTeX delimiters
        if match:
            new_type = 'math'
            new_segment = new_segment[match.start()-1:]
            current_segment = (new_segment, 'math')
            math_flag = 1
        elif re.search(r'[a-zA-Z]', token_str):
            if check_nontext_letters(token_str):
                new_type = 'ch'
            else:
                new_type = 'en'
                new_segment = token_str
                switch_flag = -1
        else:
            new_type = 'ch'
    if current_segment[1] == 'math':
        # Regex to detect a bracket pair: 
        #    - \( ... \), \[ ... \], or \$ ... \$
        #    - Also matches plain ( ... ), [ ... ], or $ ... $, or <...>
        #      Adjust as needed for your LaTeX format
        pattern = re.compile(r'(\$.*?\$|\\\(.*?\\\)|\\\[.*?\\\])', re.DOTALL)

        # 3) Search for the first bracket pair in the combined string
        match = pattern.search(new_segment)

        if match:
            #import pdb; pdb.set_trace()
            math_flag = 1
            # The leftover is whatever comes after the matched pair
            if match.end() == len(new_segment):
                leftover = ""
            else:
                leftover = new_segment[match.end():]

            if leftover:
                # Re-classify the leftover as en, ch, math, or none
                if re.search(r'[\u4E00-\u9FFF\u3000-\u303F]', leftover) or is_ch_composite:
                    new_type = 'ch'
                    new_segment = leftover
                    if current_lang == 'en':
                        switch_flag = 1
                elif re.search(r'[a-zA-Z]', leftover):
                    if check_nontext_letters(leftover):
                        new_type = current_lang
                    else:
                        if current_lang == 'ch':
                            switch_flag = -1
                        new_type = 'en'

                    new_segment = leftover
                elif re.search(r'(\\\(|\\\[|\$)', leftover):
                    new_type = 'math'
                    new_segment = leftover
                else:
                    new_type = current_lang
                    new_segment = leftover
            else:
                # If nothing is leftover, we finished the bracket pair
                new_type = current_lang #***********
                new_segment = ""
        else:
            # If we did not find a bracket pair, we remain in math mode
            new_type = 'math'
            new_segment = new_segment
            math_flag = 1

    if new_type == 'en':
        current_lang = 'en'
    elif new_type == 'ch':
        current_lang = 'ch'
    #if switch_flag !=0:
    #    print(current_segment)
    #    print(new_segment, new_type)
    #    print(switch_flag)
    #    import pdb; pdb.set_trace()
    return (new_segment, new_type), current_lang, math_flag, switch_flag

def check_answer(response: str, answer: str) -> bool:
    """
    Finds boxed... in the response string and compares the content to the given answer.
    Returns True if they match (after stripping), False otherwise.
    """
    match = re.search(r'\\boxed\{(.*?)\}', response, re.DOTALL)
    if match:
        extracted = match.group(1).strip()
        return extracted == answer.strip()
    else:
        # No \box{} found
        return False
    
def enforce_switch_on_text_tokens(mask, model, inputs, tokenizer, answer, max_token = 2048, mode='switch_to_EN', location=None, do_sample=False, temperature=1.0, top_p=1.0):
    """
    Run the force switching on all text in the dataset.
    """
    corrects = []
    token_counts = []

    for t in tqdm(range(len(mask))):
        if mask[t] == 1: # math mode
            
            generated_text = ""
            token_count = 0
            correct = False
            
        else:
            token_mode = mode
            generated_text, token_count, _, _ = generate_constrained_response(model, inputs, tokenizer, max_token = max_token, mode=token_mode, location=t, do_sample=do_sample, temperature=temperature, top_p=top_p)
            #print(generated_text)
            correct = check_answer(generated_text, answer)
            #print(f"✅ Is the answer correct? {'Yes ✅' if correct else 'No ❌'}")
        corrects.append(correct)
        token_counts.append(token_count)
        if token_count == max_token:
            print(generated_text)
    return np.array(corrects), np.array(token_counts)
    



def check_answer_gpt(prompt, gt_anwer, answer):
    pass