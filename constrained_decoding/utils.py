import torch
import re
import numpy as np
from tqdm import tqdm
import itertools
import string
import time
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

import matplotlib.pyplot as plt
import matplotlib as mpl



#font_path = "/root/code/constrained_decoding/SimKai.ttf"
#font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
font_path = "/usr/share/fonts/truetype/arphic/ukai.ttc"
zh_font = FontProperties(fname=font_path)
#zh_font = None
#font_path = "/root/code/constrained_decoding/TimesNewRoman.ttf"
#font_path = "/usr/share/fonts/truetype/msttcorefonts/times.ttf"
#en_font = FontProperties(fname=font_path)
en_font = None
color_map = {'en': '#CAC0E1', 'ch': '#97D1A0', 'math': '#FCDFBE', 'bad_math': '#FCDFBE'}

def generate_constrained_response(model, inputs, tokenizer, max_token = 2048, mode='no_EN', location=None, do_sample=False, temperature=1.0, top_p=1.0):
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
    
    vocab = [tokenizer.decode(i) for i in range(len(tokenizer))]
    vocab += [""] * (model.config.vocab_size - len(vocab))
    
    token_class = torch.zeros((model.config.vocab_size, 4)).to(model.device)
    token_class = torch.from_numpy(classify_vocabulary(vocab, tokenizer)).to(model.device)

    def find_nontext_letters(vocab):
        corner_cases = {"sin", "cos", "tan", "cot", "arcsin", "arccos", "arctan", "mod", "log", "ln"}

        # 1–2 letter combinations (e.g., 'a', 'ab', etc.), any case
        letters = string.ascii_letters  # a-zA-Z
        allowed_1_2 = {''.join(chars) for n in [1, 2] for chars in itertools.product(letters, repeat=n)}
        
        common_words = {
            "am", "an", "as", "at", "be", "by", "do", "go", "he", "if", "in", "is", "it",
            "me", "my", "no", "of", "on", "or", "so", "to", "up", "us", "we", "ve"
        }
        # Step 3: Remove any combination that matches a meaningful word in any case
        filtered = {
            word for word in allowed_1_2
            if word.lower() not in common_words and word != "I"
        }
        
        # 3–4 letter ALL UPPERCASE combinations (e.g., 'URL', 'NASA')
        uppercase_letters = string.ascii_uppercase
        allowed_3_4_upper = {''.join(chars) for n in [3, 4] for chars in itertools.product(uppercase_letters, repeat=n)}

        # Final set
        allowed_cases = corner_cases | filtered | allowed_3_4_upper
        
        indices = []
        for i, token in enumerate(vocab):
            letter_segments = re.findall(r'[a-zA-Z]+', token)
            if all(seg in allowed_cases for seg in letter_segments):
                indices.append(i)

        return indices
        
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
        
        next_token, current_segment, current_lang, math_flag = constrained_decoding(logits[0, -1, :], token_class, vocab, current_segment, current_lang, exception_indices, mode=token_mode, do_sample = do_sample, temperature=temperature, top_p=top_p, tokenizer=tokenizer)
        
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
                next_token, current_segment, current_lang, math_flag = constrained_decoding(logits[0, -1, :], token_class, vocab, current_segment, current_lang, exception_indices, mode=token_mode, do_sample = do_sample, temperature=temperature, top_p=top_p, tokenizer=tokenizer)
                mask.append(math_flag)
                #print(current_segment)
                #print(next_token)
                #import pdb; pdb.set_trace()
                generated.append(next_token.unsqueeze(0))

            

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break


    generated_ids = torch.cat(generated, dim=1)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)

    return generated_text, t+1, generated_ids, np.array(mask)


def constrained_decoding(logits, token_class, vocab, current_segment, current_lang, exception_indices, mode='no_EN', do_sample=False, temperature=1.0, top_p=1.0, tokenizer=None):
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
    
    current_segment, current_lang, math_flag = update_current_segment(vocab[token_index[0]], current_segment, current_lang)
        
    
    #print(vocab[token_index[0]])
    return token_index, current_segment, current_lang, math_flag

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
                    #labels[tok2, 0] = 1
                    #labels[tok2, 1:] = 0
    return labels
    


def check_nontext_letters(token):
    # extract all consecutive letter segments
    letter_segments = re.findall(r'[a-zA-Z]+', token)

    corner_cases = {"sin", "cos", "tan", "cot", "arcsin", "arccos", "arctan", "mod", "log", "ln"}

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


def update_current_segment(token_str, current_segment, current_lang):
    """
    Updates the current segment based on the latest token.
    The current segment is a tuple (segment_text, segment_type).
    segment_type can be one of 'en', 'ch', 'math', or 'none'.
    """
    new_segment = current_segment[0] + token_str
    math_flag = 0
    if current_segment[1] == 'none':
        
        # no letter and no chinese
        match = re.search(r'(\\\(|\\\[|\$)', new_segment)  # left LaTeX delimiters
        if match:
            new_type = 'math'
            # Keep only from the first bracket (or $) to the end
            new_segment = new_segment[match.start()-1:]
            current_segment = (new_segment, 'math')
            math_flag = 1
        elif re.search(r'[\u4E00-\u9FFF\u3000-\u303F]', token_str):
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
            
        elif re.search(r'[\u4E00-\u9FFF\u3000-\u303F]', token_str):
            new_type = 'ch'
            new_segment = token_str  # Switch to 'ch'
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
                if re.search(r'[\u4E00-\u9FFF\u3000-\u303F]', leftover):
                    new_type = 'ch'
                    new_segment = leftover
                elif re.search(r'[a-zA-Z]', leftover):
                    if check_nontext_letters(leftover):
                        new_type = current_lang
                    else:
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
                new_type = current_lang
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

        
    return (new_segment, new_type), current_lang, math_flag

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
    



def draw_token_with_stats(stats, tokens, mask, x_start=0.05, y_start=0.9, fontsize=20,
                                    max_width=0.95, ax=None, line_spacing=0.1, stats_cmap = 'Greens', label=None):
    """
    Renders tokens in either math mode (mask == 1) or normal text mode (mask == 0).
    - Uses text.usetex=True for math tokens.
    - Uses zh_font or en_font for text tokens, depending on whether they look like Chinese or not.
    - Facecolor (the background color) is derived from stats via color_map[stats[idx]].
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(20, 12))
        ax.axis('off')

    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    x = x_start
    y = y_start

    def remove_boxed(tex_str):
        """
        Remove all occurrences of \boxed{...} or \\boxed{...} etc.
        Replaces them with just the enclosed content.
        """
        pattern = re.compile(r'\\+boxed\s*\{(.*?)\}', flags=re.DOTALL)
        return pattern.sub(r'\1', tex_str)
    
    
    idx = 0
    while idx < len(tokens):
        # ---------------------------------------------------------------------
        # 1) If we're in math mode, accumulate consecutive math tokens and plot.
        # ---------------------------------------------------------------------
        
        if mask[idx] == 1:
            start_idx = idx
            while idx < len(tokens) and mask[idx] == 1:
                idx += 1
            math_segment = "".join(tokens[start_idx:idx])
            plt.rcParams['text.usetex'] = True
            text = remove_boxed(math_segment)
            x_old = x
            y_old = y
            try:
                text_type = 'math'
                
                t = ax.text(x, y, text, fontsize=fontsize, va='baseline', bbox=dict(facecolor=color_map[text_type], alpha=1.0, edgecolor='none'))
                fig.canvas.draw()
                bbox = t.get_window_extent(renderer=renderer)
                pixel_width = bbox.width
                pixel_to_data = ax.transData.inverted().transform([[0, 0], [pixel_width, 0]])
                data_width = pixel_to_data[1][0] - pixel_to_data[0][0]

                
                if x + data_width > max_width:
                    t.remove()
                    x = x_start
                    y -= line_spacing
                    t = ax.text(x, y, text, fontsize=fontsize, va='baseline', bbox=dict(facecolor=color_map[text_type], alpha=1.0, edgecolor='none'))
                    fig.canvas.draw()
                    bbox = t.get_window_extent(renderer=renderer)
                    pixel_width = bbox.width
                    pixel_to_data = ax.transData.inverted().transform([[0, 0], [pixel_width, 0]])
                    data_width = pixel_to_data[1][0] - pixel_to_data[0][0]

                x += data_width+0.02
            except:
                # bad math -> en
                t.remove()
                x = x_old
                y = y_old
                text_type = 'bad_math'
                plt.rcParams['text.usetex'] = False
                for char in text:
                    temp_t = ax.text(x, y, char, fontsize=fontsize, fontproperties=en_font, va='baseline', bbox=dict(facecolor=color_map[text_type], alpha=1.0, edgecolor='none'))
                    fig.canvas.draw()
                    bbox = temp_t.get_window_extent(renderer=renderer)
                    pixel_width = bbox.width
                    temp_t.remove()
                    pixel_to_data = ax.transData.inverted().transform([[0, 0], [pixel_width, 0]])
                    data_width = pixel_to_data[1][0] - pixel_to_data[0][0]

                    if x + data_width > max_width:
                        x = x_start
                        y -= line_spacing

                    ax.text(x, y, char, fontsize=fontsize, fontproperties=zh_font, va='baseline', bbox=dict(facecolor=color_map[text_type], alpha=1.0, edgecolor='none'))
                    x += data_width+0.02
        # ---------------------------------------------------------------------
        # 2) If we're not in math mode, plot with either zh_font or en_font.
        # ---------------------------------------------------------------------
        else:
            token = tokens[idx]
            plt.rcParams['text.usetex'] = False  # ensure normal text rendering

            # Decide whether to treat token as Chinese or English
            # (simple heuristic checking for any Chinese character)
            
            if re.search(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', token):
                text_type = 'zh'
                font = zh_font
            else:
                text_type = 'en'
                font = en_font

            # Derive facecolor from stats (assuming color_map is keyed by stats[idx])
            stat_value = stats[idx]
            minv, maxv = np.min(stats) , np.max(stats) # or compute these from the whole stats array
            norm_value = (stat_value - minv) / (maxv - minv)
            norm_value = max(0, min(norm_value, 1))  # clamp to [0,1]

            # Now map this normalized value to a color
            stats_cmap = plt.get_cmap(stats_cmap)
            facecolor = stats_cmap(norm_value)

            # Measure token width first
            temp_t = ax.text(
                x, y, token,
                fontsize=fontsize,
                fontproperties=font,
                va='baseline',
                bbox=dict(facecolor=facecolor, alpha=0.8, edgecolor='none')
            )
            fig.canvas.draw()
            bbox = temp_t.get_window_extent(renderer=renderer)
            pixel_width = bbox.width
            temp_t.remove()

            pixel_to_data = ax.transData.inverted().transform([[0, 0], [pixel_width, 0]])
            data_width = pixel_to_data[1][0] - pixel_to_data[0][0]

            # Check if this token would exceed max width; if so, move to a new line
            if x + data_width > max_width:
                x = x_start
                y -= line_spacing

            # Finally draw the token
            ax.text(
                x, y, token,
                fontsize=fontsize,
                fontproperties=font,
                va='baseline',
                bbox=dict(facecolor=facecolor, alpha=0.8, edgecolor='none')
            )
            x += data_width+0.02

            idx += 1

                    
    norm = mpl.colors.Normalize(vmin=minv, vmax=maxv)
    sm = plt.cm.ScalarMappable(cmap=stats_cmap, norm=norm)
    sm.set_array([])

   
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label(label, fontsize=fontsize)
    if label == 'Accuracy':
        plt.savefig("stats_acc.png", bbox_inches='tight', dpi=300)
    else:
        plt.savefig("stats_token_count.png", bbox_inches='tight', dpi=300)
    
    

                
        
    return ax, y