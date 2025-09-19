
from .constrained_decoding import generate_constrained_response_batch, classify_vocabulary, find_nontext_letters
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
class LogitLens():
    """
    A class to generate data for the probing task.
    """

    def __init__(self, cfg, model, tokenizer, dataset):
        
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
        
        exception_indices = set(find_nontext_letters(vocab))

        # Build index lists for each category
        ch_indices = (self.token_class[:, 0] == 1).nonzero(as_tuple=True)[0]
        en_indices = set((self.token_class[:, 1] == 1).nonzero(as_tuple=True)[0]) - set(exception_indices)
        

        # Convert to tensors (for indexing into probs[i])
        self.ch_indices = torch.tensor(list(ch_indices))
        self.en_indices = torch.tensor(list(en_indices))

    def apply_logit_lens(self, sample_idx, location=1, prompt_lang='ch'):
        """
        Generate a sample for the probing task.
        """
        device = self.model.device
        if prompt_lang == 'ch':
            inputs = self.generate_batch_prompts([self.dataset["chin_query"].iloc[sample_idx]])
        elif prompt_lang == 'en':
            inputs = self.generate_batch_prompts([self.dataset["query"].iloc[sample_idx]])
        generated_texts, token_counts, all_generated_ids, all_masks_np, all_switches_np, _, _, activations_np = generate_constrained_response_batch(self.model, inputs, self.tokenizer, self.vocab, self.token_class, self.cfg.dataset.max_tokens, "none", [location], do_sample = self.cfg.dataset.do_sample, temperature = self.cfg.dataset.temperature, top_p = self.cfg.dataset.top_p, \
                                                                                                                                   past_data=None, save_past_data=None, ch_composite_tokens=self.ch_composite_tokens, layer_num = self.cfg.dataset.layer_num)
        print(generated_texts)

        p_ens = []
        p_chs = []
        entropy_list = []
        lang_entropy_list = []
        for key in tqdm(activations_np.keys()):
            logits = self.model.model.norm(torch.from_numpy(activations_np[key]).to(device))
            logits = self.model.lm_head(logits) #[1, vocab_size]
            probs = F.softmax(logits, dim=-1) # temperature=1
            probs = probs[0]
            
            # entropy
            entropy = -torch.sum(probs* torch.log2(probs+1e-6)).item()


            # lang_entropy
            p_ch = probs[self.ch_indices].sum().item()
            p_en = probs[self.en_indices].sum().item()

            # Compute binary entropy
            lang_entropy = (-(p_ch * np.log2(p_ch/(p_ch+p_en+1e-6)+1e-6) + p_en * np.log2(p_en/(p_ch+p_en+1e-6)+1e-6))).item()

            
            p_chs.append(p_ch)
            p_ens.append(p_en)
            entropy_list.append(entropy)
            lang_entropy_list.append(lang_entropy)

        
        #import pdb; pdb.set_trace()
        p_nons = 1 - np.array(p_ens) - np.array(p_chs)
        layers = np.arange(64)

        fig, axs = plt.subplots(3, 1, figsize=(10, 6), height_ratios=[1, 1, 2], constrained_layout=True)

        # Plot 1: entropy_list heatmap
        axs[0].imshow(np.expand_dims(entropy_list, axis=0), aspect='auto', cmap='Blues')
        axs[0].set_title("Entropy")
        axs[0].set_yticks([])
        axs[0].set_xticks([])

        # Plot 2: lang_entropy_list heatmap
        axs[1].imshow(np.expand_dims(lang_entropy_list, axis=0), aspect='auto', cmap='Greens')
        axs[1].set_title("Lang Entropy")
        axs[1].set_yticks([])
        axs[1].set_xticks([])

        # Plot 3: Line plot of probabilities
        axs[2].plot(layers, p_ens, label='p_en', color='purple')
        axs[2].plot(layers, p_chs, label='p_ch', color='green')
        axs[2].plot(layers, p_nons, label='p_non', color='orange')
        axs[2].set_title("Language Probability Across Layers")
        axs[2].set_xlabel("Layer")
        axs[2].set_ylabel("Probability")
        axs[2].legend()

        plt.savefig(f"logit_lens.png")

            
            
        
        

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

        