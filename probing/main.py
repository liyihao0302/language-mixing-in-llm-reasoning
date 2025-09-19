# main.py
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.data_generator import DataGenerator
from utils.logit_lens import LogitLens
import torch
import pandas as pd
import torch.nn as nn

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig):
    # Print out the config to verify that Hydra loaded it
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True, cache_dir = cfg.model.cache_dir, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=torch.float16 if cfg.model.dtype=='float16' else torch.float32,  # Use float16 for lower memory use
        device_map="auto",          # Automatically selects the GPU
        trust_remote_code=True,
        cache_dir=cfg.model.cache_dir)
    
    
    if cfg.mode == 'data_generation':
        df_dataset = pd.read_csv(cfg.dataset_dir+'translated_'+cfg.dataset.name+'.csv')
        
        data_generator = DataGenerator(cfg, model, tokenizer, df_dataset)
        data_generator.generate_all_data(data_range=[0,5], prompt_lang='ch', num_constrained_positions=4)
        print("Data generation completed.")
    elif cfg.mode == 'logit_lens':
        cfg.dataset.layer_num = list(range(64))  # layers 0 to 63
        df_dataset = pd.read_csv(cfg.dataset_dir+'translated_'+cfg.dataset.name+'.csv')
        logit_lens = LogitLens(cfg, model, tokenizer, df_dataset)
        logit_lens.apply_logit_lens(sample_idx=0, location=158, prompt_lang='ch')
        print("Logit lens completed.")

    elif cfg.mode == 'constrained_decoding_with_probe':
        df_dataset = pd.read_csv(cfg.dataset_dir+'translated_'+cfg.dataset.name+'.csv')
        data_generator = DataGenerator(cfg, model, tokenizer, df_dataset)
        """
        The probe should take in both activations and meta_features:
        activations={'layer_name': [B,5120], 'layer_name': [B,5120], ...}, and return [B]: 0-harmful, 1-neutral, 2-beneficial
        meta_features={'if_natural':[B], 'if_en_to_ch':[B], 'heuristic':[B], 'if_zh_en':[B]}
        """
        class simple_probe(nn.Module):
            def __init__(self):
                super(simple_probe, self).__init__()
                pass
            def forward(self, activations, meta_features=None):
                for key in activations.keys():
                    break
                B = activations[key].shape[0]
                return torch.full((B,), 2, device=activations[key].device)
            
        
        probe = simple_probe()

        #-----Set data range and batch_size here-----
        data_range = [2, 150]
        batch_size = 2
        #-----------------------------

        for i in range(data_range[0], data_range[1], batch_size):
            sample_idxs = list(range(i, min(i+batch_size, data_range[1])))
            generated_texts, token_counts, probe_logs= data_generator.generate_constrained_response_with_probe(sample_idxs=sample_idxs, prompt_lang='ch', probe=probe)
            '''
            probe_logs is a list of strings: control mode (no_EN, no_CH, switch_to_EN, switch_to_CH), the constrained token (a string of the token before applying the probe)
            '''
            # write code to save to csv: save generated_texts, token_counts, probe_logs to a csv file
            print(generated_texts)
            print(probe_logs)
            import pdb; pdb.set_trace()
            
        
if __name__ == "__main__":
    main()