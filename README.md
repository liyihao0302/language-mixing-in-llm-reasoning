
# language-mixing-in-llm-reasoning

This is the **official repository** for the paper:  
**[The Impact of Language Mixing on Bilingual LLM Reasoning](https://arxiv.org/abs/2507.15849)**


## 🔍 Detecting Code Switching

This module detects code-switching points in model outputs by identifying where the language shifts between English and Chinese. The detection process involves the following steps:

- **Preprocessing**  
   + Extract all math expressions enclosed in brackets (e.g., `\[...\]`, `$$...$$`) and skip them during language segmentation.

- **Language Segmentation**  
   + Label non-math tokens as `'ch'` (Chinese), `'en'` (English), or `'none'` (non-text tokens such as punctuation or symbols).  
   + Corner cases include symbolic tokens like `"x"轴`, `三角形“ABC”`, or trigonometric functions, where math-related terms in English letters are embedded within Chinese text. These are labeled as `'none'` to avoid misclassification.

-  **Backfill Language Labels**  
   + For tokens labeled `'none'`, assign the language of the closest preceding labeled token to maintain continuity.

- **Detect Code Switch Points**  
   + Identify the first text token where the language switches from one to another (e.g., `'en'` → `'ch'` or vice versa), ignoring non-text tokens.

---

1. **Generate Detection Results**  
   Run `detection/detect.py` to produce a CSV file containing code-switching counts and context snippets.
1. **Analyze and Visualize**  
   Use `detection/code_switching_stats.ipynb` to analyze the results and visualize key statistics.


## Constrained Decoding

Currently supporting 5 decoding modes:

- `none` — no constraints
- `no_EN` / `no_CH` — block English or Chinese tokens
- `switch_to_EN` / `switch_to_CH` — enforce a language switch at a specified position; tokens before the switch follow `no_CH` or `no_EN` constraints

The `enforce_switch_on_text_tokens` function applies `switch_to_EN` / `switch_to_CH` constraints to all text tokens, computes accuracy and token count, and returns the results.  
The `draw_token_with_stats` function visualizes how switching at each token position affects model behavior.

### Constrained Decoding at the Token Level

At the token level, the `constrained_decoding` function in `utils.py` supports the same 5 modes. These modes can be **composed** to simulate more complex behaviors, such as alternating between English and Chinese.

- `none`
- `no_EN` / `no_CH`
- `switch_to_EN` / `switch_to_CH`

Use `constrained_decoding/main.py` to run decoding, and generate visualizations.


## 🔍 Probing

This project uses [Hydra](https://hydra.cc/) for flexible experiment configuration. Install it with:

```bash
pip install hydra-core
```

I highly recommend using Hydra to configure and launch probe training runs. For example, to override the default configuration and specify the probing layers:

```bash
python main.py dataset.layer_num=[63, 47, 31, 15, 0]
```

---

### 📂 Output Files

- `activations_.h5`: Stores activations of shape `[N, 5120]` (one row per recorded code-switching position).
- `stats_.csv`: Contains detailed metadata per activation, such as token counts and correctness with/without code-switching.

---

### 🔁 Workflow Overview

For each prompt:

1. **Generate an unconstrained decoding sequence** and save the corresponding key/value cache.
2. **Identify code-switching positions**:
   - If the number of *natural* switches is fewer than `data_ratio * token_count`, synthesize additional switch points.
   - Synthesis is based on **language entropy**; we choose positions where `lang_entropy > heuristic_threshold` (default: 0.2).
   - The **maximum number of recorded switches** is `data_ratio * token_count`.
3. **Rerun decoding** `num_switches` times:
   - For *natural* switches: apply `no_CH` or `no_EN` constraints.
   - For *synthetic* switches: apply `switch_to_CH` or `switch_to_EN`.
4. **Save**:
   - The activation at each switching position (to `.h5`)
   - Statistics for each switch (to `.csv`), including:

---

### 📊 CSV Fields (`stats_.csv`)

| Field                    | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `idx`                   | Row index in `activations_.h5`                                              |
| `if_natural`            | `True` if it's a natural switch; `False` if synthetic                       |
| `if_en_to_ch`           | `1` if EN→CH switch, `0` if CH→EN                                           |
| `without_switch_score`  | `1` = correct, `0` = wrong, `-1` = uncertain, awaiting manual check           |
| `without_switch_answer` | Extracted answer from unconstrained output                                  |
| `without_switch_token_count` | Number of tokens generated without constraint                         |
| `with_switch_score`     | Same format as above, for the constrained output                            |
| `with_switch_answer`    | Extracted answer from the constrained output                                |
| `with_switch_token_count` | Token count for the constrained decoding                                 |
| `gt_answer`             | Ground truth answer                                                         |
| `heuristic`             | Heuristic score (e.g., entropy) used to select the position                 |
| `problem_id`            | ID of the source problem                                                    |
| `prompt_lang`           | Original language of the prompt (`ch` or `en`)                              |

---

### ⚙️ Key Parameters (`config.yaml`)

- **`dataset.layer_num`**  
  Transformer layers to extract activations from. QwQ32B-preview has 64 layers; I recommend `[63, 47, 31, 15, 0]`.  
  Saving multiple layers upfront can help avoid rerunning the pipeline later if additional data is needed.

- **`dataset.heuristic_measure`**  
  Uses **language entropy** (lang_entropy):

  ```
  H = -(p'_en)log2(p'_en) - (p'_ch)log2(p'_ch),  where p'_en + p'_ch = 1
  H_final = (p_en + p_ch) * H
  ```
  This ensures that we only consider positions where both CH and EN are probable *and* uncertainty is high.

- **`dataset.data_ratio`**  
  For MATH500 Chinese prompts, `data_ratio=0.01` gives ~10,000 data points and ~0.38 GB of activations  
  (assuming: `500 prompts × 2048 tokens × 0.01 × 2 switch types × 5120 dim × float16`).  
  For English prompts, I recommend `data_ratio=0.005`.

- **`dataset.batch_size`**  
  Controls the number of constrained runs processed in parallel.  
  Recommended: `4` or `8`. Larger batch sizes parallelize more but may exceed GPU memory or reduce reuse of cached KV.  
  Use `nvidia-smi` to monitor memory usage.

---

### 📌 Final Notes

- **Appending behavior**:  
  The code **always appends** to existing `.h5` and `.csv` files.  
  If you're rerunning with different configurations, make sure to **delete or rename** the existing files beforehand to avoid mixing results.

- **Parallel runs across GPUs**:  
  You can split the dataset and run in parallel on different GPUs (e.g., multiple A100s) by specifying a `data_range`:
  
  ```python
  generate_all_data(data_range=(0, 250), prompt_lang="ch")
  generate_all_data(data_range=(250, 500), prompt_lang="ch")






