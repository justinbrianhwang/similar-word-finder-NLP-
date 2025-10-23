# Similar Word Finder (Skip-Gram with Negative Sampling)

This repository implements a **Skip-Gram Negative Sampling (SGNS)** model — a Word2Vec variant — to learn word embeddings and find semantically similar words.  
It can automatically fetch large-scale text corpora (e.g., WikiText) from Kaggle for training.

<img width="348" height="209" alt="image" src="https://github.com/user-attachments/assets/591face9-427b-48a2-bedb-bb142b28c665" />


<img width="561" height="528" alt="image" src="https://github.com/user-attachments/assets/37747a1d-e81b-473b-829b-449ebb7bc908" />


<img width="717" height="361" alt="image" src="https://github.com/user-attachments/assets/0ba328e6-f57e-47e4-9908-97ea768f1435" />


---

## 🚀 Features
- ✅ Auto-downloads text corpora from Kaggle (WikiText-103, 20-Newsgroups, etc.)
- 🧠 Custom tokenizer (simple English normalization)
- ⚡ PyTorch-based Skip-Gram Negative Sampling implementation
- 💾 Periodic checkpoint saving
- 🔍 Built-in quick evaluation of nearest words during training

---

## 📦 Installation
```bash
git clone https://github.com/justinbrianhwang/similar-word-finder-NLP-.git
cd similar-word-finder
pip install -r requirements.txt
```

## 🧮 Usage
1️⃣ Train

```bash
python train_sgns.py --data ./data \
    --out_dir ./ckpts \
    --epochs 2 \
    --dim 200 \
    --lower \
    --kaggle_auto
```

- If --data path is empty, it will auto-download a text dataset from Kaggle (WikiText, 20NG, etc.)
- Trained embeddings are saved as .pth files under ckpts/

2️⃣ Example: Find Similar Words
```bash
import torch, torch.nn.functional as F

ckpt = torch.load("ckpts/embeddings_epoch2.pth", map_location="cpu")
W = F.normalize(ckpt["emb_weight"], p=2, dim=1)
stoi, itos = ckpt["stoi"], ckpt["itos"]

def topk(word, k=5):
    if word not in stoi: return []
    i = stoi[word]
    s = W @ W[i]
    s[i] = -1
    vals, inds = torch.topk(s, k)
    return [(itos[j], float(vals[a])) for a, j in enumerate(inds.tolist())]

print(topk("computer", 10))
```

## ⚙️ Parameters
| Option               | Description                              |    Default   |
| :------------------- | :--------------------------------------- | :----------: |
| `--data`             | Path to text folder or file              | *(required)* |
| `--dim`              | Embedding dimension                      |      200     |
| `--window`           | Context window size                      |       5      |
| `--negatives`        | # of negative samples                    |       5      |
| `--min_count`        | Min word freq                            |       5      |
| `--batch_size`       | Training batch size                      |     4096     |
| `--epochs`           | # of epochs                              |       2      |
| `--lower`            | Lowercase normalization                  |    `False`   |
| `--grad_clip`        | Gradient clip value                      |      5.0     |
| `--quick_eval_every` | Step interval for quick similarity check |     5000     |

## Dependencies
```txt
torch
numpy
kaggle
```






