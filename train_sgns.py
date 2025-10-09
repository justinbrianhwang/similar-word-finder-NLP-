# ===== train_sgns.py (Kaggle auto-fetch enabled) =====
import os, re, math, random, argparse, json, sys, subprocess
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Kaggle auto-download settings
# -----------------------------
DEFAULT_KAGGLE_DATASETS = [
    # WikiText-103 (tokens 파일 포함)
    "rohitgr/wikitext",                 # wiki.train.tokens 등. :contentReference[oaicite:0]{index=0}
    "vadimkurochkin/wikitext-103",      # 대체 위키텍스트 103 배포. :contentReference[oaicite:1]{index=1}
    # 위키백과 평문(경량/심플)
    "ffatty/plain-text-wikipedia-simpleenglish",  # 심플 영어 위키(평문). :contentReference[oaicite:2]{index=2}
    # 소규모 대안 (텍스트 파일 다수)
    "crawford/20-newsgroups",           # 20NG 텍스트 코퍼스. :contentReference[oaicite:3]{index=3}
]

ALLOWED_EXTS = {".txt", ".tokens", ".text"}  # 학습에 사용할 파일 확장자

def iter_text_files(root):
    """root가 파일이면 해당 파일, 디렉터리면 재귀적으로 텍스트 파일(.txt/.tokens 등)들을 yield."""
    p = Path(root)
    if p.is_file():
        yield str(p)
        return
    if not p.exists():
        return
    for f in p.rglob("*"):
        if f.is_file():
            name = f.name.lower()
            if (f.suffix.lower() in ALLOWED_EXTS) or name.endswith(".tokens"):
                yield str(f)

def dir_has_text_files(root, min_files: int = 1):
    cnt = 0
    for _ in iter_text_files(root):
        cnt += 1
        if cnt >= min_files:
            return True
    return False

def simple_tokenize(line, lower=True):
    if lower: line = line.lower()
    # 알파벳/숫자/하이픈 보존, 나머지 공백화
    line = re.sub(r"[^0-9a-z\-]+", " ", line)
    toks = [t for t in line.strip().split() if t]
    return toks

def build_vocab(data_path, min_count=5, lower=True, limit_files=None):
    cnt = Counter()
    nfiles = 0
    for i, fp in enumerate(iter_text_files(data_path)):
        if limit_files and i >= limit_files: break
        nfiles += 1
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                cnt.update(simple_tokenize(line, lower=lower))
    # min_count 필터
    vocab = [w for w, c in cnt.items() if c >= min_count]
    vocab.sort()
    stoi = {w:i for i, w in enumerate(vocab)}
    itos = vocab
    print(f"[Vocab] files={nfiles}, tokens>={min_count}: {len(vocab)}")
    return stoi, itos, cnt

class SkipGramNS(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.in_embed  = nn.Embedding(vocab_size, dim)
        self.out_embed = nn.Embedding(vocab_size, dim)
        initrange = 0.5 / dim
        nn.init.uniform_(self.in_embed.weight,  -initrange, initrange)
        nn.init.zeros_(self.out_embed.weight)

    def forward(self, center, pos, neg):
        # center: (B,)
        # pos:    (B, P)
        # neg:    (B, N)
        v = self.in_embed(center)            # (B,D)
        pos_v = self.out_embed(pos)          # (B,P,D)
        neg_v = self.out_embed(neg)          # (B,N,D)

        # dot products
        pos_logits = torch.einsum("bd,bpd->bp", v, pos_v)   # (B,P)
        neg_logits = torch.einsum("bd,bnd->bn", v, neg_v)   # (B,N)

        # SGNS loss
        loss = - (F.logsigmoid(pos_logits).sum(dim=1) + F.logsigmoid(-neg_logits).sum(dim=1))
        return loss.mean()

def generate_pairs_from_line(tokens, window=5):
    for i, c in enumerate(tokens):
        left = max(0, i - window)
        right = min(len(tokens), i + window + 1)
        ctx = [tokens[j] for j in range(left, right) if j != i]
        for t in ctx:
            yield c, t

# -----------------------------
# Kaggle helper functions
# -----------------------------
def _pip_install_kaggle():
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        return True, None
    except Exception as e:
        return False, str(e)

def try_download_kaggle_dataset(dataset_slug: str, dest_dir: str, auto_install=True):
    """Kaggle 데이터셋을 dest_dir로 다운로드 & unzip. 성공 시 True."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as e:
        if auto_install:
            ok, err = _pip_install_kaggle()
            if not ok:
                return False, f"pip install kaggle 실패: {err}"
            try:
                from kaggle.api.kaggle_api_extended import KaggleApi
            except Exception as ee:
                return False, f"kaggle 모듈 임포트 실패: {ee}"
        else:
            return False, "kaggle 모듈이 설치되어 있지 않습니다."

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:
        return False, ("Kaggle API 인증 실패. Kaggle에서 API Token 생성 후 "
                       "~/.kaggle/kaggle.json(권한 600) 또는 환경변수 KAGGLE_USERNAME/KAGGLE_KEY 설정 필요. "
                       f"원본 에러: {e}")
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"[Kaggle] downloading '{dataset_slug}' into {dest_dir} ...")
        api.dataset_download_files(dataset_slug, path=dest_dir, unzip=True)
        print("[Kaggle] download completed.")
        return True, None
    except Exception as e:
        return False, str(e)

def ensure_corpus_ready(args) -> bool:
    """--data 경로에 텍스트 파일이 없으면 Kaggle에서 받아와 준비."""
    dp = Path(args.data)
    # 1) 이미 준비됨
    if dp.exists() and (dp.is_file() or dir_has_text_files(dp)):
        print(f"[Data] Using existing corpus at: {dp}")
        return True

    # 2) 자동 다운로드 끔
    if not args.kaggle_auto:
        print(f"[Data] No corpus found at {dp}. (kaggle_auto=False)")
        return False

    # 3) Kaggle에서 가져오기
    datasets = []
    if args.kaggle_dataset and args.kaggle_dataset.lower() != "auto":
        datasets = [args.kaggle_dataset]
    else:
        datasets = list(DEFAULT_KAGGLE_DATASETS)

    for ds in datasets:
        ok, err = try_download_kaggle_dataset(ds, str(dp), auto_install=args.auto_install_kaggle)
        if ok:
            # 방금 받은 폴더에서 텍스트 파일 찾기
            if dir_has_text_files(dp):
                print(f"[Data] Ready (downloaded from Kaggle: {ds})")
                return True
            else:
                print(f"[Data] Downloaded {ds} but no *.txt/*.tokens found. Try next candidate...")
        else:
            print(f"[Kaggle] Failed to download {ds}: {err}")

    print("[Data] Unable to prepare corpus automatically. "
          "수동으로 --data 경로에 텍스트 파일을 넣거나 --kaggle_dataset 로 다른 슬러그를 지정하세요.")
    return False

# -----------------------------
# Training
# -----------------------------
def train(args):
    # 코퍼스 준비(필요시 Kaggle에서 자동 다운로드)
    if not ensure_corpus_ready(args):
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    stoi, itos, counter = build_vocab(args.data, min_count=args.min_count, lower=args.lower, limit_files=args.limit_files)
    V = len(stoi)
    if V < 1000:
        print("Warning: very small vocab; consider lowering min_count or adding data.")

    # Unigram noise distribution ^0.75
    freqs = torch.zeros(V)
    for w, c in counter.items():
        if w in stoi: freqs[stoi[w]] = c
    noise = (freqs.pow(0.75) / freqs.pow(0.75).sum()).to(device)

    model = SkipGramNS(V, args.dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    g_step = 0
    for epoch in range(1, args.epochs + 1):
        rng = random.Random(epoch)
        files = list(iter_text_files(args.data))
        if args.limit_files:
            files = files[:args.limit_files]
        rng.shuffle(files)

        running = 0.0
        n_batches = 0
        for fp in files:
            with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                buf_center, buf_pos = [], []
                for line in f:
                    toks = [t for t in simple_tokenize(line, lower=args.lower) if t in stoi]
                    if len(toks) < 2: continue
                    for c, t in generate_pairs_from_line(toks, window=args.window):
                        buf_center.append(stoi[c])
                        buf_pos.append(stoi[t])
                        # 미니배치가 충분히 쌓이면 업데이트
                        if len(buf_center) >= args.batch_size:
                            center = torch.tensor(buf_center[:args.batch_size], device=device)
                            pos    = torch.tensor(buf_pos[:args.batch_size],    device=device).unsqueeze(1)  # (B,1)

                            # negative sampling
                            neg = torch.multinomial(noise, num_samples=args.batch_size * args.negatives, replacement=True)
                            neg = neg.view(args.batch_size, args.negatives)

                            loss = model(center, pos, neg)
                            opt.zero_grad(set_to_none=True)
                            loss.backward()
                            if args.grad_clip:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                            opt.step()

                            running += loss.item()
                            n_batches += 1
                            g_step += 1

                            # reset buffer
                            buf_center, buf_pos = [], []

                            if g_step % args.log_every == 0:
                                print(f"epoch {epoch} step {g_step}  loss {running/n_batches:.4f}")
                                running = 0.0; n_batches = 0

                            # quick sanity check
                            if args.quick_eval_every and (g_step % args.quick_eval_every == 0):
                                with torch.no_grad():
                                    W = F.normalize(model.in_embed.weight.detach(), p=2, dim=1)
                                    def topk(word, k=5):
                                        if word not in stoi: return []
                                        i = stoi[word]
                                        s = W @ W[i]
                                        s[i] = -1
                                        vals, inds = torch.topk(s, k)
                                        return [(itos[j], float(vals[a])) for a, j in enumerate(inds.tolist())]
                                    print("NN(september):", topk("september", 8))
                                    print("NN(october):  ", topk("october", 8))

        # epoch 끝나면 체크포인트 저장
        out_path = os.path.join(args.out_dir, f"embeddings_epoch{epoch}.pth")
        os.makedirs(args.out_dir, exist_ok=True)
        torch.save({
            "emb_weight": model.in_embed.weight.detach().cpu(),
            "stoi": stoi,
            "itos": itos,
            "dim": args.dim,
            "meta": {
                "min_count": args.min_count,
                "window": args.window,
                "negatives": args.negatives,
                "lower": args.lower,
                "epochs": epoch
            }
        }, out_path)
        print("Saved:", out_path)

    print("Training done.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="텍스트 파일 또는 폴더 경로(없으면 Kaggle에서 자동 다운로드)")
    ap.add_argument("--out_dir", type=str, default="./ckpts", help="체크포인트 출력 폴더")
    ap.add_argument("--dim", type=int, default=200)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--negatives", type=int, default=5)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--lower", action="store_true", help="소문자화 적용")
    ap.add_argument("--limit_files", type=int, default=None, help="디버깅용 파일 수 제한")
    ap.add_argument("--log_every", type=int, default=500)
    ap.add_argument("--quick_eval_every", type=int, default=5000, help="빠른 최근접 단어 점검 주기(step)")
    ap.add_argument("--grad_clip", type=float, default=5.0)

    # Kaggle 옵션
    ap.add_argument("--kaggle_dataset", type=str, default="auto",
                    help="Kaggle 데이터셋 슬러그(e.g., 'rohitgr/wikitext') 또는 'auto'")
    ap.add_argument("--kaggle_auto", dest="kaggle_auto", action="store_true",
                    help="데이터가 없으면 Kaggle에서 자동 다운로드(기본 on)")
    ap.add_argument("--no_kaggle_auto", dest="kaggle_auto", action="store_false",
                    help="자동 다운로드 끄기")
    ap.set_defaults(kaggle_auto=True)

    ap.add_argument("--auto_install_kaggle", dest="auto_install_kaggle", action="store_true",
                    help="kaggle 모듈이 없으면 pip로 자동 설치(기본 on)")
    ap.add_argument("--no_auto_install_kaggle", dest="auto_install_kaggle", action="store_false",
                    help="자동 설치 끄기")
    ap.set_defaults(auto_install_kaggle=True)

    args = ap.parse_args()
    train(args)
# ===== end =====
