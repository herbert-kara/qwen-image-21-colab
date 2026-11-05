import json, os, subprocess, textwrap

ROOT = r"Z:\hermes\projects\qwen-image21-colab"
os.makedirs(ROOT, exist_ok=True)

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}

def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": textwrap.dedent(src).strip("\n").splitlines(keepends=True)}

cells = [
md(r"""
# Qwen-Image-2.1 Turbo — نسخهٔ بدون سانسور روی کولب

**قبل از Run all:** `Runtime → Change runtime type → T4 GPU`.

تولید و ادیت در یک مدل ۷B. انکودر متن «هریتیک» (abliterated) است، پس پرامپت را رد نمی‌کند.

| قطعه | فایل | حجم |
|---|---|---:|
| DiT (turbo، ۸ استپ) | `qwen-image-2.1-turbo-Q4_K_M.gguf` (abenzerps) | 4.05 GB |
| Text encoder هرتیک | `qwen3vl_8b_heretic-Q4_K_M.gguf` (pottokao) | 5.03 GB |
| Vision tower انکودر | `mmproj-qwen3vl_8b_heretic-f16.gguf` | 1.16 GB |
| VAE | `qwen_image_2.1_vae_bf16.safetensors` | 0.68 GB |
| LoRA محتوایی (اختیاری) | `NSFW Qwen Lora.safetensors` (f23gg) | 0.16 GB |

جمع ≈ ۱۱ گیگ → روی T4 ۱۶ گیگی جا می‌شود.

> انکودر هریتیک **باید** همراهش فایل `mmproj` باشد؛ نام هر دو را عوض نکن (پچ، mmproj را از روی نام پیدا می‌کند).
> روی A100/L4 همین نوتبوک را با `Q8_0` برای DiT و انکودر اجرا کن (متغیر `QUANT` در سل ۳).

**اگر GPU نداد (`Cannot connect to GPU backend — usage limits`):** سل‌های ۱→۳ را روی runtime **CPU** اجرا کن؛ مدل‌ها در گوگل‌درایو استیج می‌شوند و بعد هر وقت GPU آزاد شد فقط سل ۱ و ۴ لازم است (~۲ دقیقه). سهمیهٔ رایگان GPU معمولاً تا ~۲۴ ساعت ریست می‌شود؛ ورود با اکانت گوگل دوم هم سهمیهٔ تازه می‌دهد.

**اگر سل ۴ پیام `Port 8188 is already in use` داد، داری نسخهٔ قدیمی نوتبوک را اجرا می‌کنی:** تب کولب را کامل ببند و لینک گیت‌هاب را از نو باز کن.
"""),
code(r"""
# ۱) بررسی GPU و دیسک
import shutil, subprocess
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip() or "NO GPU -> Runtime > Change runtime type > T4 GPU")
import torch; print("torch", torch.__version__, "cuda:", torch.cuda.is_available())
t, u, f = shutil.disk_usage("/content"); print(f"disk free: {f/1e9:.1f} GB")
"""),
code(r"""
%%bash
set -e
cd /content
[ -d ComfyUI ] || git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -q -r requirements.txt
cd custom_nodes
[ -d ComfyUI-GGUF ]            || git clone -q --depth 1 https://github.com/city96/ComfyUI-GGUF
[ -d ComfyUI-GGUF-Qwen3VL-TE ] || git clone -q --depth 1 https://github.com/pottokao-dotcom/ComfyUI-GGUF-Qwen3VL-TE
pip install -q -r ComfyUI-GGUF/requirements.txt
pip install -q hf_transfer
echo "--- custom_nodes ---"; ls
"""),
code(r"""
# ۳) کش در گوگل‌درایو + دانلود مدل‌ها
# بدون GPU هم قابل اجراست (CPU runtime رایگان است): ۱۱ گیگ یک‌بار در Drive می‌نشیند و
# بار بعد که GPU آزاد شد، نوتبوک در ~۲ دقیقه بالا می‌آید، نه ۲۰ دقیقه.
import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

# کش ۱۱ گیگ: اگر Drive mount شد آنجا، وگرنه لوکال runtime.
# drive.mount در اجرای غیرتعاملی/بیسهمیه ValueError میدهد و اگر مهار نشود کل نوتبوک را میکشد.
CACHE = "/content/qwen21-models"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    CACHE = "/content/drive/MyDrive/qwen21-models"
except Exception as e:
    print(f"drive mount نشد ({type(e).__name__}) -> کش لوکال: {CACHE}", flush=True)

QUANT = "Q4_K_M"          # T4 -> Q4_K_M | A100/L4 -> Q8_0
TE_Q = "Q8_0" if QUANT == "Q8_0" else QUANT
M = "/content/ComfyUI/models"

jobs = [
    ("abenzerps/Qwen-Image-2.1-Turbo-Quantized", f"qwen-image-2.1-turbo-{QUANT}.gguf", "unet"),
    ("pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF", f"qwen3vl_8b_heretic-{TE_Q}.gguf", "text_encoders"),
    ("pottokao/Qwen-Image-2.1-Text-Encoder-Heretic-GGUF", "mmproj-qwen3vl_8b_heretic-f16.gguf", "text_encoders"),
    ("abenzerps/Qwen-Image-2.1-Turbo-Quantized", "vae/qwen_image_2.1_vae_bf16.safetensors", "vae"),
    ("f23gg/NSFW-LORA-Qwen-Image-2.1", "NSFW Qwen Lora.safetensors", "loras"),
]
for repo, fn, sub in jobs:
    src = os.path.join(CACHE, sub); os.makedirs(src, exist_ok=True)
    p = hf_hub_download(repo_id=repo, filename=fn, local_dir=src)
    dst_dir = os.path.join(M, sub); os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(fn))   # symlink: ۱۱ گیگ کپی نمی‌شود
    if os.path.lexists(dst): os.remove(dst)
    os.symlink(p, dst)
    print(f"{os.path.getsize(p)/1e9:6.2f} GB  {dst}", flush=True)
print("done")
"""),
code(r"""
# ۴) اجرای ComfyUI + لینک — اگر از قبل بالا باشد دوباره استارت نمی‌زند
import subprocess, sys, threading, time, socket, urllib.request, re

def up():
    try:
        urllib.request.urlopen("http://127.0.0.1:8188/system_stats", timeout=3).close(); return True
    except Exception:
        return False

if not up():
    # هر نسخهٔ ComfyUI را می‌گیرد (با یا بدون --share)؛ الگوی [m] تا خود pkill را نکشد
    subprocess.run('pkill -f "[m]ain.py" || true', shell=True)

    def busy():                       # تا ۱۰ ثانیه صبر می‌کنیم پورت آزاد شود
        s = socket.socket(); s.settimeout(1)
        try: return s.connect_ex(("127.0.0.1", 8188)) == 0
        finally: s.close()
    for _ in range(10):
        if not busy(): break
        time.sleep(1)
    if busy():
        subprocess.run("fuser -k 8188/tcp || true", shell=True); time.sleep(3)

    proc = subprocess.Popen([sys.executable, "main.py", "--port", "8188"],
                            cwd="/content/ComfyUI", stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    def tail(): [print(l, end="", flush=True) for l in proc.stdout]
    threading.Thread(target=tail, daemon=True).start()
    for _ in range(40):       # ~۲ دقیقه: ایمپورت torch و نودها روی T4 اول‌بار کند است
        if up(): break
        time.sleep(3)

if up():
    print("\n>>> ComfyUI روی 127.0.0.1:8188 آماده است. حالا تونل عمومی:", flush=True)
    tun = subprocess.Popen(
        "ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null "
        "-o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -R 80:localhost:8188 serveo.net",
        shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    found = []
    def pump():
        for l in tun.stdout:
            print(l, end="", flush=True)
            m = re.search(r"https://[a-z0-9.-]+serveousercontent\.com", l)
            if m: found.append(m.group(0))
    threading.Thread(target=pump, daemon=True).start()
    for _ in range(25):          # ~۵۰ ثانیه تا serveo آدرس را چاپ کند
        if found: break
        time.sleep(2)
    print("\n>>>>>> لینک ComfyUI:", found[0] if found else "نیامد — لاگِ بالا را بخوان", flush=True)
else:
    print("\n!! سرور روی ۸۱۸۸ جواب نداد — ارور در لاگِ بالاست.", flush=True)
"""),
md(r"""
### چرا serveo و نه پروکسی/iframe کولب

روی همین شبکه اندازه‌گیری شد: `*.colab.googleusercontent.com` (میزبان iframe) تایم‌اوت می‌دهد، لینک `*.prod.r.colab.dev`
خطای «You don't have authorization» می‌دهد و `*.trycloudflare.com` اصلاً DNS حل نمی‌کند — ولی `*.serveousercontent.com`
باز می‌شود (تست‌شده: ۲۰۰ از تونل). پس لینک از سل ۴، تونل serveo است.
"""),
md(r"""
## تنظیمات سَمپلر (در ComfyUI)

**گراف:** `Unet Loader (GGUF)` → `qwen-image-2.1-turbo-Q4_K_M.gguf` · `CLIPLoader (GGUF)` با type = `qwen_image` → `qwen3vl_8b_heretic-Q4_K_M.gguf` · `VAELoader` → vae.
(یا از `Workflow → Browse Templates → Qwen-Image` تمپلیت را بگیر و فقط نود لودر DiT را با نسخهٔ GGUF عوض کن.)

- **۸ استپ**، `cfg 1.0`، sampler `euler`، scheduler `simple`، رزولوشن 1024×1024 (تا 1328). بیشتر از ۸ استپ لازم نیست — مدل turbo است.
- لورای محتوایی: `LoraLoaderModelOnly` با `NSFW Qwen Lora.safetensors` روی **0.8**.
- **متن داخل تصویر** (تابلو/برچسب): دو نود `KSamplerAdvanced` زنجیره‌ای — اولی `cfg 1.0`، `start 0 → end 15`، `return_with_leftover_noise=enable`؛ دومی `add_noise=disable`، `cfg 3.0`، `start 15 → end 25`. یک seed، یک پاس.
- **OOM روی T4؟** سل ۴ را با `--lowvram` اجرا کن (کندتر، ولی ۱۰۰٪ اجرا می‌شود).
- ادیت دستوری (بدون ماسک) همان مدل است: تمپلیت Qwen-Image Edit + لودر GGUF؛ ۱ تا ۳ تصویر مرجع.

**نکته:** ادیت این مدل ماسک ندارد — کاملاً دستوری است. برای اصلاح نقطه‌ای دقیق مناسب نیست.
"""),
]

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
out = os.path.join(ROOT, "Qwen_Image_2.1_Uncensored_Colab.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
json.load(open(out, encoding="utf-8"))  # валидация
print("notebook ok:", out, len(open(out, encoding='utf-8').read()), "bytes")
