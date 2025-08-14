from transformers import AutoModelForCausalLM, AutoTokenizer
from strategy import Strategy
from vad import PredEmo
import re
from steering import interactive_eval, calculate_means, calculate_weighted_means,interactive_eval_autocast
from dotenv import load_dotenv
import math
import torch
import torch.nn as nn
import numpy as np
from utils.llm_model_utils import load_llm_model_with_insertions
import os
import pickle

load_dotenv()
strategy = Strategy()
predemo = PredEmo()

def force_cast_model_dtype(model: torch.nn.Module, target_dtype: torch.dtype):
    """
    モデル内の全ての Parameter と Buffer を target_dtype に強制キャストする。
    register_buffer にドット名を渡さないよう、親モジュールを辿って buffer を置き換える。
    量子化レイヤ等の特殊Paramは dtype変更できない場合があるので try/except でスキップ。
    """
    import torch.nn as nn

    # Parameters
    for name, p in model.named_parameters(recurse=True):
        try:
            if p is not None and getattr(p, "dtype", None) is not None and p.dtype.is_floating_point:
                if p.dtype is not target_dtype:
                    with torch.no_grad():
                        p.data = p.data.to(dtype=target_dtype)
        except Exception:
            # 量子化Paramなどで失敗したらスキップ
            pass

    # Buffers
    for full_name, b in model.named_buffers(recurse=True):
        try:
            if b is None:
                continue
            if getattr(b, "dtype", None) is None or not b.dtype.is_floating_point:
                continue
            if b.dtype is target_dtype:
                continue

            # 親モジュールを取得して、ドット無しの末尾名だけで設定する
            if "." in full_name:
                parent_path, buf_name = full_name.rsplit(".", 1)
                parent = model.get_submodule(parent_path)
            else:
                parent = model
                buf_name = full_name

            new_b = b.to(dtype=target_dtype)
            parent._buffers[buf_name] = new_b
        except Exception:
            # 一部のバッファは読み取り専用など。失敗したらスキップ
            pass


def dump_model_dtypes(model, name="model"):
    from collections import Counter
    c = Counter(p.dtype for p in model.parameters())
    print(f"[{name}] dtype distribution:", {str(k): v for k, v in c.items()})
    try:
        print(f"[{name}] first param dtype/device:", next(model.parameters()).dtype, next(model.parameters()).device)
    except StopIteration:
        print(f"[{name}] has no parameters?")


def _model_dtype_device(model: torch.nn.Module):
    p = next(model.parameters())
    return p.dtype, p.device

def extract_translation(content):
    # 最後の <|message|> の位置を見つけて、その後の内容を取得
    last_message_match = None
    for match in re.finditer(r'<\|message\|>', content):
        last_message_match = match
    
    if last_message_match:
        start_pos = last_message_match.end()
        remaining = content[start_pos:]
        return_match = re.search(r'<\|return\|>', remaining)
        if return_match:
            return remaining[:return_match.start()].strip()
        else:
            return remaining.strip()
    else:
        return None

def transrate(prompt, model, tokenizer, device):
    messages = [
        {"role":"system","content":"あなたは優秀な翻訳者です。与えられた文章を日本語のニュアンスを残したまま、英語に翻訳してください。翻訳だけをすればいいです。"},
        {"role":"user","content":prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer([text],return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=32768
    )
    content = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:])
    content = extract_translation(content)
    return content

def create_strategy_messages(history,strategy, user_prompt):
    strategies = {
            "@[Question]": "質問を通じて相手の考えや気持ちを深く理解し、対話を促進する戦略",
            "@[Greeting]": "親しみやすい挨拶や雰囲気作りを通じて、良好な関係性を築く戦略", 
            "@[Restatement or Paraphrasing]": "相手の発言を言い換えて確認し、理解を示す戦略",
            "@[Reflection of feelings]": "相手の感情を察知し、共感的に反映する戦略",
            "@[Self-disclosure]": "適切な自己開示を通じて親近感を生み出す戦略",
            "@[Affirmation and Reassurance]": "相手を肯定し、安心感を与える戦略",
            "@[Providing Suggestions or Information]": "有用な提案や情報を提供する戦略",
            "@[Others]": "その他の柔軟なアプローチを用いる戦略"
        }
    history = " ".join(history) if isinstance(history, (list, tuple)) else (history or "")
        
    system_content = f"""以下の会話戦略に従って返答を生成してください：

        選択された戦略: {strategy}
        戦略の説明: {strategies.get(strategy, "未定義の戦略")}

        この戦略に基づいて、ユーザーのメッセージに対して適切で効果的な返答を生成してください。
        返答は自然で人間らしく、相手との良好なコミュニケーションを促進するものにしてください。
        返答は簡潔にしてください。"""

    messages = [
        {"role":"system","content":history},
        {"role":"system", "content": system_content},
        {"role":"system","content":"あなたは18歳の女の子として振舞ってください。返答はできるだけ短文で日本語で生成してください。"},
        {"role": "user", "content": user_prompt}
    ]
    return messages

def chat(history,strategy,prompt, model, tokenizer, device):
    messages = create_strategy_messages(history,strategy,prompt)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    inputs = tokenizer([text],return_tensors="pt").to(device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=32768
    )
    content = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:])
    content = extract_translation(content)
    print(content)
    return content

def interactive_eval_autocast_multi_gpu(
    emotions, pred_emo, lam, history, strategy, prompt,
    means, ovr_r_means, total_mean, llm_model, tokenizer,
    insertion_layers, save_path, device,
    target="hidden",           # "hidden"（既定） or "mlp"
    mlp_out_sizes=None         # target="mlp" のときのみ使用（例: {15: 11008, 16: 11008, 17: 11008}）
):
    import torch
    import torch.nn as nn
    import numpy as np
    import traceback
    from contextlib import nullcontext

    # ========== utils ==========
    def _model_dtype_device(model):
        p = next(model.parameters())
        return p.dtype, p.device

    def _hidden_size_from_model(model):
        hs = getattr(getattr(model, "config", object()), "hidden_size", None)
        if isinstance(hs, int) and hs > 0:
            return hs
        try:
            emb = getattr(model.model, "embed_tokens", None)
            if emb is not None and hasattr(emb, "weight"):
                return int(emb.weight.shape[1])
        except Exception:
            pass
        lyr0 = getattr(model.model, "layers", [None])[0]
        if lyr0 is not None:
            o_proj = getattr(getattr(lyr0, "self_attn", object()), "o_proj", None)
            if o_proj is not None and hasattr(o_proj, "out_features"):
                return int(o_proj.out_features)
        raise RuntimeError("hidden_size を推定できませんでした。")

    def _robust_mlp_out_features(mlp):
        # 代表的な名前を優先
        for name in ("up_proj", "gate_proj", "fc1", "w1"):
            sub = getattr(mlp, name, None)
            if sub is not None and hasattr(sub, "weight"):
                w = getattr(sub, "weight", None)
                if isinstance(w, torch.Tensor) and w.ndim == 2:
                    return int(w.shape[0])
        # 全 Linear 走査
        cands = []
        for mod in mlp.modules():
            w = getattr(mod, "weight", None)
            if isinstance(w, torch.Tensor) and w.ndim == 2:
                cands.append(int(w.shape[0]))
        if cands:
            return max(cands)
        raise RuntimeError("MLP out_features を推定できませんでした。")

    def _set_steering(mlp_module, sv_tensor, lam_value, m_dtype, m_device):
        mlp_module.steering_vector = nn.Parameter(sv_tensor, requires_grad=False)
        lam_tensor = torch.tensor(lam_value, dtype=m_dtype, device=m_device)
        try:
            mlp_module.register_buffer("steer_lambda", lam_tensor, persistent=False)
        except Exception:
            mlp_module.steer_lambda = lam_tensor

    def _ensure_hook(mlp_module, add_dim):
        # add_dim: 加算先の最終次元（hidden or mlp）
        if hasattr(mlp_module, "_steer_hook_handle") and mlp_module._steer_hook_handle is not None:
            return

        def steering_hook(module, inputs, output):
            sv = getattr(module, "steering_vector", None)
            lam = getattr(module, "steer_lambda", None)
            if sv is None or lam is None:
                return output

            # 出力テンソルを取得（tuple の場合は先頭）
            out = output[0] if isinstance(output, tuple) else output

            # dtype/device 揃え
            sv = sv.to(out.dtype).to(out.device)
            lam = lam.to(out.dtype).to(out.device)

            # 形状調整
            if sv.numel() != add_dim:
                # ここに来たら設定ミス
                raise RuntimeError(f"steering_vector size {sv.numel()} != expected {add_dim}")
            add = (lam * sv).reshape(1, 1, -1)  # [1,1,add_dim]

            out = out + add
            if isinstance(output, tuple):
                return (out,) + tuple(output[1:])
            return out

        mlp_module._steer_hook_handle = mlp_module.register_forward_hook(steering_hook)

    # ========== 入力整形 ==========
    messages = create_strategy_messages(history, strategy, prompt)
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    

    # ========== dtype/device ==========
    m_dtype, m_device = _model_dtype_device(llm_model)

    # ========== ベクトル差分 ==========
    emo_idx = emotions.index(pred_emo)
    
    m = np.asarray(means[emo_idx], dtype=np.float32).reshape(-1)
    o = np.asarray(ovr_r_means[emo_idx], dtype=np.float32).reshape(-1)
    diff = m - o
    
    # ========== どの次元に加算するか ==========
    if target == "hidden":
        hidden = _hidden_size_from_model(llm_model)  # ← 今回は 2880 が取得されている
        per_layer = hidden
        expected_total = per_layer * len(insertion_layers)
        if diff.size != expected_total:
            raise ValueError(
                f"diff size mismatch: diff.size={diff.size}, "
                f"required={expected_total} (hidden={hidden} x layers={len(insertion_layers)})"
            )
        add_dims = [per_layer] * len(insertion_layers)

    elif target == "mlp":
        mlp_sizes = []
        for idx in insertion_layers:
            mlp = getattr(llm_model.model.layers[idx], "mlp", None)
            if mlp is None:
                raise TypeError(f"layer[{idx}].mlp is None; expected nn.Module")
            size = int(mlp_out_sizes[idx]) if (mlp_out_sizes and idx in mlp_out_sizes) else _robust_mlp_out_features(mlp)
            mlp_sizes.append(size)
        expected_total = int(sum(mlp_sizes))
        if diff.size != expected_total:
            raise ValueError(
                f"diff size mismatch: diff.size={diff.size}, required={expected_total} "
                f"(per-layer sizes={mlp_sizes})"
            )
        add_dims = mlp_sizes
    else:
        raise ValueError("target must be 'hidden' or 'mlp'.")

    # ========== 正確にスライスして注入 ==========
    offsets = np.cumsum([0] + add_dims)
    chunks = [diff[offsets[i]:offsets[i+1]] for i in range(len(add_dims))]

    for n, layer_idx in enumerate(insertion_layers):
        layer = llm_model.model.layers[layer_idx]
        mlp = getattr(layer, "mlp", None)
        if mlp is None or isinstance(mlp, (tuple, list)):
            raise TypeError(f"layer[{layer_idx}].mlp is {type(mlp)}; expected nn.Module")

        sv = torch.tensor(chunks[n], dtype=m_dtype, device=m_device).flatten().contiguous()

        _set_steering(mlp, sv, lam, m_dtype, m_device)
        _ensure_hook(mlp, add_dim=add_dims[n])

        lam_attr = getattr(mlp, "steer_lambda", None)

    # ========== 生成 ==========
    amp_dtype = torch.bfloat16 if m_dtype == torch.bfloat16 else (torch.float16 if m_dtype == torch.float16 else None)
    ctx = (torch.autocast(device_type=str(m_device).split(":")[0], dtype=amp_dtype)
           if amp_dtype else nullcontext())
    try:
        with torch.inference_mode(), ctx:
            generated_ids = llm_model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_new_tokens=32768,
            )
        
    except Exception:
        traceback.print_exc()
        raise

    content = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[-1]:])
    content = extract_translation(content)
    print(f"生成文：{content}")
    return content





def calculate_lam(mood, emo):
    Emotion_dict = {
        'anger':   [-0.51, 0.59, 0.25],
        'disgust': [-0.60, 0.35, 0.11],
        'fear':    [-0.62, 0.82, -0.43],
        'joy':     [0.81, 0.51, 0.46],
        'neutral': [ 0.0,  0.0,  0.0],
        'sadness': [-0.63, -0.27, -0.33],
        'surprise':[ 0.40, 0.67, -0.13]
    }
    k = 0.5
    d = math.sqrt(sum((a - b)**2 for a, b in zip(Emotion_dict[emo], mood)))
    lam = 2 / (1 + k * d)
    return lam

def load_activations(vector_path):
    with open('/workspace/santa/StyleVectorsChatBot-jp/output/activations/weighted_activations_train.pkl','rb') as f:
        train = pickle.load(f)

    with open('/workspace/santa/StyleVectorsChatBot-jp/output/activations/weighted_activations_test.pkl','rb') as f:
        test = pickle.load(f)

    return train,test

def check_gpu_availability():
    """GPUの使用可能性をチェック"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    gpu_count = torch.cuda.device_count()
    print(f"Available GPUs: {gpu_count}")
    
    if gpu_count < 2:
        print("Warning: Less than 2 GPUs available. Using available GPUs.")
        return min(2, gpu_count)
    
    return 2

def setup_devices():
    """デバイスの設定"""
    gpu_count = check_gpu_availability()
    
    # GPU使用量をチェック
    for i in range(gpu_count):
        try:
            torch.cuda.set_device(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
            memory_reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
            memory_total = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            memory_free = memory_total - memory_reserved
            print(f"GPU {i}: Total: {memory_total:.2f}GB, Reserved: {memory_reserved:.2f}GB, Free: {memory_free:.2f}GB")
        except Exception as e:
            print(f"GPU {i}: Error checking memory - {e}")
    
    # デバイスの割り当て - メモリ使用量に基づいて決定
    if gpu_count >= 2:
        # GPU 1のメモリ使用量をチェック
        try:
            torch.cuda.set_device(1)
            memory_reserved_gpu1 = torch.cuda.memory_reserved(1) / (1024**3)
            memory_total_gpu1 = torch.cuda.get_device_properties(1).total_memory / (1024**3)
            memory_free_gpu1 = memory_total_gpu1 - memory_reserved_gpu1
            
            if memory_free_gpu1 > 8.0:  # 8GB以上の空きがある場合のみGPU 1を使用
                DEVICE_MAIN = torch.device("cuda:0")
                DEVICE_TRANSLATE = torch.device("cuda:1")
                print("Using GPU 0 for main model and GPU 1 for translation model")
            else:
                print(f"GPU 1 has insufficient memory ({memory_free_gpu1:.2f}GB free), using GPU 0 for both models")
                DEVICE_MAIN = torch.device("cuda:0")
                DEVICE_TRANSLATE = torch.device("cuda:0")
        except Exception as e:
            print(f"Error checking GPU 1 memory, using GPU 0 for both models: {e}")
            DEVICE_MAIN = torch.device("cuda:0")
            DEVICE_TRANSLATE = torch.device("cuda:0")
    else:
        DEVICE_MAIN = torch.device("cuda:0")
        DEVICE_TRANSLATE = torch.device("cuda:0")
        print("Using single GPU for both models")
    
    return DEVICE_MAIN, DEVICE_TRANSLATE

# メイン処理
def main():
    # GPU メモリをクリア
    torch.cuda.empty_cache()
    
    # PyTorch CUDA memory allocation configuration
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # デバイス設定
    DEVICE_MAIN, DEVICE_TRANSLATE = setup_devices()
    INSERTION_LAYERS = [15,16,17]
    
    print("Loading main model...")
    # メインモデル（チャット用）をGPU 0に配置
    main_model, main_tokenizer = load_llm_model_with_insertions(DEVICE_MAIN, INSERTION_LAYERS)
    dump_model_dtypes(main_model, "main_model")
    print("Loading translation model...")
    # 翻訳モデルの設定
    translate_model_id = "/workspace/santa/emo-chat/model/openai"
    translate_tokenizer = AutoTokenizer.from_pretrained(translate_model_id)

    # メモリ効率的な読み込み
    translate_model = None
    try:
        if DEVICE_TRANSLATE.index != DEVICE_MAIN.index:  # 異なるGPUを使用する場合
            # GPU メモリをクリア
            torch.cuda.empty_cache()
            torch.cuda.set_device(DEVICE_TRANSLATE)
            
            translate_model = AutoModelForCausalLM.from_pretrained(
                translate_model_id,
                torch_dtype=torch.float16,  # メモリ使用量を削減
                device_map={"": DEVICE_TRANSLATE},  # 直接デバイスを指定
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print(f"Translation model loaded on {DEVICE_TRANSLATE}")
        else:
            # 同じGPUを使用する場合
            translate_model = AutoModelForCausalLM.from_pretrained(
                translate_model_id,
                torch_dtype=torch.float16,
                device_map={"": DEVICE_TRANSLATE},
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            print(f"Both models loaded on {DEVICE_TRANSLATE}")
            
    except Exception as e:
        print(f"GPU model loading error: {e}")
        print("Trying CPU-only loading with explicit settings...")
        
        # 完全にCPUに強制する
        try:
            # 環境変数でCUDAを無効化
            old_cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            translate_model = AutoModelForCausalLM.from_pretrained(
                translate_model_id,
                torch_dtype=torch.float32,  # CPUではfloat32を使用
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            DEVICE_TRANSLATE = torch.device("cpu")
            print("Translation model successfully loaded on CPU")
            
            # 環境変数を復元
            if old_cuda_visible is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible
            else:
                del os.environ["CUDA_VISIBLE_DEVICES"]
                
        except Exception as e2:
            print(f"CPU loading also failed: {e2}")
            print("Using main model for translation as well...")
            translate_model = main_model
            DEVICE_TRANSLATE = DEVICE_MAIN
            print("Using main model for both chat and translation")
    target_dtype = torch.bfloat16   # ← ここを torch.float16 にすれば fp16 統一にできます
    main_model.to(device=DEVICE_MAIN)          # 念のため device を先に合わせる
    translate_model.to(device=DEVICE_TRANSLATE)

    force_cast_model_dtype(main_model, target_dtype)
    force_cast_model_dtype(translate_model, target_dtype)

    dump_model_dtypes(main_model, "main_model (after cast)")
    dump_model_dtypes(translate_model, "translate_model (after cast)")
    print("Loading activations...")
    # アクティベーションデータの読み込み
    ACTIVATION_VECTOR_PATH = os.getenv("ACTIVATIONS_PATH")
    SAVE_PATH = os.getcwd()
    train, test = load_activations(ACTIVATION_VECTOR_PATH)

    train = [entry for entry in train if len(entry)==3]
    test = [entry for entry in test if len(entry)==3]

    emotion_names = ['joy', 'sadness', 'surprise', 'anger', 'fear', 'disgust', 'trust']
    # emotion_idx は未使用なので削除してもOK
    # emotion_idx = [0,1,2,3,4,5,6,7]

    means, ovr_r_means, total_mean = calculate_weighted_means(train, test, emotion_names, INSERTION_LAYERS)
    
    print("Setup complete. Starting chat loop...")
    print(f"Main model device: {DEVICE_MAIN}")
    print(f"Translation model device: {DEVICE_TRANSLATE}")
    
    # トークナイザーの設定
    if translate_model == main_model:
        translate_tokenizer = main_tokenizer
    
    # チャットループ
    utt1 = ""
    emo = ["joy"]
    personality = [0.6, 0.48, 0.31, 0.46, 0.56]
    mood = [1.0, 1.0 , 0.0]
    history = []
    
    while True:
        try:
            prompt = input("入力：")
            print("prompt:",prompt)
            history.append(f"user:{prompt}")
            
            # 翻訳（指定されたデバイスで実行）
            content = transrate(prompt, translate_model, translate_tokenizer, DEVICE_TRANSLATE)
            print("content:",content)
            
            # 戦略予測
            pred_strategy = strategy.pred_strategy(content)
            print("pred_strategy:",pred_strategy)
            
            # 感情予測
            emo, mood = predemo.pred_emo(utt1, content, emo, personality, mood)
            mood = mood.squeeze(0).tolist()
            lam = calculate_lam(mood, emo)
            if emo == "neutral":
                emo = "trust"
            emo = [emo]
            print("emo:",emo)
            print("mood:",mood)
            print("lam:",lam)
            # チャット生成（GPU 0で実行）
            utt1 = interactive_eval_autocast_multi_gpu(
                emotion_names, emo[0], lam, history, pred_strategy, prompt, 
                means, ovr_r_means, total_mean, llm_model=main_model, 
                tokenizer=main_tokenizer, insertion_layers=INSERTION_LAYERS, 
                save_path=SAVE_PATH, device=DEVICE_MAIN
            )
            history.append(f"system:{utt1}")
            
            
            # 逆翻訳（指定されたデバイスで実行）
            utt1 = transrate(utt1, translate_model, translate_tokenizer, DEVICE_TRANSLATE)
            
        except KeyboardInterrupt:
            print("\nChat terminated by user.")
            break
        except Exception as e:
            print(f"Error occurred: {e}")
            continue

if __name__ == "__main__":
    main()
