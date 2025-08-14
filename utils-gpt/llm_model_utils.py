import os
import transformers
from dotenv import load_dotenv
import torch
import torch.nn as nn

load_dotenv()

# ---- ユーティリティ ----
def _device_to_map_value(device: torch.device) -> str:
    if device.type == "cuda":
        idx = 0 if device.index is None else device.index
        return f"cuda:{idx}"
    return "cpu"

def _model_dtype(model: torch.nn.Module):
    try:
        return next(model.parameters()).dtype
    except StopIteration:
        return torch.float16

def inspect_model_architecture(model, layer_idx=0):
    print(f"\n=== モデルアーキテクチャの調査 ===")
    print(f"モデルクラス: {type(model).__name__}")
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        print(f"レイヤー数: {len(layers)}")
        if len(layers) > layer_idx:
            layer = layers[layer_idx]
            print(f"レイヤー {layer_idx} のクラス: {type(layer).__name__}")
            if hasattr(layer, 'mlp'):
                mlp = layer.mlp
                print(f"MLP クラス: {type(mlp).__name__}")
                print(f"MLP 属性: {[attr for attr in dir(mlp) if not attr.startswith('_')]}")
                important_attrs = ['gate_proj','up_proj','down_proj','o_proj','fc1','fc2','dense']
                existing = [a for a in important_attrs if hasattr(mlp, a)]
                print(f"既存の重要な属性: {existing}")
    print("=== 調査終了 ===\n")

# ---- Steering Layer 実装（MoE/tuple出力対応、dtype/device 追従、衝突回避）----
def _find_primary_tensor(x):
    """
    x が Tensor/tuple/list のいずれでも、最初の Tensor を返す。
    戻り値: (main_tensor, index, container)
      - x が Tensor: (x, None, None)
      - x が tuple/list: (最初のTensor, そのindex, x)
      - 見つからない: (None, None, x)
    """
    if isinstance(x, torch.Tensor):
        return x, None, None
    if isinstance(x, (tuple, list)):
        for i, item in enumerate(x):
            if isinstance(item, torch.Tensor):
                return item, i, x
    return None, None, x

class _BaseSteeringWrapper(nn.Module):
    """
    元MLPの前段/後段を変更せず、出力テンソル（もしくは出力タプルの第1テンソル）の
    最終次元（hidden）に (steer_lambda * steering_vector) を加算する。
    """
    def __init__(self, original_mlp: nn.Module):
        super().__init__()
        self.original_mlp = original_mlp
        # 後で set_steering() で正式登録するため、最初は None にしておく
        self.steering_vector = None  # type: nn.Parameter | None
        # 係数は buffer として保持（optimizer 対象外）
        self.register_buffer("steer_lambda", torch.tensor(0.0), persistent=False)

    def set_steering(self, vec: torch.Tensor, lam: float | torch.Tensor):
        """
        vec: shape [hidden] の Tensor（Parameter化し登録）
        lam: 係数（float/Tensor）
        """
        if not isinstance(vec, torch.Tensor):
            raise TypeError("steering vector must be a torch.Tensor")
        # 既存を置換（requires_grad=False）
        param = nn.Parameter(vec.detach().contiguous(), requires_grad=False)
        self.steering_vector = param
        # 正式に Parameter として登録
        self.register_parameter("steering_vector", self.steering_vector)

        lam_t = lam if isinstance(lam, torch.Tensor) else torch.tensor(lam)
        self.steer_lambda = lam_t.to(dtype=param.dtype, device=param.device)

    def _apply_effect(self, out):
        """
        out が Tensor でも (Tensor, aux...) でも動作。
        main tensor の最後の次元（hidden）に加算。
        """
        if (self.steering_vector is None) or (self.steer_lambda is None):
            return out

        main, idx, container = _find_primary_tensor(out)
        if main is None:
            return out

        sv = self.steering_vector.to(dtype=main.dtype, device=main.device)
        lam = self.steer_lambda.to(dtype=main.dtype, device=main.device)

        hidden = main.shape[-1]
        if sv.numel() != hidden:
            raise RuntimeError(f"steering_vector size {sv.numel()} != hidden {hidden}")

        # [*, hidden] にブロードキャスト（B,T,…の手前次元に 1 を立てる）
        add = (lam * sv).reshape(*([1] * (main.ndim - 1)), hidden)
        main2 = main + add

        if idx is None:
            return main2
        if isinstance(container, tuple):
            lst = list(container)
            lst[idx] = main2
            return tuple(lst)
        container[idx] = main2
        return container

    def forward(self, x, *args, **kwargs):
        out = self.original_mlp(x, *args, **kwargs)
        return self._apply_effect(out)

class AdaptiveLLaMASteeringLayer(_BaseSteeringWrapper):
    pass

class AdaptiveGPTSteeringLayer(_BaseSteeringWrapper):
    pass

class AdaptiveBERTSteeringLayer(_BaseSteeringWrapper):
    pass

class GenericSteeringLayer(_BaseSteeringWrapper):
    def __init__(self, original_mlp: nn.Module):
        super().__init__(original_mlp)

def create_adaptive_steering_layer(original_mlp: nn.Module) -> nn.Module:
    """
    モデル実装ごとに代表的な属性からラッパークラスを選択。
    どれでもなければ Generic を使う（動作は同一）。
    """
    mlp = original_mlp
    if hasattr(mlp, 'gate_proj') and hasattr(mlp, 'up_proj') and hasattr(mlp, 'down_proj'):
        return AdaptiveLLaMASteeringLayer(mlp)
    elif hasattr(mlp, 'fc1') and hasattr(mlp, 'fc2'):
        return AdaptiveGPTSteeringLayer(mlp)
    elif hasattr(mlp, 'dense'):
        return AdaptiveBERTSteeringLayer(mlp)
    else:
        return GenericSteeringLayer(mlp)

# ---- モデル読み込み（bnbは“使えたら使う”、ダメならfp16） ----
def load_llm_model(device: torch.device):
    MODEL_WEIGHTS_FOLDER = os.getenv("MODEL_WEIGHTS_FOLDER")

    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")

    HAS_BNB = False
    try:
        import bitsandbytes as bnb  # noqa: F401
        print(f"bitsandbytes version: {getattr(bnb, '__version__', 'unknown')}")
        HAS_BNB = True
    except Exception:
        print("bitsandbytes not installed")
        HAS_BNB = False

    llm_model = None

    # 1) 4bit
    if HAS_BNB:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            llm_model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_WEIGHTS_FOLDER,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map=_device_to_map_value(device),
            )
            print("✓ 4bit量子化でモデル読み込み成功")
        except Exception as e:
            print(f"✗ 4bit量子化でエラー: {e}")

    # 2) 8bit
    if llm_model is None and HAS_BNB:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            llm_model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_WEIGHTS_FOLDER,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map=_device_to_map_value(device),
            )
            print("✓ 8bit量子化でモデル読み込み成功")
        except Exception as e:
            print(f"✗ 8bit量子化でエラー: {e}")

    # 3) 非量子(fp16)
    if llm_model is None:
        try:
            llm_model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_WEIGHTS_FOLDER,
                trust_remote_code=True,
                device_map=_device_to_map_value(device),
                torch_dtype=torch.float16,
            )
            print("✓ Float16量子化なしでモデル読み込み成功")
        except Exception as e:
            print(f"✗ Float16量子化なしでエラー: {e}")
            raise

    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_WEIGHTS_FOLDER)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if hasattr(llm_model, 'quantization_config') and llm_model.quantization_config is not None:
        print(f"量子化設定: {llm_model.quantization_config}")
    else:
        print("量子化なし")
        print(f"モデルのデータタイプ: {_model_dtype(llm_model)}")

    if torch.cuda.is_available():
        print(f"GPU メモリ使用量: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU メモリ予約量: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    inspect_model_architecture(llm_model)
    return llm_model, tokenizer

def add_steering_layers(llm_model, insertion_layers):
    """
    指定レイヤーの mlp をステアリング対応ラッパに置換。
    ※ Parameter は未設定のため、この段階で .to(dtype=...) はモジュール全体に対してのみ適用。
    """
    print(f"SteeringLayerを追加するレイヤー: {insertion_layers}")
    for layer_idx in insertion_layers:
        try:
            if layer_idx >= len(llm_model.model.layers):
                print(f"警告: レイヤー {layer_idx} は存在しません（最大: {len(llm_model.model.layers)-1}）")
                continue
            original_mlp = llm_model.model.layers[layer_idx].mlp
            wrapper = create_adaptive_steering_layer(original_mlp)
            # モジュール自体をモデルと同じ device/dtype へ。（Parameterは set_steering 時に登録）
            wrapper.to(device=next(llm_model.parameters()).device, dtype=_model_dtype(llm_model))
            llm_model.model.layers[layer_idx].mlp = wrapper
            print(f"✓ レイヤー {layer_idx} に適応的なSteeringLayerを追加")
        except Exception as e:
            print(f"✗ レイヤー {layer_idx} でエラー: {e}")
            continue
    return llm_model

def load_llm_model_with_insertions(device, insertion_layers):
    print(f"デバイス: {device}")
    print(f"挿入レイヤー: {insertion_layers}")
    llm_model, tokenizer = load_llm_model(device)
    llm_model = add_steering_layers(llm_model, insertion_layers)
    print("モデル読み込みとSteeringLayer追加が完了しました")
    return llm_model, tokenizer

# ----（任意）hidden次元用のステアベクトル設定ヘルパ ----
def set_hidden_steering_from_diff(llm_model, insertion_layers, diff_np, lam):
    """
    diff_np: shape [hidden * len(insertion_layers)] の 1次元配列（numpy）
    各レイヤーに hidden 要素ずつ順に割り当てて set_steering する。
    """
    hidden = getattr(getattr(llm_model, "config", object()), "hidden_size", None)
    if not isinstance(hidden, int) or hidden <= 0:
        # 埋め込みから推定
        emb = getattr(getattr(llm_model, "model", object()), "embed_tokens", None)
        if emb is not None and hasattr(emb, "weight"):
            hidden = int(emb.weight.shape[1])
    if not isinstance(hidden, int) or hidden <= 0:
        raise RuntimeError("hidden_size を推定できませんでした。")

    expected = hidden * len(insertion_layers)
    if diff_np.size != expected:
        raise ValueError(f"diff size mismatch: {diff_np.size} != {expected} (= hidden {hidden} x layers {len(insertion_layers)})")

    device = next(llm_model.parameters()).device
    dtype = _model_dtype(llm_model)

    # 分割
    for i, layer_idx in enumerate(insertion_layers):
        start = i * hidden
        end = start + hidden
        chunk = diff_np[start:end]
        vec = torch.tensor(chunk, dtype=dtype, device=device)
        wrapper = llm_model.model.layers[layer_idx].mlp  # すでに SteeringWrapper
        if not isinstance(wrapper, _BaseSteeringWrapper):
            raise TypeError(f"layer[{layer_idx}].mlp は SteeringWrapper ではありません")
        wrapper.set_steering(vec, lam)
        print(f"layer {layer_idx}: set_steering(hidden={hidden}) 完了")
