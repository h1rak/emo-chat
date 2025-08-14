import torch
import torch.nn as nn
import pickle

def _device_of_module(mod: nn.Module) -> torch.device:
    p = next(mod.parameters(), None)
    if p is None or p.device.type == "cpu":
        return torch.device("cpu")
    return torch.device(f"{p.device.type}:{p.device.index}")

def _infer_hidden_size_from_mlp(mlp: nn.Module) -> int:
    """
    さまざまな実装の MLP から hidden_size を推定。
    出力側(=縮小)Linearを優先し、なければ入力側を見る。最後は Linear 総当り。
    """
    out_candidates = [
        "down_proj", "out_proj", "o_proj", "proj", "fc2", "c_proj",
        "dense_4h_to_h", "wo",
    ]
    in_candidates = [
        "up_proj", "gate_proj", "fc1", "c_fc",
        "dense_h_to_4h", "wi", "w1", "w2",
    ]
    for name in out_candidates:
        layer = getattr(mlp, name, None)
        if isinstance(layer, nn.Linear):
            return layer.out_features
    for name in in_candidates:
        layer = getattr(mlp, name, None)
        if isinstance(layer, nn.Linear):
            return layer.in_features
    for m in mlp.modules():
        if isinstance(m, nn.Linear):
            return getattr(m, "out_features", None) or getattr(m, "in_features")
    raise AttributeError("SteeringLayer: hidden_size を推定できませんでした。")

class SteeringLayer(nn.Module):
    """
    既存の MLP (= layer_of_interest) の出力に、学習可能なステアリングベクトルを加えるラッパー。
    - y = a * base(x) + b * v  (add_steering=True, ignore_activations=False)
    - y = b * v                (add_steering=True, ignore_activations=True)
    - y = base(x) + b*(v - base(x)) = (1-b)*base(x) + b*v (shift_with_new_idea=True)
    """
    def __init__(self, layer_of_interest: nn.Module, hidden_size: int | None = None):
        super().__init__()
        self.layer_of_interest = layer_of_interest
        self.device = _device_of_module(self.layer_of_interest)

        if hidden_size is None:
            hidden_size = _infer_hidden_size_from_mlp(layer_of_interest)

        self.hidden_size = int(hidden_size)

        # v は (H,) で持ち、使用時に (1,1,H) へ拡張して (B,T,H) にブロードキャスト
        v = torch.empty(self.hidden_size, device=self.device)
        nn.init.xavier_normal_(v.unsqueeze(0))  # seed-friendly
        self.steering_vector = nn.Parameter(v)  # shape: (H,)

        # 振る舞いフラグ & 係数
        self.add_steering = True
        self.ignore_activations = False
        self.shift_with_new_idea = False
        self.a = 1.0
        self.b = 1.0

    def _steering_for(self, y: torch.Tensor) -> torch.Tensor:
        """
        出力テンソル y と同じ dtype/device に合わせた steering ベクトルを返す。
        y: (B, T, H) or (N, H) など。最後の次元 H に合わせてブロードキャストする。
        """
        v = self.steering_vector
        if v.device != y.device:
            v = v.to(y.device)
        if v.dtype != y.dtype:
            v = v.to(dtype=y.dtype)
        # (H,) -> (1,1,H) にして (B,T,H) へブロードキャスト
        while v.ndim < y.ndim:
            v = v.unsqueeze(0)
        # 最後の次元が合っているか確認
        if v.shape[-1] != y.shape[-1]:
            raise RuntimeError(f"Steering vector hidden size mismatch: v={v.shape}, y={y.shape}")
        return v

    def reset_steering_vector(self):
        self.device = _device_of_module(self.layer_of_interest)
        v = torch.empty(self.hidden_size, device=self.device)
        nn.init.xavier_normal_(v.unsqueeze(0))
        self.steering_vector = nn.Parameter(v)

    def load_steering_vector(self, steering_vector_path: str, key: str | None = None):
        """
        pickle から v を読み込む。
        - sd が {str: Tensor} の想定。key 未指定なら最初の Tensor を使う。
        """
        self.device = _device_of_module(self.layer_of_interest)
        with open(steering_vector_path, "rb") as f:
            sd = pickle.load(f)

        tensor = None
        if isinstance(sd, dict):
            if key is None:
                # 既存コードのキー推定ロジックを一応踏襲
                guessed = steering_vector_path.split("/")[-1].split(".")[0].split("_")[-1]
                tensor = sd.get(guessed, None)
                if tensor is None:
                    # 先頭の Tensor を拾う
                    for k, v in sd.items():
                        if isinstance(v, torch.Tensor):
                            tensor = v
                            break
            else:
                tensor = sd.get(key, None)

        if not isinstance(tensor, torch.Tensor):
            raise KeyError(f"Could not find a Tensor in {steering_vector_path}. Provide a valid 'key'.")

        if tensor.shape[-1] != self.hidden_size:
            raise RuntimeError(f"Loaded steering vector dim mismatch: {tensor.shape} vs hidden_size={self.hidden_size}")

        self.steering_vector = nn.Parameter(tensor.to(self.device))

    def forward(self, *x, **kw):
        # base 出力は一度だけ計算
        y = self.layer_of_interest(*x, **kw)

        if self.shift_with_new_idea:
            v = self._steering_for(y)
            return y + self.b * (v - y)  # = (1 - b) * y + b * v

        if self.add_steering:
            v = self._steering_for(y)
            if self.ignore_activations:
                return y * 0.0 + self.b * v
            return self.a * y + self.b * v

        return y
