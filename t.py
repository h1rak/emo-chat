import os
import transformers
from utils.steering_layer import SteeringLayer
from dotenv import load_dotenv
import torch

load_dotenv()

def inspect_model_architecture(model, layer_idx=0):
    """モデルのアーキテクチャを詳しく調査"""
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
                
                # 主要な属性をチェック
                important_attrs = ['gate_proj', 'up_proj', 'down_proj', 'o_proj', 'fc1', 'fc2', 'dense']
                existing_attrs = []
                for attr in important_attrs:
                    if hasattr(mlp, attr):
                        existing_attrs.append(attr)
                        print(f"  - {attr}: {type(getattr(mlp, attr)).__name__}")
                
                print(f"既存の重要な属性: {existing_attrs}")
    
    print("=== 調査終了 ===\n")

def create_adaptive_steering_layer(original_mlp):
    """MLPのタイプに応じて適応的なSteeringLayerを作成"""
    mlp_type = type(original_mlp).__name__
    print(f"MLP タイプ: {mlp_type}")
    
    # 異なるMLP構造に対応
    if hasattr(original_mlp, 'gate_proj') and hasattr(original_mlp, 'up_proj') and hasattr(original_mlp, 'down_proj'):
        # LLaMA/Qwen2 スタイルのMLP
        print("LLaMA/Qwen2 スタイルのMLPを検出")
        return AdaptiveLLaMASteeringLayer(original_mlp)
    elif hasattr(original_mlp, 'fc1') and hasattr(original_mlp, 'fc2'):
        # GPT スタイルのMLP
        print("GPT スタイルのMLPを検出")
        return AdaptiveGPTSteeringLayer(original_mlp)
    elif hasattr(original_mlp, 'dense'):
        # BERT スタイルのMLP
        print("BERT スタイルのMLPを検出")
        return AdaptiveBERTSteeringLayer(original_mlp)
    else:
        # 汎用的なアプローチ
        print("汎用的なSteeringLayerを使用")
        return GenericSteeringLayer(original_mlp)

class AdaptiveLLaMASteeringLayer(torch.nn.Module):
    """LLaMA/Qwen2スタイルのMLP用SteeringLayer"""
    def __init__(self, original_mlp):
        super().__init__()
        self.original_mlp = original_mlp
        self.steering_vector = None
        self.b = 0.0
        
    def forward(self, x):
        # 元のMLP処理
        original_output = self.original_mlp(x)
        
        # ステアリング処理を適用
        if self.steering_vector is not None:
            steering_effect = self.b * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return original_output + steering_effect
        else:
            return original_output

class AdaptiveGPTSteeringLayer(torch.nn.Module):
    """GPTスタイルのMLP用SteeringLayer"""
    def __init__(self, original_mlp):
        super().__init__()
        self.original_mlp = original_mlp
        self.steering_vector = None
        self.b = 0.0
        
    def forward(self, x):
        original_output = self.original_mlp(x)
        
        if self.steering_vector is not None:
            steering_effect = self.b * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return original_output + steering_effect
        else:
            return original_output

class AdaptiveBERTSteeringLayer(torch.nn.Module):
    """BERTスタイルのMLP用SteeringLayer"""
    def __init__(self, original_mlp):
        super().__init__()
        self.original_mlp = original_mlp
        self.steering_vector = None
        self.b = 0.0
        
    def forward(self, x):
        original_output = self.original_mlp(x)
        
        if self.steering_vector is not None:
            steering_effect = self.b * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return original_output + steering_effect
        else:
            return original_output

class GenericSteeringLayer(torch.nn.Module):
    """汎用的なSteeringLayer"""
    def __init__(self, original_mlp):
        super().__init__()
        self.original_mlp = original_mlp
        self.steering_vector = None
        self.b = 0.0
        
        # 元のMLPの全ての属性をコピー
        for name, module in original_mlp.named_children():
            setattr(self, name, module)
        
        # その他のパラメータもコピー
        for name, param in original_mlp.named_parameters(recurse=False):
            setattr(self, name, param)
    
    def forward(self, x):
        original_output = self.original_mlp(x)
        
        if self.steering_vector is not None:
            steering_effect = self.b * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return original_output + steering_effect
        else:
            return original_output

def load_llm_model(device):
    MODEL_WEIGHTS_FOLDER = os.getenv("MODEL_WEIGHTS_FOLDER")
    
    print(f"transformers version: {transformers.__version__}")
    print(f"torch version: {torch.__version__}")
    
    # BitsAndBytesConfigの互換性をチェック
    try:
        import bitsandbytes
        print(f"bitsandbytes version: {bitsandbytes.__version__}")
        HAS_BNB = True
    except ImportError:
        print("bitsandbytes not installed")
        HAS_BNB = False
    
    # 段階的にモデル読み込みを試行
    llm_model = None
    
    # 方法1: 4bit量子化（推奨）
    if HAS_BNB:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,  # 計算は float16 で実行
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            llm_model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_WEIGHTS_FOLDER,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✓ 4bit量子化でモデル読み込み成功")
        except Exception as e:
            print(f"✗ 4bit量子化でエラー: {e}")
    
    # 方法2: 8bit量子化
    if llm_model is None and HAS_BNB:
        try:
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            llm_model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_WEIGHTS_FOLDER,
                trust_remote_code=True,
                quantization_config=bnb_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✓ 8bit量子化でモデル読み込み成功")
        except Exception as e:
            print(f"✗ 8bit量子化でエラー: {e}")
    
    # 方法3: Float16量子化なし
    if llm_model is None:
        try:
            llm_model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_WEIGHTS_FOLDER,
                trust_remote_code=True,
                device_map="auto",
                torch_dtype=torch.float16
            )
            print("✓ Float16量子化なしでモデル読み込み成功")
        except Exception as e:
            print(f"✗ Float16量子化なしでエラー: {e}")
    
    # 方法4: デフォルト設定
    if llm_model is None:
        try:
            llm_model = transformers.AutoModelForCausalLM.from_pretrained(
                MODEL_WEIGHTS_FOLDER,
                trust_remote_code=True,
                device_map="auto"
            )
            # デフォルト精度からFloat16に変換（量子化されていない場合のみ）
            if not hasattr(llm_model, 'quantization_config') or llm_model.quantization_config is None:
                llm_model = llm_model.to(torch.float16)
            print("✓ デフォルト設定でモデル読み込み成功")
        except Exception as e:
            print(f"✗ デフォルト設定でもエラー: {e}")
            raise e
    
    # トークナイザーの読み込み
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_WEIGHTS_FOLDER)
    
    # パディングトークンの設定
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 量子化情報の表示
    if hasattr(llm_model, 'quantization_config') and llm_model.quantization_config is not None:
        print(f"量子化設定: {llm_model.quantization_config}")
    else:
        print("量子化なし")
        # モデルの最終的なデータタイプを確認
        model_dtype = next(llm_model.parameters()).dtype
        print(f"モデルのデータタイプ: {model_dtype}")
    
    # GPUメモリ使用量を表示（CUDA利用可能な場合）
    if torch.cuda.is_available():
        print(f"GPU メモリ使用量: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"GPU メモリ予約量: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # モデルアーキテクチャを調査
    inspect_model_architecture(llm_model)
    
    return llm_model, tokenizer

def add_steering_layers(llm_model, insertion_layers):
    """指定されたレイヤーに適応的なSteeringLayerを追加する"""
    print(f"SteeringLayerを追加するレイヤー: {insertion_layers}")
    
    for layer_idx in insertion_layers:
        try:
            # レイヤーの存在確認
            if layer_idx >= len(llm_model.model.layers):
                print(f"警告: レイヤー {layer_idx} は存在しません（最大: {len(llm_model.model.layers)-1}）")
                continue
                
            # 適応的なSteeringLayerでラップ
            original_mlp = llm_model.model.layers[layer_idx].mlp
            llm_model.model.layers[layer_idx].mlp = create_adaptive_steering_layer(original_mlp)
            print(f"✓ レイヤー {layer_idx} に適応的なSteeringLayerを追加")
            
        except Exception as e:
            print(f"✗ レイヤー {layer_idx} でエラー: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return llm_model

def load_llm_model_with_insertions(device, insertion_layers):
    """モデルを読み込み、指定されたレイヤーにSteeringLayerを追加する"""
    print(f"デバイス: {device}")
    print(f"挿入レイヤー: {insertion_layers}")
    
    # モデルとトークナイザーを読み込み
    llm_model, tokenizer = load_llm_model(device)
    
    # SteeringLayerを追加
    llm_model = add_steering_layers(llm_model, insertion_layers)
    
    print("モデル読み込みとSteeringLayer追加が完了しました")
    return llm_model, tokenizer

# 使用例とテスト用コード
if __name__ == "__main__":
    # テスト実行
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    INSERTION_LAYERS = [10, 15, 20]  # 例
    
    try:
        model, tokenizer = load_llm_model_with_insertions(DEVICE, INSERTION_LAYERS)
        print("テスト成功: モデルとトークナイザーが正常に読み込まれました")
        
        # 簡単なテスト
        test_text = "こんにちは"
        inputs = tokenizer(test_text, return_tensors="pt")
        print(f"テスト用トークナイザー出力: {inputs.input_ids.shape}")
        
        # SteeringLayerのテスト
        for layer_idx in INSERTION_LAYERS:
            if layer_idx < len(model.model.layers):
                steering_layer = model.model.layers[layer_idx].mlp
                print(f"レイヤー {layer_idx} のSteering Layer: {type(steering_layer).__name__}")
        
    except Exception as e:
        print(f"テスト失敗: {e}")
        import traceback
        traceback.print_exc()