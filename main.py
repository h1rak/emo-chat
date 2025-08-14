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

def transrate(prompt,model,tokenizer):
    messages = [
        {"role": "system", "content":"あなたは優秀な翻訳者です。与えられた文章を日本語のニュアンスを残したまま、英語に翻訳してください。翻訳だけをすればいいです。"},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False # 思考モードと非思考モードを切り替え、デフォルトは True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # テキスト補完の実行
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # 思考コンテキストのパース
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    #thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
    print("content:", content)
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
        {"role":"system","content":"あなたは25歳の男性として振舞ってください。返答はできるだけ短文で日本語で生成してください。"},
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

def interactive_eval_autocast_multi_gpu(
    emotions, pred_emo, lam, history, strategy, prompt,
    means, ovr_r_means, total_mean, llm_model, tokenizer,
    insertion_layers, save_path, device,
    target="hidden",           # "hidden"（既定） or "mlp"
    mlp_out_sizes=None         # target="mlp" のときのみ使用（例: {15: 11008, 16: 11008, 17: 11008}）
):
    # ========== 入力整形 ==========
    messages = create_strategy_messages(history, strategy, prompt)
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    #inputs = {k: v.to(device) for k, v in inputs.items()}


    # ========== ベクトル差分 ==========
    emo_idx = emotions.index(pred_emo)
    
    vec_split = np.split(means[emo_idx] - ovr_r_means[emo_idx], len(insertion_layers))
    for n, _ in enumerate(insertion_layers):
        llm_model.model.layers[insertion_layers[n]].mlp.steering_vector = nn.Parameter(torch.from_numpy(vec_split[n]).to(device))
        llm_model.model.layers[insertion_layers[n]].mlp.b = lam

    with torch.amp.autocast('cuda', dtype=torch.float16):
        gen_tokens = llm_model.generate(
            **inputs,                 # input_ids / attention_mask をそのまま渡す
            max_new_tokens=300,       # max_length ではなく「新規トークン数」で指定
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    # --- 生成部分だけ取り出す ---
    input_len = inputs.input_ids.shape[1]
    new_tokens = gen_tokens[0, input_len:]                 # 入力ぶんをスキップ
    gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    print("生成だけ：", gen_text)
    return gen_text



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
    
    print("Loading translation model...")
    # 翻訳モデルの設定
    #translate_model_id = "/workspace/santa/emo-chat/model/openai"
    #translate_tokenizer = AutoTokenizer.from_pretrained(translate_model_id)

    # メモリ効率的な読み込み
    #translate_model = None
    
    target_dtype = torch.bfloat16   # ← ここを torch.float16 にすれば fp16 統一にできます
    main_model.to(device=DEVICE_MAIN)          # 念のため device を先に合わせる
    #translate_model.to(device=DEVICE_TRANSLATE)

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
    #if translate_model == main_model:
    #    translate_tokenizer = main_tokenizer
    
    # チャットループ
    utt1 = ""
    emo = ["joy"]
    personality = [0.6, 0.48, 0.31, 0.46, 0.56]
    mood = [1.0, 1.0 , 0.0]
    history = []
    
    while True:
        prompt = input("入力：")
        print("prompt:",prompt)
        history.append(f"user:{prompt}")
            
        # 翻訳（指定されたデバイスで実行）
        content = transrate(prompt, main_model, main_tokenizer)
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
        utt1 = transrate(utt1, main_model, main_tokenizer)

if __name__ == "__main__":
    main()