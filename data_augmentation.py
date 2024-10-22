import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
import json
import gc
import random
from vllm import LLM, SamplingParams

# モデルとトークナイザーの初期化
model_name = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
tokenizer = AutoTokenizer.from_pretrained(model_name)
llm = LLM(
    model=model_name,
    tensor_parallel_size=torch.cuda.device_count(), 
    gpu_memory_utilization=0.95,  # GPUメモリ使用率を調整
    max_model_len=8192,  # モデルの最大長を調整
    dtype=torch.float16
)

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=1024
)

BATCH_SIZE = 50  # バッチサイズを小さくする（大きなモデルのため）
SAVE_INTERVAL = 200  # 保存間隔を調整

# システムプロンプトのリスト
SYSTEM_PROMPTS = [
    "あなたは好奇心旺盛で知識欲の高いアシスタントです。どんな質問にも詳細に答え、新しい視点を提供することを心がけてください。",
    "あなたは論理的で分析力に優れたアシスタントです。問題を細分化し、段階的に解決策を提示することを得意としています。",
    "あなたは創造性豊かで斬新なアイデアを生み出すアシスタントです。常識にとらわれない発想で問題解決に取り組んでください。",
    "あなたは温かみのある共感力の高いアシスタントです。相手の気持ちを理解し、寄り添うような問題解決と返答を心がけてください。",
    "あなたは効率と生産性を重視するアシスタントです。無駄を省いた簡潔で実用的な解決策を提供することを目指してください。",
    "あなたは歴史や文化に精通したアシスタントです。様々な時代や地域の知識を活かし、多角的な視点から回答してください。",
    "あなたは冷静沈着で公平なアシスタントです。感情に左右されず、客観的な事実に基づいた回答を提供してください。",
    "あなたは楽観的でユーモアのセンスがあるアシスタントです。難しい問題でも前向きに捉え、時には冗談を交えて問題回答してください。",
    "あなたは細部にこだわる完璧主義者のアシスタントです。丁寧で正確な情報提供を心がけ、些細な点も見逃さないようにしてください。",
    "あなたは柔軟性が高く適応力のあるアシスタントです。状況に応じて異なるアプローチを取り、多様な解決策を提示してください。"
]

def load_progress(filename="progress.json"):
    """進捗をロード"""
    if os.path.exists(filename):
        with open(filename, "r") as f:
            return json.load(f)
    return {"processed_count": 0}

def save_progress(progress, filename="progress.json"):
    """進捗を保存（トランザクション的）"""
    temp_filename = filename + ".tmp"
    with open(temp_filename, "w") as f:
        json.dump(progress, f)
    os.rename(temp_filename, filename)

def append_outputs(data, filename="generated_sets.jsonl"):
    """生成されたデータを保存（アペンドモード）"""
    with open(filename, "a", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def clean_instruction(instruction):
    """指示文の冒頭をクリーンアップする関数"""
    # 指示文の冒頭30文字以内に "指示文" があるか確認
    if "指示文" in instruction[:30]:
        # "指示文" 以降20文字以内に "\n\n" があるか確認
        split_index = instruction[instruction.index("指示文"):].find("\n\n")
        if split_index != -1 and split_index <= 20:
            return instruction[instruction.index("指示文") + split_index + 2:].strip()
    return instruction.strip()  # 条件に合わない場合はそのまま返す

def load_instruction_samples(filename="instruction_samaples.json"):
    with open(filename, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return [item["japanese"] for item in samples]

# サンプル指示文の読み込み
INSTRUCTION_SAMPLES = load_instruction_samples()

def generate_texts(system_prompt, prompts):
    """バッチ処理でテキストを生成（同期）"""
    messages_batch = [
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        for prompt in prompts
    ]
    texts = [tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) for messages in messages_batch]
    outputs = llm.generate(texts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]

def generate_pipeline(batch_size):
    """指示文→推論→回答→回答改良のパイプラインをバッチ処理で実行"""

    # 1. システムプロンプトをランダムに選択
    system_prompt = random.choice(SYSTEM_PROMPTS)

    # 指示文の生成（サンプルを参考に）
    reference_samples = random.sample(INSTRUCTION_SAMPLES, min(4, len(INSTRUCTION_SAMPLES)))
    instruction_prompts = [
        f"以下は指示文の例です：" + "\n\n".join(reference_samples) + "\n\n上記の例を参考に、新しい指示文を1つだけ作成してください。ただし、上記の例とまったく同じ内容にならないようにしてください。\n作成した指示文以外は一切書かないことに注意してください．\n\n指示文："
        for _ in range(batch_size)
    ]
    
    instructions = generate_texts(system_prompt, instruction_prompts)
    
    # ここで clean_instruction を適用
    cleaned_instructions = [clean_instruction(instruction) for instruction in instructions]

    # 以降の処理では cleaned_instructions を使用する
    reasoning_prompts = [
        f"指示: {instruction}\n\nこの指示を達成するための論理的な推論手順を段階的に簡潔に説明してください。\n説明の内容以外は一切書かないでください。"
        for instruction in cleaned_instructions
    ]
    reasonings = generate_texts(system_prompt, reasoning_prompts)
    
    # 4. 最初の回答の生成
    answer_prompts = [
        f"指示: {instruction}\n\n推論手順: {reasoning}\n\n推論手順に基づき、指示に対して簡潔に回答してください。\n回答の内容以外は一切書かないでください。"
        for instruction, reasoning in zip(instructions, reasonings)
    ]
    answers = generate_texts(system_prompt, answer_prompts)

    # 5. 回答の改良 (Self-Refine 機構)
    refine_prompts = [
        f"指示: {instruction}\n\n元の回答: {answer}\n\nこの回答の品質を高めるために、誤解を避けつつ、さらに簡潔で洗練された回答に改良してください。\n改良後の回答の内容以外は一切書かないでください。"
        for instruction, answer in zip(instructions, answers)
    ]
    refined_answers = generate_texts(system_prompt, refine_prompts)

    # 6. 生成結果をまとめる
    generated_sets = [
        {
            "instruction": instruction,
            "reasoning": reasoning,
            "initial_answer": answer,
            "refined_answer": refined_answer
        }
        for instruction, reasoning, answer, refined_answer in zip(instructions, reasonings, answers, refined_answers)
    ]

    return generated_sets

def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()

def main():
    # 進捗の読み込み
    progress = load_progress()
    processed_count = progress["processed_count"]

    total_generations = 100000

    # バッファの初期化
    buffer = []

    try:
        while processed_count < total_generations:
            remaining = total_generations - processed_count
            batch_size = min(BATCH_SIZE, remaining)

            # 1バッチのデータを生成
            generated_sets = generate_pipeline(batch_size=batch_size)

            # バッファに追加
            buffer.extend(generated_sets)
            processed_count += len(generated_sets)

            # 進捗表示
            print(f"処理済み: {processed_count}/{total_generations}", end='\r')

            # バッファが十分に溜まったら保存
            if len(buffer) >= SAVE_INTERVAL:
                append_outputs(buffer)
                buffer = []  # バッファをクリア
                print(f"\n{processed_count}件処理しました。")
                save_progress({"processed_count": processed_count})

                # メモリをクリア
                clear_memory()

    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        # 保存処理を行う（必要に応じて）
        if buffer:
            append_outputs(buffer)
            buffer = []
        save_progress({"processed_count": processed_count})
        # メモリをクリア
        clear_memory()
        raise  # 再度例外を投げて終了

    finally:
        # 最終的なバッファの保存
        if buffer:
            append_outputs(buffer)
            buffer = []

        print(f"\nすべての{total_generations}件の生成が完了しました。")
        # 最終的な進捗保存
        save_progress({"processed_count": processed_count})

        # メモリをクリア
        clear_memory()

if __name__ == "__main__":
    main()