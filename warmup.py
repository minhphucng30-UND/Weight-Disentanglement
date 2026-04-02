from vllm import LLM, SamplingParams
from verl.utils.reward_score.deepscaler import rllm_reward_fn_math
import pandas as pd

if __name__ == "__main__":
    df = pd.read_parquet("data/polaris/Qwen2.5-3B-Instruct_buffer_512.parquet")
    prompts = df["prompt"].tolist()
    ground_truths = df["reward_model"].tolist()
    gts = [ele['ground_truth'] for ele in ground_truths]
    responses = df["response"].tolist()

    correct_prompts = []
    correct_responses = []
    print(len(prompts), len(responses), len(ground_truths))

    for prompt, response, gt in zip(prompts, responses, gts):
        if rllm_reward_fn_math("", response, gt) == 1:
            correct_prompts.append(prompt)
            correct_responses.append(response)
    
    print(len(correct_prompts), len(correct_responses))

    df = pd.DataFrame({"prompt": correct_prompts, "response": correct_responses})
    df = df.sample(frac=1).reset_index(drop=True)
    df.to_parquet("data/polaris/Qwen2.5-3B-Instruct_buffer_512_correct.parquet")

    # llm = LLM(model="models/Qwen2.5-3B-Instruct", enable_prefix_caching=True, gpu_memory_utilization=0.85)
    # tokenizer = llm.get_tokenizer()
    # sampling_params = SamplingParams(temperature=1.0, max_tokens=3072, n=16)
    
    # df = pd.read_parquet("data/polaris/train.parquet")
    # df = df.sample(n=512)
    # prompts = df["prompt"].tolist()
    # ground_truths = df["reward_model"].tolist()


    # chat_prompts = [tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts]
    # outputs = llm.generate(chat_prompts, sampling_params)
    # all_prompts = []
    # all_responses = []
    # all_gts = []

    # for prompt, output, gt in zip(prompts, outputs, ground_truths):
    #     for i in range(len(output.outputs)):
    #         all_prompts.append(prompt)
    #         all_responses.append(output.outputs[i].text)
    #         all_gts.append(gt)

    # df = pd.DataFrame({"prompt": all_prompts, "response": all_responses, "reward_model": all_gts})
    # df = df.sample(frac=1).reset_index(drop=True)
    # df.to_parquet("data/polaris/Qwen2.5-3B-Instruct_buffer_512.parquet")