# 解题说明 · Continuous Batching 调度器

## 与静态 batching 的对比

传统 static batching 要求 batch 内所有请求同时结束——短请求要等长请求，slot 空转。Continuous batching 允许请求**随时加入/退出** running 集，把 GPU 利用率提到接近饱和。

## prefill 简化

真实系统 prefill 是「一步跑完 prompt_len 个 token」；本题把它抽象成 1 个额外 step（相当于把 prefill 视作 1 个特殊 decode token）。这不影响调度语义、只是让总步数好算。

## admit 时机

每 step **先 admit 再消耗**。所以一个请求 `arrival=t, gen_len=2`：t 步 admit + 消耗 → rem=2；t+1 rem=1；t+2 rem=0 finish。占用 3 个 step（1 prefill + 2 decode）。

## FCFS 排序

按 `(arrival_step, request_id)` 升序，保证「同 arrival 时按输入顺序」。这也是 vLLM 早期版本的默认策略。
