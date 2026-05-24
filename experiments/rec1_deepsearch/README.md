# 1 【大模型LLM学习】Agentic RL—基于Qwen3-4b的Deep Search Agent

- 流程详见https://zhuanlan.zhihu.com/p/2034621850806957514
- 环境安装需要安装ms-swift 4.0以上版本，从源代码安装，记住源代码的位置，之后需要修改源代码

# 2 具体步骤
## 2.1 工具说明
- 租了一个阿里云的服务器用于部署search服务，运行search_server.py部署websearch_http.py和webvisit_http.py

## 2.2 数据准备
- 如果需要自己收集QA seed数据，可以获取维基百科的中英文版本的历史上的今天，每天里面的每个条目作为seed ground truth，然后用Qwen-Max根据seed ground truth造原始的seed QA，接着有了seed QA后，调用百炼API，启动websearchh的gent或者max模式，让distill_websearch_qa_generation_agent_full_en.py对问题进行复杂化，得到multi-hop QA
- 其他数据可以在huggingface搜索“webshaper”，webshaper有500条高质量数据
- 天池的deep search比赛的第一轮提供了GAIA的40条纯web搜索能解决的问题，这个作为测试集，在/eval/下“gaia.jsonl”
- 天池的deep search比赛的数据和答案在在/eval/下“tianchi_data_stage_1.jsonl”和“tianchi_data_stage_2.jsonl”

## 2.3 获取teacher轨迹
- 基于Tongyi DeepResearch的resum代码，调用main.py，设置好模型和数据集，把要蒸馏的数据集放到/eval_data下，调用Qwen3-Max
- 获取完轨迹后，检查轨迹是否符合格式要求，回答是否准确

## 2.4 离线SFT
- 参考/script/final_lora.sh训练模型

## 2.5 在线RL
- 参考https://zhuanlan.zhihu.com/p/2034621850806957514
- 需要修改multi_turn.py和orm.py，直接复制有可能因为ms-swift版本问题导致报错，直接修改swift对应源代码会好一点
- 修改完之后参考/script/rollout.sh和/script/grpo.sh

## 2.6 本地评测
- 测试统计accuracy（pass@K）

