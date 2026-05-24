# 1 【大模型LLM学习】Agentic RL—基于Qwen3-4b训练Travel Planning Agent
- 详见https://zhuanlan.zhihu.com/p/2041642745836692804
- 环境安装ms-swift 4.0以上版本，从源代码安装，记住源代码位置，后续需要修改源代码

# 2 使用流程
## 2.0 工具准备
- 使用高德的地图API，阿里IQS搜索API

## 2.1 数据准备
- 使用高德的ArenaRL提供的数据集，训练集部分先分类，筛选出属于旅游行程规划的部分，剔除掉涉及海外的部分（高德API不支持）
- 使用Tongyi DeepResearch的resum框架，收集教师轨迹数据，修改main.py对应的数据和模型部分跑起来
- 得到轨迹数据后，过滤掉格式错误、工具错误等

## 2.2 离线SFT
- 参考/script/final_lora.sh训练模型

## 2.3 在线RL
- 参考https://zhuanlan.zhihu.com/p/2041642745836692804
- 需要修改multi_turn.py和orm.py，直接复制有可能因为ms-swift版本问题导致报错，直接修改swift对应源代码会好一点
- 修改完之后参考/script/rollout.sh和/script/grpo.sh

## 2.4 本地评测
- 测试统计win rate