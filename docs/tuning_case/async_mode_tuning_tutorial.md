# 异步 RL 训练调优指南

本文面向使用 RL Insight 分析异步 RL 训练任务的用户，介绍如何识别训练（Trainer）与推理采样（Rollouter）之间的负载情况，并针对 `separate async`、`fully async` 和 `colocate async` 三种范式选择调优策略。

值得一提的是在 [2026 Q4 的 Roadmap](https://github.com/verl-project/verl/issues/8186#:~:text=Deprecate%20verl/experimental/fully_async_policy%7Cone_step_off_policy%20to%20verl%2Drecipe) 中已明确表示，`fully async`将会在主分支中被弃用，转至 `verl-recipe`。

文中的结论来自 Qwen3-30B、Qwen3.5-35B 和 Qwen3-Omni 的实测案例。具体收益会随模型、序列长度、硬件、并行策略和代码版本变化，应以同一任务的对照实验为准。

## 1. 调优目标

异步训练的目标是让 Trainer 或 Rollouter 互相掩盖，在不影响训练质量的前提下尽可能提高端到端吞吐。
在多次异步调优的过程中，我认为应该重点关注以下三类问题：

1. **Rollouter 空泡**：一般来自于 Rollouter 提前完成所有可生成的样本，等待 Trainer 完成 update actor 后收到新的 prompt 才能继续采样。
2. **Trainer 空泡**：Trainer 等待足够样本产生。
3. **有效性损失**：增大 Off-policy 的程度后，是否可能引起 reward 训练质量波动或训练偏差。

建议同时观察以下指标：

| 维度   | 重点信号                                                                | 说明                                                 |
| ---- | ------------------------------------------------------------------- | -------------------------------------------------- |
| 时间线  | Trainer/Rollouter 活跃区间、parameter version、validation 区间              | 判断空泡出现在哪一侧以及是否周期性发生                                |
| 采样负载 | 每个 replica 的请求数、`running request`、scheduler state、cache utilization | 判断请求是否均匀、是否存在部分 rank 空闲                            |
| 训练负载 | mini-batch 执行时间、训练显存、`timing_s/gen`                                 | `fully async` 中，`timing_s/gen` 可反映 Trainer 等待样本的时间 |
| 数据分布 | 输入/输出序列长度分布                                                         | 长尾序列会造成 replica 和 rank 间负载不均                       |
| 训练质量 | reward、accuracy、staleness、RS/IS 相关指标                                | 确认吞吐优化没有造成不可接受的质量退化                                |

## 2. 三种异步范式及主要调节旋钮

| 范式               | 资源布局                                                                            | 主要矛盾                   | 优先调节项                                                                            |
| ---------------- | ------------------------------------------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------- |
| `separate async` | 训推异卡，Rollouter 有自己独立的资源池，Trainer 采用 Hybrid Engine 的模式，相比 Fully Async 有了更复杂的转换逻辑 | 训练和推理消费生成样本的速度不匹配，     | Trainer/Rollouter 资源比、`use_trainer_do_validate`、`warm_up_batch`、PPO token budget |
| `fully async`    | 训推异卡，Rollouter 有自己独立的资源池，Trainer 采用 Hybrid Engine 的模式，可以在 val 阶段加入推理            | 空泡与 off-policy 程度之间的权衡 | staleness、Rollouter 规模、parameter sync/trigger interval、RS/IS                     |
| `colocate async` | 训推共卡，串行执行                                                                       | --                     | `warm_up_batch`                                                                  |

> \[!NOTE]
> 参数名称可能随分支或版本变化。本文沿用实验环境中的名称，例如 `parameter_sync_step`、`warm_up_batch` 和 `use_trainer_do_validate`。

## 3. 推荐调优流程

### 3.1 建立基线

固定模型、数据、总卡数和训练超参数，至少记录：

- 端到端吞吐；
- Trainer 与 Rollouter 的设备数；
- train batch、PPO mini-batch、PPO max token；
- `parameter_sync_step`、`warm_up_batch`、staleness；
- validation 周期；
- reward 和 accuracy 曲线。

每轮实验只改变一类变量，并避开初始化和 warm-up 区间比较稳定态吞吐。

### 3.2 根据时间线定位瓶颈

| 观察结果                           | 可能原因                           | 优先动作                                                          |
| ------------------------------ | ------------------------------ | ------------------------------------------------------------- |
| Rollouter 长时间空闲，Trainer 持续更新   | 推理资源过量，或 staleness 上限过小        | `separate async` 中减少 Rollouter；`fully async` 中先评估增大 staleness |
| Trainer 周期性等待样本                | 采样不足，或参数同步周期内可消费样本不足           | 增加有效采样能力、提高 PPO token budget，或调整 staleness                    |
| validation 时 Trainer 空闲        | Trainer 未参与 validation rollout | 评估开启 `use_trainer_do_validate`                                |
| 部分 replica/rank 空闲             | 长度分布不均、请求数不能均分或调度长尾            | 检查序列长度、replica 数量和请求分配                                        |
| Rollouter 很快耗尽 staleness 额度后空闲 | Rollouter 过多，或 staleness 过小    | 不要继续盲目增加 Rollouter；比较减资源和增 staleness                          |
| reward 波动随 staleness 增大        | off-policy 程度上升                | 回退 staleness，或重新选择 RS/IS 策略                                   |

### 3.3 按成本从低到高调参

1. 先消除可解释的固定空泡，例如 validation 切换或不合理的资源整除关系。
2. 再平衡 Trainer/Rollouter 资源，避免一侧长期领先。
3. 调整 batch、PPO token budget 和 `warm_up_batch`，提高单阶段有效工作量。
4. 最后调整 staleness、RS/IS 等可能影响训练质量的参数。
5. 每次修改后同时检查吞吐、时间线和质量指标，不能只看单一利用率。

## 4. Separate Async 调优

### 4.1 理解 Hybrid Engine 与 standalone replica

在的 `separate async` 初始化流程中：

1. 先执行基类 `_setup()`，创建 Hybrid Engine；
2. 再通过 `init_standalone()` 创建独立 resource pool，并启动新的 rollout server；
3. standalone replica 的 rank 从 Hybrid Engine replica 之后开始编号，以避免 Ray named actor 冲突。

因此，监控中看到的 replica 总数可能大于独立 Rollouter 数量。下例中共有 32 个 replica，其中 `0-7` 是 Trainer/Rollouter 共置的 Hybrid Engine。

![Separate async replica 初始化与编号](图片和附件/image%2021.png)

![Hybrid Engine 对应的 replica 0-7](图片和附件/image%2011.png)

调优时应先区分 Hybrid Engine 与 standalone Rollouter，不能直接用 replica 总数推导独立推理资源。

### 4.2 利用 Trainer 参与 validation

实验版本中，Hybrid Engine 会在 `on_validate_begin()` 切换到 rollout 模式并参与 validation，即原 `use_trainer_do_validate` 行为。validation 结束后，它会保持推理模式，直到下一次 replay buffer sample 完成并触发 `on_sample_end()`，再切回 Trainer。

因此，在 validation 附近观察到 Hybrid Engine 执行 generate 是预期行为：

![Hybrid Engine 在 validation 阶段参与 generate](图片和附件/image%209.png)

若 validation 周期固定且 Trainer 在此期间明显空闲，可启用该能力消除周期性空泡。评估时需要计入切换开销，并确认不会影响下一轮训练启动。

### 4.3 平衡 Trainer 与 Rollouter 资源

#### 案例：Qwen3-30B

基线配置：

| 参数            | 配置                    |
| ------------- | --------------------- |
| Train nodes   | 16                    |
| Rollout nodes | 48                    |
| Validation 周期 | 每 5 step              |
| `gen_tp`      | 2，即每个 replica 使用 2 张卡 |

基线时间线显示 Rollouter 存在大量空泡：

![Qwen3-30B 基线时间线](图片和附件/image%208.png)

序列长度分布如下，分析资源利用率时需要同时考虑长度差异造成的负载不均：

![Qwen3-30B 序列长度分布](图片和附件/img_v3_0213l_0df55db5-647a-4fe0-ac70-39ee23faef9g.jpg)

将 Rollout nodes 从 48 减少到 32 后，推理阶段空泡减少，端到端吞吐提升约 **25%**：

![Rollout nodes 从 48 减少到 32 后的吞吐结果](图片和附件/image%2024.png)

这个案例说明：当 Rollouter 长期领先 Trainer 时，增加推理资源不一定提高吞吐。减少过量 Rollouter 既能降低空泡，也能改善单位资源产出。

### 4.4 调整训练侧工作量

#### 案例：Qwen3-Omni

初始配置：

| 参数                    | 配置  |
| --------------------- | --- |
| Train batch           | 256 |
| PPO mini-batch        | 32  |
| `parameter_sync_step` | 8   |
| `warm_up_batch`       | 1   |
| PPO max token         | 10k |

在 `r8t8` 配置下，第一个 replay buffer sample 完成前，Hybrid Engine 也可以参与 rollout。但整体上推理明显快于训练，Rollouter 约领先 Trainer 5 个 PPO mini-batch，形成大量空泡。

![r8t8 时间线](图片和附件/image%203.png)

![r8t8 负载细节](图片和附件/image%2022.png)

![r8t8 空泡细节](图片和附件/image%2016.png)

`r4t12` 实验中，开启 switch 后每个 replica 处理的请求反而减少：

![r4t12 开启 switch 后的 replica 请求数](图片和附件/image%2018.png)

该现象需要结合请求总数和 replica 间的整除关系判断。为满足均匀分配约束，实验中增加了 Trainer 资源：

![增加 Trainer 资源后的分配结果](图片和附件/image%202.png)

作为对照，以下是 `r8` 且关闭 switch 的结果：

![r8 且关闭 switch 的结果](图片和附件/image%2014.png)

在 `r8t16` 下，将 `warm_up_batch` 调整为 2，其他核心参数保持不变，空泡从约 5 个缩小到约 3 个 PPO mini-batch，但仍有优化空间：

![r8t16、warm\_up\_batch=2 的时间线](图片和附件/image%206.png)

训练显存仍有余量：

![r8t16 训练显存](图片和附件/image%2025.png)

随后将 PPO max token 从 10k 提高到 30k，增加训练侧单轮工作量，推理每轮空泡进一步缩小到约 1.5 个 mini-batch：

![PPO max token 提高到 30k 后的时间线](图片和附件/image%2013.png)

训练显存利用率随之提高，但该区间表现出一定不稳定性，需要继续观察 OOM、step time 抖动和训练质量：

![PPO max token 为 30k 时的训练显存](图片和附件/image%2019.png)

不同配置的整体吞吐对比如下：

![Separate async 不同配置的吞吐对比](图片和附件/image)

本案例的调优顺序是：先调整训推资源比，再利用显存余量提高 PPO token budget。提高 token budget 前，应确认显存余量并保留安全边界。

## 5. Fully Async 调优

### 5.1 用 staleness 覆盖流水线空泡

staleness 允许 Rollouter 使用较旧的 parameter version 继续生成样本，从而提高训推重叠度。它能减少等待，但也会提高 off-policy 程度，因此必须同时观察吞吐和训练质量。

#### 案例：Qwen3.5-35B，staleness = 0.3

下图中黄线表示 parameter version 的边界：

![staleness=0.3 的 parameter version 时间线](图片和附件/image%2027.png)

![staleness=0.3 的负载概览](图片和附件/img_v3_0213m_4a24fab5-e4b3-4c0c-9b77-a8b644ba134g.jpg)

时间线暴露出两类空泡：

1. **Rollouter 侧**：完成约 `1 + staleness` 对应的样本额度后，仍需等待 Trainer 完成 update。
2. **Trainer 侧**：
   - 每 5 step 的 validation 使 Trainer 固定进入空闲，可通过 `use_trainer_do_validate` 让 Trainer 切换为 Rollouter 并参与 validation；
   - 当前 trigger step 为 8。按实验中的估算，`0.3 / (1/8) > 2`，额外样本可覆盖前两个 trigger step；后续 Trainer 空泡主要来自等待 Rollouter 生成样本。

![staleness=0.3 的空泡拆解](图片和附件/image%2020.png)

此时有两种候选策略：

- **增加 Rollouter**：可能缩短 Trainer 等待样本的时间，但 Rollouter 也可能更快耗尽 staleness 额度并在后半段空闲。若 Rollouter 已占主要资源，这可能降低总体资源效率。
- **增大 staleness**：利用后半段空闲的 Rollouter 继续生成样本，通常更有利于覆盖空泡，但需要承担更高的训练质量风险。

### 5.2 对比不同 staleness

**staleness = 0.3**

![staleness=0.3 的负载](图片和附件/image%2010.png)

**staleness = 0.5**

![staleness=0.5 的负载](图片和附件/img_v3_0213r_2d57c6bf-cea7-4670-b2e7-0a77ca2f088g.jpg)

**staleness = 0.7**

![staleness=0.7 的负载](图片和附件/image%2015.png)

实验结果：

1. 增大 staleness 后，从 step 2 开始，Trainer 与 Rollouter 之间的空泡覆盖明显改善。
2. 在本组 `fully async` 实验中，`timing_s/gen` 表示 Trainer 等待 Rollouter 生成样本的时间。当 `staleness >= 0.5` 时，不同配置的该指标基本在相同区间内波动，继续增大 staleness 的吞吐边际收益有限。
3. staleness 增大后，reward 波动幅度明显增加，但 accuracy 仍保持相对平稳的上升趋势。
4. 不同 RS + IS 策略对 reward 上升过程有较大影响，不能只根据系统吞吐选择 staleness。

![不同 staleness 下的 timing\_s/gen](图片和附件/image%2023.png)

![不同 staleness 下的 reward 对比](图片和附件/image%2012.png)

![不同 staleness 下的 accuracy 对比](图片和附件/image%204.png)

![不同 RS + IS 策略的结果对比](图片和附件/image%201.png)

对该案例，`0.5` 是更合理的候选起点：它已经基本消除主要等待，且相较 `0.7` 保留了更低的 off-policy 风险。该值不是通用最优值，仍需在目标任务上验证。

## 6. Colocate Async 调优

`colocate async` 通过同一批设备在训练和推理阶段间切换，避免静态划分造成的长期资源失衡。相较 `separate async`，可调的资源比例更少，调优重点转向：

- 选择合适的 `warm_up_batch`；
- 使用 PPO token budget 平衡训练阶段与推理阶段的耗时；
- 检查长尾请求、abort 和部分 rank 空闲；
- 评估阶段切换开销。

> \[!IMPORTANT]
> 原实验进行时，GitHub 主分支尚未合入 `colocate async` 支持，实验基于 `npu_0917_qjh` 分支。当前使用前应以目标版本的实际实现和配置为准。

### 6.1 `warm_up_batch = 2`，PPO max token 为基线值

该配置的时间线如下：

![warm\_up\_batch=2 的 colocate async 时间线](图片和附件/image%2017.png)

从 cache utilization 和 scheduler state 看，每一步仍有部分 rank 出现 `running request = 0`：

![colocate async 的 cache utilization](图片和附件/image%2028.png)

![colocate async 的 scheduler state](图片和附件/image%205.png)

![部分 rank 的 running request 为 0](图片和附件/image.png)

结合时间线，这些空闲出现在 actor update 阶段。一个可能的解释是短请求先完成，剩余长请求在切换时被 abort。该判断需要结合请求级日志进一步验证：

![actor update 附近的长尾与 abort 现象](图片和附件/image%2026.png)

### 6.2 `warm_up_batch = 1`，PPO max token 提高到 3 倍

在该实验中，`colocate async` 获得了所有对比配置中的最高吞吐。同卡数下，相比同步模式，吞吐从 **593 提升到 746，约提升 25.8%**：

![colocate async 与同步模式的吞吐对比](图片和附件/image%207.png)

该结果说明 colocate 能有效缓解固定资源划分造成的失衡，但不能据此忽略 rank 间负载不均。吞吐达标后仍应检查长尾、abort 和质量指标。

## 7. 范式选择建议

| 场景                                         | 建议                                                      |
| ------------------------------------------ | ------------------------------------------------------- |
| Trainer 与 Rollouter 可使用独立资源，且负载相对稳定        | 优先使用 `separate async`，通过资源比获得可控的性能                      |
| 希望最大化流水线重叠，并能容忍和校正一定 off-policy 数据         | 评估 `fully async`，重点搜索 staleness 与 RS/IS 的组合             |
| 总资源固定且训练/推理速度随阶段变化明显                       | 优先评估 `colocate async`，减少静态资源划分造成的空泡                     |
| validation 占比高，Trainer 在 validation 期间明显空闲 | 评估 `use_trainer_do_validate`，让 Hybrid Engine 参与 rollout |
| 序列长度长尾明显                                   | 先解决请求分配和长尾问题，再增加设备；否则扩容可能只增加空闲 rank                     |

## 9. 核心结论

1. **先定位空泡属于哪一侧，再决定增减资源。** Rollouter 空闲时继续增加 Rollouter 通常无效。
2. **资源平衡优先于单侧峰值性能。** Qwen3-30B 案例中，减少 Rollout nodes 反而使吞吐提升约 25%。
3. **staleness 是吞吐与训练质量之间的旋钮。** 当等待时间已进入平台区后，不应继续仅为利用率增大 staleness。
4. **充分利用 Trainer 的可切换能力。** validation 期间让 Hybrid Engine 参与 rollout，可以消除固定周期空泡。
5. **显存余量可以转化为训练侧工作量。** 提高 PPO token budget 能缩小训推速度差，但必须监控稳定性。
6. **Colocate async 在资源失衡明显时具有优势。** 本文案例中，同卡数吞吐从 593 提升至 746，但长尾和 abort 仍需单独治理。

