# Async RL Training Tuning Guide

This guide is intended for users who analyze asynchronous RL training jobs with RL Insight. It explains how to identify load imbalance between the Trainer and Rollouter and how to tune three execution modes: `separate async`, `fully async`, and `colocate async`.

The findings are based on experiments with Qwen3-30B, Qwen3.5-35B, and Qwen3-Omni. Actual gains depend on the model, sequence length, hardware, parallel strategy, and code version. Always validate changes with controlled experiments on the target workload.

## 1. Tuning Objectives

The goal of asynchronous training is not to maximize Trainer or Rollouter utilization in isolation. The goal is to improve end-to-end throughput without violating training quality requirements. Focus on three types of issues:

1. **Rollouter bubbles**: Rollouters finish all currently allowed samples and wait for the Trainer to update parameters or release more generation quota.
2. **Trainer bubbles**: The Trainer waits for the replay buffer to produce enough samples before it can continue updating.
3. **Effectiveness loss**: Increasing asynchrony raises sample staleness and may cause reward instability or training bias.

Monitor the following signals together:

| Dimension | Key signals | Purpose |
| --- | --- | --- |
| Timeline | Active Trainer/Rollouter intervals, parameter versions, validation intervals | Identify which side is idle and whether the bubbles are periodic |
| Rollout load | Requests per replica, `running request`, scheduler state, cache utilization | Detect uneven request distribution and idle ranks |
| Training load | Mini-batch execution time, training memory, `timing_s/gen` | In `fully async`, `timing_s/gen` reflects how long the Trainer waits for samples |
| Data distribution | Input and output sequence length distributions | Long-tail sequences can create imbalance across replicas and ranks |
| Training quality | Reward, accuracy, staleness, and RS/IS metrics | Verify that throughput improvements do not cause unacceptable quality degradation |

## 2. Async Modes and Primary Tuning Knobs

| Mode | Resource layout | Main trade-off | Primary knobs |
| --- | --- | --- | --- |
| `separate async` | The Trainer and standalone Rollouters use separate resource pools; Hybrid Engines may also be present | Fixed resources can be overprovisioned when training and inference speeds do not match | Trainer/Rollouter ratio, `use_trainer_do_validate`, `warm_up_batch`, PPO token budget |
| `fully async` | Training and rollout run continuously, and samples from older policies may be consumed | Pipeline bubbles versus off-policy severity | Staleness, Rollouter scale, parameter sync/trigger interval, RS/IS |
| `colocate async` | Training and inference time-share the same devices | Switching overhead, long-tail requests, and abort-related utilization loss | `warm_up_batch`, PPO token budget, request distribution |

> [!NOTE]
> Parameter names may vary by branch or version. This guide uses the names from the experiment environment, including `parameter_sync_step`, `warm_up_batch`, and `use_trainer_do_validate`.

## 3. Recommended Tuning Workflow

### 3.1 Establish a Baseline

Keep the model, dataset, total device count, and training hyperparameters fixed. Record at least:

- End-to-end throughput;
- Trainer and Rollouter device counts;
- Train batch, PPO mini-batch, and PPO max token;
- `parameter_sync_step`, `warm_up_batch`, and staleness;
- Validation interval;
- Reward and accuracy curves.

Change only one class of variables per experiment, and compare steady-state throughput outside initialization and warm-up intervals.

### 3.2 Locate the Bottleneck from the Timeline

| Observation | Likely cause | First action |
| --- | --- | --- |
| Rollouters are idle for long periods while the Trainer keeps updating | Excess inference resources or an overly strict staleness limit | Reduce Rollouters in `separate async`; evaluate higher staleness first in `fully async` |
| The Trainer periodically waits for samples | Insufficient rollout throughput or not enough consumable samples within a parameter sync interval | Increase effective rollout capacity, raise the PPO token budget, or adjust staleness |
| The Trainer is idle during validation | The Trainer does not participate in validation rollout | Evaluate `use_trainer_do_validate` |
| Some replicas or ranks are idle | Uneven sequence lengths, indivisible request counts, or scheduling tails | Check sequence lengths, replica count, and request assignment |
| Rollouters quickly exhaust the staleness quota and then become idle | Too many Rollouters or insufficient staleness | Do not add more Rollouters blindly; compare reducing resources with increasing staleness |
| Reward becomes more volatile as staleness increases | Higher off-policy severity | Reduce staleness or revise the RS/IS strategy |

### 3.3 Tune from Lower-Risk to Higher-Risk Changes

1. Eliminate deterministic bubbles first, such as validation switching gaps or invalid resource divisibility.
2. Rebalance Trainer and Rollouter resources so that neither side stays far ahead.
3. Adjust batch size, PPO token budget, and `warm_up_batch` to increase useful work per phase.
4. Tune parameters that may affect training quality, such as staleness and RS/IS, last.
5. After every change, check throughput, the timeline, and quality metrics together. Do not optimize a single utilization metric in isolation.

## 4. Separate Async Tuning

### 4.1 Understand Hybrid Engines and Standalone Replicas

In the experimental `separate async` implementation:

1. The base `_setup()` runs first and creates Hybrid Engines.
2. `init_standalone()` then creates an independent resource pool and starts new rollout servers.
3. Standalone replica ranks start after the Hybrid Engine replicas to avoid Ray named actor conflicts.

The replica count shown in monitoring can therefore be larger than the number of standalone Rollouters. In the following example, there are 32 replicas in total, and replicas `0-7` are Hybrid Engines colocated with the Trainer.

![Separate async replica initialization and rank assignment](图片和附件/image%2021.png)

![Hybrid Engines mapped to replicas 0-7](图片和附件/image%2011.png)

Before tuning, distinguish Hybrid Engines from standalone Rollouters. Do not infer standalone inference resources directly from the total replica count.

### 4.2 Let the Trainer Participate in Validation

In the experimental implementation, Hybrid Engines switch to rollout mode in `on_validate_begin()` and participate in validation, corresponding to the original `use_trainer_do_validate` behavior. After validation, they remain in inference mode until the next replay buffer sample completes and triggers `on_sample_end()`, at which point they switch back to the Trainer.

It is therefore expected to see Hybrid Engines generating around validation:

![Hybrid Engines generating during validation](图片和附件/image%209.png)

If validation runs at a fixed interval and the Trainer is clearly idle during it, enabling this feature can remove periodic bubbles. Include switching overhead in the evaluation and verify that the next training phase is not delayed.

### 4.3 Balance Trainer and Rollouter Resources

#### Case Study: Qwen3-30B

Baseline configuration:

| Parameter | Value |
| --- | --- |
| Train nodes | 16 |
| Rollout nodes | 48 |
| Validation interval | Every 5 steps |
| `gen_tp` | 2, meaning 2 devices per replica |

The baseline timeline shows substantial Rollouter bubbles:

![Qwen3-30B baseline timeline](图片和附件/image%208.png)

The sequence length distribution is shown below. Length variance must be considered when analyzing load imbalance:

![Qwen3-30B sequence length distribution](图片和附件/img_v3_0213l_0df55db5-647a-4fe0-ac70-39ee23faef9g.jpg)

Reducing Rollout nodes from 48 to 32 decreased inference bubbles and improved end-to-end throughput by approximately **25%**:

![Throughput after reducing Rollout nodes from 48 to 32](图片和附件/image%2024.png)

This case demonstrates that adding inference resources does not necessarily improve throughput when Rollouters consistently run ahead of the Trainer. Removing excess Rollouters can reduce bubbles and improve throughput per device.

### 4.4 Increase Useful Training-Side Work

#### Case Study: Qwen3-Omni

Initial configuration:

| Parameter | Value |
| --- | --- |
| Train batch | 256 |
| PPO mini-batch | 32 |
| `parameter_sync_step` | 8 |
| `warm_up_batch` | 1 |
| PPO max token | 10k |

With `r8t8`, Hybrid Engines can participate in rollout before the first replay buffer sample completes. However, inference is much faster than training overall, so Rollouters run approximately five PPO mini-batches ahead of the Trainer and then remain idle.

![r8t8 timeline](图片和附件/image%203.png)

![r8t8 load details](图片和附件/image%2022.png)

![r8t8 bubble details](图片和附件/image%2016.png)

In the `r4t12` experiment, enabling switching unexpectedly reduced the number of requests handled by each replica:

![Requests per replica after enabling switching with r4t12](图片和附件/image%2018.png)

This behavior must be interpreted together with the total request count and divisibility across replicas. The experiment added Trainer resources to satisfy the distribution constraint:

![Request distribution after adding Trainer resources](图片和附件/image%202.png)

For comparison, the following result uses `r8` with switching disabled:

![r8 with switching disabled](图片和附件/image%2014.png)

With `r8t16`, increasing `warm_up_batch` to 2 while keeping the other core parameters unchanged reduced the bubble from approximately five to three PPO mini-batches, but there was still room for improvement:

![r8t16 with warm_up_batch=2](图片和附件/image%206.png)

Training memory still had headroom:

![Training memory with r8t16](图片和附件/image%2025.png)

The PPO max token value was then raised from 10k to 30k to increase training-side work per iteration. The rollout bubble decreased further to approximately 1.5 mini-batches:

![Timeline after increasing PPO max token to 30k](图片和附件/image%2013.png)

Training memory utilization increased, but this range showed some instability. Continue monitoring OOMs, step-time variance, and training quality:

![Training memory with PPO max token set to 30k](图片和附件/image%2019.png)

The overall throughput comparison is shown below:

![Throughput comparison across separate async configurations](图片和附件/image)

The tuning sequence in this case was to rebalance training and rollout resources first, then use available memory to raise the PPO token budget. Before increasing the token budget, confirm memory headroom and retain a safety margin.

## 5. Fully Async Tuning

### 5.1 Use Staleness to Cover Pipeline Bubbles

Staleness allows Rollouters to continue generating with older parameter versions, increasing overlap between training and rollout. It can reduce waiting time but also raises off-policy severity, so throughput and training quality must be evaluated together.

#### Case Study: Qwen3.5-35B with staleness = 0.3

The yellow lines in the following timeline mark parameter version boundaries:

![Parameter version timeline with staleness=0.3](图片和附件/image%2027.png)

![Load overview with staleness=0.3](图片和附件/img_v3_0213m_4a24fab5-e4b3-4c0c-9b77-a8b644ba134g.jpg)

The timeline reveals two types of bubbles:

1. **Rollouter side**: After producing the sample quota corresponding to approximately `1 + staleness`, Rollouters still wait for the Trainer to finish its update.
2. **Trainer side**:
   - Validation every five steps makes the Trainer periodically idle. `use_trainer_do_validate` can switch the Trainer to rollout mode so it participates in validation.
   - The current trigger step is 8. In the experiment, `0.3 / (1/8) > 2`, so extra samples can cover the first two trigger steps. Later Trainer bubbles are mainly caused by waiting for Rollouters.

![Bubble breakdown with staleness=0.3](图片和附件/image%2020.png)

There are two candidate strategies:

- **Add Rollouters**: This may shorten the Trainer's wait for samples, but Rollouters may exhaust the staleness quota sooner and remain idle later in the iteration. If Rollouters already consume most resources, overall resource efficiency may decrease.
- **Increase staleness**: This allows otherwise idle Rollouters to generate more samples and typically covers more bubbles, but it introduces greater training-quality risk.

### 5.2 Compare Staleness Settings

**staleness = 0.3**

![Load with staleness=0.3](图片和附件/image%2010.png)

**staleness = 0.5**

![Load with staleness=0.5](图片和附件/img_v3_0213r_2d57c6bf-cea7-4670-b2e7-0a77ca2f088g.jpg)

**staleness = 0.7**

![Load with staleness=0.7](图片和附件/image%2015.png)

Experimental findings:

1. Increasing staleness noticeably improves bubble coverage between the Trainer and Rollouters from step 2 onward.
2. In these `fully async` experiments, `timing_s/gen` is the time the Trainer waits for Rollouters to generate samples. When `staleness >= 0.5`, this metric fluctuates within a similar range across configurations, indicating limited marginal throughput benefit from further increases.
3. Reward becomes substantially more volatile as staleness increases, while accuracy continues to rise relatively steadily.
4. The RS + IS strategy has a significant effect on reward progression. Staleness must not be selected from system throughput alone.

![timing_s/gen across staleness settings](图片和附件/image%2023.png)

![Reward across staleness settings](图片和附件/image%2012.png)

![Accuracy across staleness settings](图片和附件/image%204.png)

![Comparison of RS + IS strategies](图片和附件/image%201.png)

For this case, `0.5` is a more reasonable starting candidate: it removes most of the waiting while retaining lower off-policy risk than `0.7`. This is not a universal optimum and must be validated on the target workload.

## 6. Colocate Async Tuning

`colocate async` switches the same devices between training and inference, avoiding persistent imbalance caused by static resource partitioning. Compared with `separate async`, there are fewer resource-ratio knobs. Tuning should focus on:

- Selecting an appropriate `warm_up_batch`;
- Using the PPO token budget to balance training and inference phase duration;
- Detecting long-tail requests, aborts, and idle ranks;
- Measuring phase-switching overhead.

> [!IMPORTANT]
> At the time of the original experiment, `colocate async` support had not yet been merged into the GitHub main branch. The experiment used the `npu_0917_qjh` branch. Check the implementation and configuration of the target version before applying these findings.

### 6.1 `warm_up_batch = 2` with the Baseline PPO Max Token

The timeline for this configuration is shown below:

![Colocate async timeline with warm_up_batch=2](图片和附件/image%2017.png)

Cache utilization and scheduler state show that some ranks still have `running request = 0` in every step:

![Cache utilization in colocate async](图片和附件/image%2028.png)

![Scheduler state in colocate async](图片和附件/image%205.png)

![Some ranks have zero running requests](图片和附件/image.png)

According to the timeline, the idle intervals occur during actor update. One possible explanation is that short requests finish first while the remaining long requests are aborted during the switch. Request-level logs are required to verify this hypothesis:

![Long-tail and abort behavior around actor update](图片和附件/image%2026.png)

### 6.2 `warm_up_batch = 1` with 3x PPO Max Token

In this experiment, `colocate async` achieved the highest throughput among all compared configurations. With the same number of devices, throughput increased from **593 to 746**, an improvement of approximately **25.8%** over synchronous mode:

![Throughput comparison between colocate async and synchronous mode](图片和附件/image%207.png)

This result shows that colocation can effectively mitigate imbalance caused by static resource partitioning. It does not eliminate the need to investigate load imbalance across ranks. After throughput reaches the target, continue checking long tails, aborts, and quality metrics.

## 7. Mode Selection Recommendations

| Scenario | Recommendation |
| --- | --- |
| The Trainer and Rollouters can use separate resources, and their loads are relatively stable | Start with `separate async` for predictable tuning through the resource ratio |
| Maximum pipeline overlap is required, and some off-policy data can be tolerated and corrected | Evaluate `fully async`, focusing on the combination of staleness and RS/IS |
| Total resources are fixed, and training/inference speed changes substantially by phase | Evaluate `colocate async` first to reduce bubbles caused by static partitioning |
| Validation is expensive and the Trainer is clearly idle during validation | Evaluate `use_trainer_do_validate` so Hybrid Engines can participate in rollout |
| Sequence lengths have a strong long tail | Fix request assignment and tail behavior before adding devices; otherwise, scaling may only create more idle ranks |

## 8. Tuning Checklist

### Before the Experiment

- [ ] Fix the model, dataset, total device count, and quality evaluation method.
- [ ] Record the Trainer/Rollouter topology and parallel settings such as TP and DP.
- [ ] Confirm whether the replica count includes Hybrid Engines.
- [ ] Record key validation, parameter sync, and replay buffer settings.

### During the Experiment

- [ ] Collect timelines, device utilization, request states, and sequence length distributions together.
- [ ] Distinguish initialization, warm-up, validation, and steady-state training intervals.
- [ ] Change only one class of variables at a time and retain a baseline.
- [ ] Check whether requests can be distributed reasonably across replicas and ranks.

### After the Experiment

- [ ] Compare end-to-end throughput rather than only local kernel or phase timing.
- [ ] Verify that bubbles are removed rather than shifted from one side to the other.
- [ ] Check reward, accuracy, and staleness for quality regressions.
- [ ] For high PPO token budgets, check peak memory, OOMs, and step-time stability.
- [ ] For `colocate async`, check long-tail requests, aborts, and switching overhead.

## 9. Key Takeaways

1. **Identify which side owns the bubble before changing resources.** Adding Rollouters is usually ineffective when Rollouters are already idle.
2. **Resource balance matters more than peak performance on one side.** In the Qwen3-30B case, reducing Rollout nodes improved throughput by approximately 25%.
3. **Staleness is a knob between throughput and training quality.** Once waiting time reaches a plateau, do not keep increasing staleness only to improve utilization.
4. **Use the Trainer's switching capability when appropriate.** Letting Hybrid Engines participate in rollout during validation can remove deterministic periodic bubbles.
5. **Convert memory headroom into useful training work.** A higher PPO token budget can reduce the training/rollout speed gap, but stability must be monitored.
6. **Colocate async is effective when resource imbalance is substantial.** In this case, throughput increased from 593 to 746 with the same device count, but long tails and aborts still require separate treatment.
