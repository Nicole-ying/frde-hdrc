# FDRE-HRDC Agent 自主奖励搜索交付说明

## 1. 核心定位

本交付强调的是无人工领域知识的奖励函数自进化过程。系统不预先写入 LunarLander 的状态语义，也不人工规定“距离、速度、姿态、触地”等具体奖励项；agent 只接收通用环境接口、原始环境奖励、训练日志、评估分数、episode 长度、奖励函数运行错误等黑盒反馈，然后通过“生成候选奖励函数 - 短程训练 - 诊断反馈 - 再生成 - 多 seed 验证”的闭环搜索出可用奖励函数。

设环境为马尔可夫决策过程：

$$
\mathcal{M}=(\mathcal{S},\mathcal{A},P,r^{env},\gamma),
$$

其中 $s_t\in\mathcal{S}$ 为状态，$a_t\in\mathcal{A}$ 为动作，$P(s_{t+1}|s_t,a_t)$ 为状态转移，$r_t^{env}$ 为环境原始奖励。FDRE-HRDC 不改变最终评价口径，训练阶段可使用 agent 搜索得到的辅助奖励，评估阶段统一使用环境原始奖励：

$$
r_t^{train}=f_{\theta}(s_t,a_t,s_{t+1},r_t^{env},p_t),
\qquad
R^{eval}=\sum_{t=0}^{T}r_t^{env}.
$$

因此报告中的 baseline、对比实验、消融实验和 1M 稳定性验证都可以直接按 LunarLander-v3 的 solved 标准比较。

## 2. 无领域知识搜索闭环

第 $k$ 轮迭代中，agent 观察到的信息定义为：

$$
o_k=\{\mathcal{D}_k,\mathcal{H}_k,\mathcal{E}_k,\mathcal{I}\},
$$

其中 $\mathcal{D}_k$ 是环境交互产生的转移样本，$\mathcal{H}_k$ 是历史候选奖励函数及其训练结果，$\mathcal{E}_k$ 是 reward runtime error、训练中断和评估统计，$\mathcal{I}$ 只包含 observation/action space 这类通用接口描述。系统不向 agent 注入环境专属状态含义。

候选奖励函数由 agent 根据当前反馈自动生成：

$$
\mathcal{F}_k=A_{\phi}(o_k,M_k)
=\left\{f_{\theta_1}^{(k)},f_{\theta_2}^{(k)},\ldots,f_{\theta_m}^{(k)}\right\},
$$

其中 $A_{\phi}$ 表示大模型驱动的奖励搜索 agent，$M_k$ 表示搜索记忆。每个候选函数只允许使用：

$$
f_{\theta}(s_t,a_t,s_{t+1},r_t^{env},p_t),
$$

即当前观测、动作、下一观测、原始奖励和训练进度，不能使用人工补充的环境规则。

## 3. HRDC 结构

HRDC 在当前交付中不表示人工预定义的 LunarLander 子目标，而表示 agent 在候选程序中自动形成的层次化奖励结构。其一般形式为：

$$
r_t^{FDRE}
=\lambda_0 r_t^{env}
+\sum_{i=1}^{m}w_i(p_t)\phi_i^{(k)}(s_t,a_t,s_{t+1},r_t^{env})
-c^{(k)}(a_t),
$$

其中 $\phi_i^{(k)}$、$w_i(p_t)$ 和 $c^{(k)}$ 均由 agent 在第 $k$ 轮候选奖励函数中生成，不由人工指定具体物理含义。阶段权重采用通用形式：

$$
w_i(p_t)=(1-p_t)w_i^{early}+p_t w_i^{late},
$$

用于表达训练早期和训练后期对不同候选信号的关注差异。该设计保留了可解释的层次结构，但避免将环境知识直接写死到奖励函数中。

## 4. 训练反馈与自我调试

每一轮候选函数都经过运行检查、短程训练和评估统计。训练反馈定义为：

$$
g_k=G(\bar R_k,S_k,L_k,E_k,I_k),
$$

其中：

$$
\bar R_k=\frac{1}{n}\sum_{j=1}^{n}R_{j,k}^{eval},
\qquad
S_k=\frac{1}{n}\sum_{j=1}^{n}\mathbb{I}(R_{j,k}^{eval}\ge 200),
$$

$$
L_k=\frac{1}{n}\sum_{j=1}^{n}T_{j,k},
\qquad
E_k=\text{reward error count},
\qquad
I_k=\text{interruption indicator}.
$$

agent 的调试依据不是人工告诉它“应该奖励什么”，而是根据以下黑盒证据自我修正：分数是否提高、episode 是否过早终止、reward 函数是否报错、是否出现训练中断、当前候选是否弱于历史最优候选。

## 5. 候选选择目标

系统对每个候选奖励函数使用统一目标进行筛选：

$$
J(f)=\bar R(f)+\alpha S(f)-\beta L(f)-\gamma E(f)-\eta I(f)-\rho C(f),
$$

其中 $\bar R(f)$ 是原始环境奖励下的平均评估分数，$S(f)$ 是 solved 成功率，$L(f)$ 是平均 episode 长度，$E(f)$ 是奖励函数运行错误数，$I(f)$ 是训练中断指示，$C(f)$ 是奖励函数复杂度。最终选择：

$$
f^{*}=\arg\max_{f\in\cup_k\mathcal{F}_k}J(f).
$$

这个目标使系统不是单纯追求某一次高分，而是在最终性能、稳定性、可运行性和复杂度之间进行自动筛选。

## 6. 与人工奖励设计的区别

本方法中的人工部分只定义实验协议、安全边界和评价目标，例如使用 PPO、使用 100k/1M timesteps、使用原始环境奖励评估、统计 reward error 和 interruption。奖励函数的具体形式、候选信号组合、阶段权重、是否保留或回退到历史最优，均由 agent 基于训练反馈自动完成。交付代码中保留的固定奖励程序仅作为 legacy/canonical 对照和消融参考，主 FDRE-HRDC 路径已经改为调用 RewardEvolver 进行自主迭代搜索。

## 7. 实验结果

1M timesteps 稳定性验证中，FDRE-HRDC 三个 seed 的原始环境得分为：

$$
R_{42}=223.40,\qquad R_{43}=219.98,\qquad R_{44}=234.74.
$$

因此：

$$
\mu_R=226.04,\qquad \sigma_R=6.31,\qquad \min_jR_j=219.98>200.
$$

这说明在 1M 训练预算下，FDRE-HRDC 不仅平均分超过 solved 标准，而且三个随机种子全部超过 200 分，具备稳定复现的交付价值。

100k timesteps 对比实验用于展示样本效率和迭代搜索优势：

$$
R_{FDRE}=159.80\pm8.10,
$$

$$
R_{PPO}=-4.57\pm41.13,\qquad R_{LLM}= -58.01\pm51.44,
$$

$$
R_{w/o\ diag}=27.91\pm21.78,\qquad R_{w/o\ dyn}=19.16\pm38.15.
$$

相对提升为：

$$
\Delta_{FDRE-PPO}=164.37,\qquad
\Delta_{FDRE-LLM}=217.81,
$$

$$
\Delta_{FDRE-diag}=131.89,\qquad
\Delta_{FDRE-dyn}=140.64.
$$

该结果体现了两个层面的效果：短训练预算下，agent 自主搜索能比原始 PPO 和单次 LLM 生成更快找到有效奖励；长训练预算下，最终策略能稳定超过 200 分 solved 标准。

## 8. 稳定性结果

当前交付记录中，奖励函数运行错误次数为：

$$
E=0.
$$

训练因奖励函数错误中断次数为：

$$
I=0.
$$

这说明奖励函数生成、注入、运行检查、训练反馈和候选筛选流程能够稳定完成实验，没有出现由于奖励函数语法错误、运行错误或不安全访问导致实验中断的问题。

## 9. 交付结论

FDRE-HRDC 的创新点可以概括为：以 agent 自主奖励搜索替代人工领域奖励设计，通过黑盒训练反馈驱动候选奖励函数自我迭代，并用 HRDC 结构保留可解释的阶段化奖励组合。当前交付在 LunarLander-v3 上验证了该闭环的有效性：100k 实验中明显优于 PPO baseline、单次 LLM 和消融版本，1M 实验中三个 seed 全部超过 200 分，平均得分达到 $226.04\pm6.31$，同时 reward error 和训练中断均为 0。该结果可以支撑“agent 技术能够在无人工领域知识输入的条件下搜索出更优奖励函数，并提升复杂控制任务的样本效率与最终策略稳定性”的论文式表述。
