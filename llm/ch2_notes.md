# LLM基础知识

## 构建

### 预训练

Definition: 使用与下游任务无关的大规模数据进行模型参数的初始训练，可以认为是为模型参数找到一个较好的“初值点”,方便后面根据特定任务优化微调。

### 指令微调与人类对齐

**指令微调(supervised fine-tuning)**：使用任务输入与输出的配对数据进行模型训练，属于模仿学习的范畴，目的也是加强模型对标准答案的复刻学习。主要起到了对于模型能力的**激发作用**，而不是知识注入作用。

**人类对齐**：Reinforcement Learning from Human Feedback.

## 扩展法则

### KM扩展法则：
模型规模（𝑁）、数据规模（𝐷）和计算算力（𝐶）的关系
$$
L(N) = \left( \frac{N_c}{N} \right)^{\alpha_N}, \quad \alpha_N \sim 0.076, N_c \sim 8.8 \times 10^{13}\\

L(D) = \left( \frac{D_c}{D} \right)^{\alpha_D}, \quad \alpha_D \sim 0.095, D_c \sim 5.4 \times 10^{13}\\

L(C) = \left( \frac{C_c}{C} \right)^{\alpha_C}, \quad \alpha_C \sim 0.050, C_c \sim 3.1 \times 10^8
$$

其中$L(\dot)$表示用以nat为单位的交叉熵损失。其中，𝑁𝑐、𝐷𝑐和𝐶𝑐是实验性的常数数值，分别对应于非嵌入参数数量、训练数据数量和实际的算力开销。

进一步的，可以分为不可约损失（真实数据分布决定）与可约损失（真实分布与模型分布间KL散度的估计）：
$$ \text L(x) = \text L_{\inf} + (\frac {x_0}{x})^{\alpha_x}$$

### Chinchilla 扩展法则
$$\text{L(N,D)} = \text{E} + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中𝐸 = 1.69, 𝐴 = 406.4, 𝐵 = 410.7，𝛼 = 0.34 和𝛽 = 0.28。

当给定C~6ND时，在算力资源固定下，数据规模分配如下：
$$
\text{N_opt(C)} = G(\frac{\text{C}}{6})^{a}\\
\text{D_opt(C)} = G^{-1}(\frac{\text{C}}{6})^{b}
$$
其中 ，$\text{a} = \frac{\alpha}{\alpha + \beta}$,$ \text{b} = \frac{\beta}{\alpha + \beta}$,G是由A B $\alpha$ $\beta$ 计算得到的扩展系数。

上面的式子可以近似看作两种扩展法则的公式，也就是以算力为核心时模型参数与数据量的分配。

## 涌现能力
Definition：大模型所具有的典型能力。
### 代表性能力
**上下文学习**：在提示中为语言模型提供自然语言指令和多个任务示例（Demonstration），无需显式的训练或梯度更新，仅输入文本的单词序列就能为测试样本生成预期的输出。同样取决于具体的下游任务。
**指令遵循**：大语言模型能够按照自然语言指令来执行对应的任务。需要使用指令微调（监督微调），基于自然语言描述的多任务实例数据集微调。
**逐步推理**：基于思维链（COT),在提示中引入任务相关的中间推理步骤来加强复杂任务的求解，从而获得更为可靠的答案。
