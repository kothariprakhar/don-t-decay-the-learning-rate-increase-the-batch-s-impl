# Don't Decay the Learning Rate, Increase the Batch Size

It is common practice to decay the learning rate. Here we show one can usually obtain the same learning curve on both training and test sets by instead increasing the batch size during training. This procedure is successful for stochastic gradient descent (SGD), SGD with momentum, Nesterov momentum, and Adam. It reaches equivalent test accuracies after the same number of training epochs, but with fewer parameter updates, leading to greater parallelism and shorter training times. We can further reduce the number of parameter updates by increasing the learning rate $ε$ and scaling the batch size $B \propto ε$. Finally, one can increase the momentum coefficient $m$ and scale $B \propto 1/(1-m)$, although this tends to slightly reduce the test accuracy. Crucially, our techniques allow us to repurpose existing training schedules for large batch training with no hyper-parameter tuning. We train ResNet-50 on ImageNet to $76.1\%$ validation accuracy in under 30 minutes.

## Implementation Details

## Deep Dive: Don't Decay the Learning Rate, Increase the Batch Size

### 1. The Research Hypothesis
Standard Deep Learning wisdom dictates that as training progresses, we should decay the Learning Rate (LR). The intuition is that initially, we take large steps to traverse the loss landscape quickly, but as we approach a minimum, we must take smaller steps to settle into the optimal solution without overshooting.

Smith et al. challenge this implementation detail. They argue that the **noise scale** in Stochastic Gradient Descent (SGD) is the governing factor for convergence, not just the step size. 

The stochastic gradient update is defined as:
$$\theta_{t+1} = \theta_t - \epsilon \hat{g}_B(\theta_t)$$
where $\epsilon$ is the learning rate and $\hat{g}_B$ is the gradient estimated from a batch of size $B$. 

The paper demonstrates that the variance (noise) of this update is proportional to the ratio $\epsilon / B$. Therefore, **decaying $\epsilon$ by a factor of $\alpha$ is theoretically equivalent to increasing the batch size $B$ by the same factor $\alpha$.**

### 2. Implementation Walkthrough

#### The Data Strategy: CIFAR-10
We utilize CIFAR-10 as a proxy for ImageNet. While ImageNet requires massive compute, CIFAR-10 allows us to observe the same learning dynamics (initial rapid learning, followed by plateaus requiring noise reduction) in a fraction of the time. 

#### The Model: ResNet-18 (Adapted)
Lines `23-33` implement a modified ResNet-18. Standard ResNet models (provided by `torchvision`) are architected for 224x224 images. CIFAR-10 images are 32x32. Using a standard ResNet would result in the feature map shrinking to 1x1 too early. We replace the initial $7\times7$ convolution and max-pooling with a finer $3\times3$ convolution to preserve spatial dimensions for the smaller dataset.

#### The Core Innovation: `BatchSizeScheduler`
Instead of the standard `torch.optim.lr_scheduler.StepLR`, we implement a custom class `BatchSizeScheduler` (Lines `39-65`).

1.  **Dynamic Re-instantiation**: PyTorch `DataLoader` instances usually have fixed batch sizes. To increase the batch size dynamically, our scheduler destroys the old iterator and creates a new `DataLoader` with $B_{new} = B_{old} \times \text{multiplier}$ whenever a milestone epoch is reached.
2.  **No LR Decay**: Notice in the optimizer definition (Line `94`), we set a static `lr=0.1`. We never change this value during training.

### 3. Training Dynamics Analysis

During the execution of the code, you will observe the following phases:

1.  **Phase 1 (Epoch 0-20)**: High noise. Batch Size is 128. The model learns general features quickly. Accuracy rises to ~75-80%.
2.  **Phase 2 (Epoch 20)**: The `BatchSizeScheduler` kicks in. Batch Size jumps (e.g., to 640). 
    *   *Mathematical Equivalence*: This reduces the variance of the gradient estimate significantly. It is mathematically similar to dividing the LR by 5. 
    *   *System Effect*: The number of iterations (parameter updates) per epoch drops by 5x. However, the computational work per epoch (forward/backward passes) remains roughly the same (processing 50k images).
    *   *Result*: You will see a spike in accuracy similar to what is observed when decaying LR.
3.  **Phase 3 (Epoch 30)**: Another jump in Batch Size. The model refines the weights into a sharp minimum.

### 4. Why This Matters

This technique is crucial for **Large Scale Training**:
*   **Parallelism**: Increasing batch size is the easiest way to scale training across more GPUs. If you decay LR, you are bound by sequential updates. If you increase BS, you can distribute the larger batch across more hardware.
*   **Wall-Clock Speed**: While the number of epochs remains comparable, the number of *updates* decreases. On hardware with high communication overhead (or when the batch size is small relative to GPU memory), larger batches are more efficient.

The implementation provided allows you to validate that increasing batch size yields validation accuracy competitive with the traditional "decay LR" strategy, effectively unlocking massive parallelism for late-stage training.

## Verification & Testing

The code provides a mathematically valid implementation of the 'Don't Decay the Learning Rate, Increase the Batch Size' strategy (Smith et al.) adapted for CIFAR-10.

### Strengths
1.  **Architecture Adaptation**: The modification of ResNet18 (replacing the initial 7x7 convolution with 3x3 and removing the first MaxPool) is crucial for small 32x32 CIFAR images. Without this, the feature maps would become 1x1 too early in the network.
2.  **Scheduler Logic**: The `BatchSizeScheduler` correctly re-instantiates the `DataLoader` with the updated batch size at the specified milestones. This effectively reduces gradient noise variance, mimicking the dynamics of a learning rate decay.
3.  **Correct Training Loop**: The standard SGD optimization and evaluation loops are implemented correctly.

### Minor Observations & Limitations
1.  **VRAM Scaling**: The strategy assumes infinite GPU memory. Increasing batch size by 5x or 10x repeatedly (e.g., 128 -> 640 -> 3200) will quickly cause Out-Of-Memory (OOM) errors on most single GPUs. The code acknowledges this in comments, but users should be aware this strategy often requires distributed training (DataParallel or DDP) for later epochs.
2.  **Worker Overhead**: Re-creating the `DataLoader` inside `step()` tears down and respawns worker processes. While technically correct, this introduces a slight latency spike at the start of milestone epochs, though it is negligible compared to total training time.
3.  **Shuffle Reset**: Re-instantiating the loader resets the random sampler state. This is generally acceptable but strictly alters the randomness compared to a continuous loader.