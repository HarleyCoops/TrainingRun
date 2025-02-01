# GRPO Fine-Tuning Demo Roadmap

## Quickstart

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HarleyCoops/TrainingRun.git
   cd TrainingRun
   ```

2. **Create and activate a virtual environment:**
   - **On Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```
   - **On macOS/Linux/WSL:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *For WSL, use:* 
   ```bash
   python3 -m pip install -r requirements.txt
   ```

4. **Configure your environment:**
   - Edit the `.env` file in the repository root and add your API keys (for example, your WandB API key):
     ```
     WANDB_API_KEY=your_api_key_here
     ```

5. **Run the training script:**
   - **Using a Virtual Environment:**
     ```bash
     python grpo_demo.py
     ```
     *In WSL, use:*
     ```bash
     python3 grpo_demo.py
     ```

6. **For Notebook Users:**
   - Open the companion Colab notebook `colab_notebook.ipynb` in Google Colab. Follow the included instructions to load the fine-tuned model, run inference tests, and visualize training performance metrics interactively.

7. **For WSL Users (on Windows):**
   - Open your WSL terminal.
   - Verify Python 3 is installed:
     ```bash
     python3 --version
     ```
     If not installed, run:
     ```bash
     sudo apt-get update
     sudo apt-get install python3 python3-pip
     ```
   - Navigate to your project directory (e.g., `/mnt/c/Users/admin/trainingrun`):
     ```bash
     cd /mnt/c/Users/admin/trainingrun
     ```
   - Install dependencies in WSL:
     ```bash
     python3 -m pip install -r requirements.txt
     ```
   - Run the training script:
     ```bash
     python3 grpo_demo.py
     ```

8. **Quickstart using Docker (Recommended for Isolation):**

   If dependency management proves too challenging, you can use Docker to run the training in an isolated Linux environment:

   - **Install Docker:**  
     Download and install [Docker Desktop](https://www.docker.com/products/docker-desktop) for your platform.

   - **Build the Docker Image:**  
     From the project root, run:
     ```bash
     docker build -t grpo-finetuning .
     ```

   - **Run the Docker Container:**  
     After building the image, start the container with:
     ```bash
     docker run --rm -it grpo-finetuning
     ```
     This will execute the training script (`grpo_demo.py`) inside a controlled Linux environment, ensuring all dependencies compile and run properly.

Enjoy experimenting with the GRPO fine-tuning demo!

---

## Overview

This project demonstrates a self-contained fine-tuning script, `grpo_demo.py`, which uses the GRPO (Gradient Reward Policy Optimization) approach to fine-tune a LLaMA 1B (or Qwen 1.5B when applicable) model (base provided by Ollama 1B) on a GSM8K-based dataset. Although the GRPO method is generally more complex, this prototype has been streamlined into a single training script.

The training process leverages:
- **Transformers** for model and tokenizer functionality.
- **Datasets** (via Hugging Face) for data ingestion.
- **TRL (Transformer Reinforcement Learning)** for GRPO training.
- **PEFT (Parameter-Efficient Fine-Tuning)** (currently available but commented out) for potential modular enhancements.
- **WandB** integration for logging performance metrics (with a possibility to integrate TensorBoard/gradio dashboards).

---

## Dependencies

The following key libraries and frameworks are used:
- **cmake:** For building C extensions.
- **torch:** For tensor computations and model training.
- **transformers==4.48.2:** To load the pre-trained model and tokenizer.
- **datasets:** For loading and processing the training dataset.
- **peft==0.13.0:** For low-resource, parameter-efficient fine-tuning.
- **trl:** Provides the GRPO training framework.
- **wandb:** Used for logging training metrics.
- **huggingface-hub==0.17.3:** Required for interfacing with Hugging Face.
- **vllm==0.6.6.post1:** Required by TRL.

> **Note:** Environment variables and API keys are managed via the `.env` file.
> 
> ### Windows Build Prerequisites
> 
> If you are running on Windows, please ensure you have the following before building vllm:
> - **CMake:** Install via `pip install cmake` or download the native installer from [cmake.org](https://cmake.org/download/). Ensure CMake is in your system PATH.
> - **Visual Studio Build Tools:** Confirm that the appropriate compiler tools (e.g., MSVC) are installed and configured.
> 
> These prerequisites help ensure that any C extensions (such as vllm._C) build successfully on Windows.

---

## Dependency Conflict Resolution

If you encounter dependency conflicts during installation, try the following steps:

1. **Gradio & MarkupSafe Conflict:**  
   Gradio requires `markupsafe~=2.0`. If a conflict arises, downgrade MarkupSafe:
   ```bash
   pip install markupsafe==2.1.0
   ```

2. **Litellm & HTTPX Conflict:**  
   Litellm requires `httpx<0.28.0` (and ≥0.23.0). Downgrade HTTPX if necessary:
   ```bash
   pip install httpx==0.27.0
   ```

3. **Datasets Conflicts:**  
   The datasets package may require older versions of `dill` and `fsspec`. Resolve by installing:
   ```bash
   pip install dill==0.3.8
   pip install fsspec==2024.9.0
   ```

4. **Manim & Numpy Conflict:**  
   Manim requires `numpy>=2.1` (for Python 3.10+). Ensure your numpy version meets this requirement:
   ```bash
   pip install numpy>=2.1
   ```

5. **TensorFlow & Protobuf/Wrapt Conflict:**  
   TensorFlow (tensorflow-intel) may require specific versions. Downgrade if necessary:
   ```bash
   pip install protobuf==3.20.3
   pip install wrapt==1.14.0
   ```

6. **Other Conflicts:**  
   Review the pip install output for any additional errors and adjust package versions accordingly.

---

## Dataset Overview

The training script currently uses the GSM8K dataset provided by OpenAI, which consists of elementary-level mathematical word problems. In this setup, the GSM8K dataset is loaded and its questions and answers are formatted for fine-tuning.

*Note:* While future iterations will explore adapting and fine-tuning on the Stoney Nakoda Q&A dataset, the current focus is on getting the GSM8K-based training pipeline up and running.

---

## Training Performance Monitoring

### Current Setup
The training script (`grpo_demo.py`) uses Weights & Biases (WandB) for logging metrics. Logging is configured to provide frequent updates on training progress, including hyperparameters, system metrics, and live training stats.

### Weights & Biases Integration
WandB enables you to quickly track experiments, visualize results, and identify model regressions. The integration involves:
- **Installation:** Install WandB using:
  ```bash
  pip install wandb
  ```
- **Setup:** Log in using:
  ```bash
  wandb login
  ```
  with your API key (managed in your `.env` file).
- **Logging:** The training script initializes a WandB run with configuration details (e.g., learning rate, batch size, epochs, run name). During training, relevant metrics such as loss and accuracy are logged for real-time visualization.
- **Example Code:**
  ```python
  import wandb
  wandb.init(
      project="GRPO-Fine-Tuning",
      config={
           "learning_rate": training_args.learning_rate,
           "adam_beta1": training_args.adam_beta1,
           "adam_beta2": training_args.adam_beta2,
           "weight_decay": training_args.weight_decay,
           "warmup_ratio": training_args.warmup_ratio,
           "batch_size": training_args.per_device_train_batch_size,
           "num_train_epochs": training_args.num_train_epochs,
           "run_name": training_args.run_name
      }
  )
  ```
- **Tracking:** As the model trains, WandB logs are updated in real time, allowing you to monitor training dynamics and system performance.

### Planned Enhancements
Beyond the current WandB setup, future improvements include:
- **Integrate TensorBoard:** Use `torch.utils.tensorboard.SummaryWriter` for supplementary visualization.
- **Explore Gradio Dashboards:** Investigate Gradio for interactive live monitoring interfaces.
- **Enhanced Callbacks:** Expand logging to capture additional metrics such as gradient norms and learning rate changes.

---

## Companion Colab Notebook

A companion Colab notebook will be created that:
- **Loads the Fine-Tuned Model:** Provides cells for loading the model and tokenizer.
- **Runs Inference Tests:** Allows interactive prompt input and model response observation.
- **Displays Live Performance Metrics:** Embeds visualizations using libraries such as Matplotlib, Plotly, or built-in dashboard tools.
- **Contains Step-by-Step Instructions:** Guides users through replicating the training environment and executing inference.

---

## Academic-Level Discussion & Future Directions

This roadmap builds on and is inspired by several foundational works in reward optimization, reinforcement learning, and alignment for language models. Key papers that inform the design and future extensions of the GRPO method include:

1. **Group Robust Preference Optimization in Reward-free RLHF**  
   [https://arxiv.org/abs/2405.20304](https://arxiv.org/abs/2405.20304)  
   This paper introduces robust strategies for optimizing preferences without explicit rewards, providing a theoretical foundation for group-level optimization.

2. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**  
   [https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)  
   This work proposes advanced techniques for boosting the reasoning capabilities of language models, informing our fine-tuning strategy.

3. **REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models**  
   [https://arxiv.org/abs/2501.03262](https://arxiv.org/abs/2501.03262)  
   This paper presents a streamlined method for aligning language models using reinforcement learning, influencing our GRPOTrainer configuration.

4. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**  
   [https://arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)  
   This work explores reinforcement learning strategies for enhancing reasoning performance in large language models and offers practical insights for future enhancements.

Future directions will leverage these insights to further refine model alignment, enhance reasoning capabilities, and improve parameter efficiency. We also plan to incorporate community feedback iteratively to ensure the approach remains scalable and robust.

---

## Conclusion

This roadmap provides a comprehensive plan for running your GRPO fine-tuning code. It outlines two approaches: using a virtual environment in WSL and using Docker for a fully isolated Linux environment. Follow the provided instructions to set up your environment, install dependencies, and run your training script.

Happy fine-tuning!
