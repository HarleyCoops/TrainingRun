---
library_name: transformers
tags: []
---

# Model Card for Model ID

<!-- Provide a quick summary of what the model is/does. -->

This model is a fine-tuned version of Qwen2.5-0.5B-Instruct, trained using Generative Reinforcement Policy Optimization (GRPO) on the gsm8k math dataset. It is designed to solve math problems by generating reasoning steps and answers in a specific XML format.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model card describes a 🤗 transformers model that has been pushed on the Hub. This model card has been automatically generated.

- **Developed by:** HarleyCoops
- **Funded by [optional]:** [More Information Needed]
- **Shared by [optional]:** [More Information Needed]
- **Model type:** Causal Language Model
- **Language(s) (NLP):** English
- **License:** Apache 2.0
- **Finetuned from model [optional]:** Qwen/Qwen2.5-0.5B-Instruct

### Model Sources [optional]

<!-- Provide the basic links for the model. -->

- **Repository:** https://github.com/HarleyCoops/TrainingRun
- **Paper [optional]:** https://arxiv.org/abs/2309.16676 (Qwen paper)
- **Demo [optional]:** [More Information Needed]

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The model is intended for solving math problems presented in English. It generates reasoning steps and an answer, formatted in XML tags. It can be used as a standalone tool for math problem-solving or integrated into a larger application.

### Downstream Use [optional]

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The model can be further fine-tuned on other math datasets or used as a component in a more complex system that requires mathematical reasoning.

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

The model is specifically trained for math problem-solving. It may not perform well on tasks outside of this domain, such as general language understanding or generation. It may also struggle with math problems that require knowledge outside of the training data.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model's performance is limited by the quality and quantity of the training data. It may exhibit biases present in the gsm8k dataset. The model's ability to generalize to unseen math problems is also limited.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Use the code below to get started with the model.

```python
# Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# First, load the base model architecture
base_model = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Load the fine-tuned weights
checkpoint_path = "outputs/Qwen-0.5B-GRPO/checkpoint-1868"  # Specific checkpoint folder
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    checkpoint_path,
    trust_remote_code=True
)

# Rest of the inference code remains the same
SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def solve_math_problem(question: str):
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question}
    ]

    input_text = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test with a sample problem
test_question = "A train travels at 60 miles per hour. If the journey is 270 miles long, how many hours will the trip take?"
print("\nQuestion:", test_question)
print("\nResponse:", solve_math_problem(test_question))
```

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

The model was trained on the gsm8k dataset, which contains a set of grade school math problems.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

The model was fine-tuned using Generative Reinforcement Policy Optimization (GRPO). The training process involved defining reward functions to encourage correct answers and proper formatting.

#### Preprocessing [optional]

The dataset was preprocessed to format the questions and answers in a conversational prompt format.

#### Training Hyperparameters

- **Training regime:** bf16 mixed precision
- **Learning Rate:** 5e-6
- **Adam Beta1:** 0.9
- **Adam Beta2:** 0.99
- **Weight Decay:** 0.1
- **Warmup Ratio:** 0.1
- **LR Scheduler Type:** cosine
- **Per Device Train Batch Size:** 1
- **Gradient Accumulation Steps:** 4
- **Number of Generations:** 16
- **Max Prompt Length:** 256
- **Max Completion Length:** 200
- **Number of Train Epochs:** 1
- **Max Grad Norm:** 0.1

#### Speeds, Sizes, Times [optional]

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The model was evaluated on the gsm8k dataset.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

[More Information Needed]

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

The primary metric is the correctness of the generated answer. Additional metrics include adherence to the specified XML format.

### Results

[More Information Needed]

#### Summary

[More Information Needed]

## Model Examination [optional]

<!-- Relevant interpretability work for the model goes here -->

[More Information Needed]

## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** [More Information Needed]
- **Hours used:** [More Information Needed]
- **Cloud Provider:** [More Information Needed]
- **Compute Region:** [More Information Needed]
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications [optional]

### Model Architecture and Objective

[More Information Needed]

### Compute Infrastructure

[More Information Needed]

#### Hardware

[More Information Needed]

#### Software

[More Information Needed]

## Citation [optional]

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

[More Information Needed]

**APA:**

[More Information Needed]

## Glossary [optional]

<!-- If relevant, include terms and calculations in this section that can help readers understand the model or model card. -->

[More Information Needed]

## More Information [optional]

[More Information Needed]

## Model Card Authors [optional]

[More Information Needed]

## Model Card Contact

[More Information Needed]
