# Lab 03 – Developing a Tiny Transformer Language Model

## 1. Introduction

This practical focuses on developing a small Transformer language model from scratch using PyTorch. The main purpose is to understand the fundamental architecture of Transformer-based language models and how they learn patterns from text to generate new content.

For this experiment, the TinyStories dataset was obtained from Hugging Face. A subset of 1,500 stories was selected to train a character-level Transformer language model. The model was designed to predict the next character based on previously observed characters. Following training, the model was evaluated and tested by generating stories from different starting prompts.

The practical covers the complete language modelling workflow, including dataset exploration, preprocessing, tokenization, training data preparation, Transformer implementation, model training, evaluation, experimentation and model saving.

## 2. Objective

The objective of this practical is to develop an understanding of Transformer language models through hands-on implementation. This includes loading and exploring a text dataset, converting text into numerical representations, preparing sequences for next-character prediction, implementing a decoder-only Transformer architecture, training the model using PyTorch, evaluating its performance and generating new text using the trained model.

## 3. Tools and Technologies

The practical was implemented using Python and Jupyter Notebook.

The following libraries and technologies were used:

- **Python** – Main programming language.
- **PyTorch** – Implementing, training and evaluating the Transformer model.
- **Hugging Face Datasets** – Loading the TinyStories dataset.
- **NumPy** – Numerical operations and data processing.
- **Pandas** – Organising and analysing dataset and experimental results.
- **Matplotlib** – Visualising dataset characteristics and model performance.
- **Jupyter Notebook** – Development and execution environment.

The model automatically detects the available computing device and uses CUDA when available or CPU otherwise.

## 4. Dataset

**Dataset:** TinyStories

**Source:** [Hugging Face – TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)

TinyStories is a collection of short stories designed for research and experimentation with small language models. The dataset contains stories with relatively simple vocabulary and sentence structures.

For this practical, the first 1,500 stories from the training split were selected.

### Dataset Configuration

| Parameter | Description |
|---|---|
| Dataset | TinyStories |
| Source | Hugging Face |
| Selected stories | 1,500 |
| Text representation | Character-level |
| Tokenization | Custom character vocabulary |
| Training split | 90% |
| Validation split | 10% |

The selected stories were combined into a single text sequence. A vocabulary was created from the unique characters appearing in the text.

The actual vocabulary size and dataset statistics are calculated during notebook execution.

## 5. Implementation

### 5.1 Loading the Dataset

The TinyStories dataset was loaded from Hugging Face using the `datasets` library.

A subset of 1,500 stories was selected from the training split. The selected stories were converted into a list of text samples, and several examples were displayed to examine their content and structure.

### 5.2 Exploring the Dataset

The dataset was explored to understand the length and structure of the selected stories.

The number of words and characters in each story was calculated. Statistical measurements, including average, minimum and maximum story lengths, were examined.

A histogram was also generated using Matplotlib to visualise the distribution of story lengths.

### 5.3 Text Preprocessing and Tokenization

Character-level tokenization was implemented to convert text into numerical representations.

All selected stories were combined into a single text sequence. A vocabulary was then created using the unique characters found in the dataset.

Two mappings were implemented:

- `stoi` – Converts characters into numerical IDs.
- `itos` – Converts numerical IDs back into characters.

Custom encoding and decoding functions were developed to convert between text and numerical representations.

This approach allows the model to process text without requiring a pretrained tokenizer.

### 5.4 Preparing Training Data

The encoded text was divided into training and validation portions using a 90:10 split.

A sliding-window approach was used to prepare input and target sequences for next-character prediction.

Each target sequence was shifted one character forward relative to its corresponding input sequence. This allows the model to learn how to predict the next character using the preceding context.

The following configuration was used:

| Parameter | Value |
|---|---:|
| Training split | 90% |
| Validation split | 10% |
| Batch size | 32 |
| Block size | 128 |

### 5.5 Building the Transformer Architecture

A decoder-only Transformer language model was implemented from scratch using PyTorch.

The architecture consists of the following components:

- **Token Embeddings:** Convert input character IDs into numerical vectors.
- **Positional Embeddings:** Provide information about the position of each character in the sequence.
- **Masked Multi-Head Self-Attention:** Learns relationships between characters while preventing access to future positions.
- **Feed-Forward Networks:** Process the representations produced by the attention mechanism.
- **Residual Connections:** Support information flow through Transformer blocks.
- **Layer Normalization:** Helps stabilize the training process.
- **Dropout:** Reduces over-reliance on individual activations.
- **Linear Output Layer:** Produces predictions for the next character.

The model uses two Transformer blocks with four attention heads in each block and an embedding dimension of 128.

### 5.6 Model Initialization

The Transformer model was initialized using PyTorch.

The AdamW optimizer was used to update the model parameters, while CrossEntropyLoss was used as the loss function for next-character prediction.

The main training configuration included:

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer layers | 2 |
| Dropout | 0.1 |
| Learning rate | 0.001 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |

### 5.7 Training the Transformer

The model was trained for 10 epochs using mini-batch training.

During each training iteration, input sequences were passed through the Transformer to predict their corresponding target sequences.

The cross-entropy loss was calculated, and backpropagation was performed to update the model parameters.

AdamW was used for optimization, while gradient clipping was applied to control excessively large gradients.

A total of 50 training steps were performed during each epoch. Training and validation losses were recorded to monitor the model's learning progress.

### 5.8 Monitoring Training Performance

Training and validation losses were recorded after each epoch.

Matplotlib was used to generate a loss graph showing the changes in training and validation performance throughout the training process.

The graph provides an overview of how the model learns from the training data and how well it performs on the validation data.

### 5.9 Model Evaluation

The trained Transformer was evaluated using the validation dataset.

Two evaluation measurements were calculated:

- **Validation Loss:** Measures the prediction error when predicting the next character in the validation data.
- **Perplexity:** Measures the uncertainty of the model when predicting the next character. Lower perplexity generally indicates better predictive performance.

The final evaluation values were calculated after training and displayed in the notebook.

### 5.10 Story Generation

After training, the model was used to generate text based on different starting prompts.

The primary starting prompt was:

`Once upon a time`

Additional prompts were also tested:

- `One day, a little girl`
- `There was a small dog`
- `Once upon a time, a boy`

The model generated additional characters based on the provided prompts.

Temperature sampling and Top-K sampling were used during generation to control the selection of subsequent characters.

The generated outputs were examined to understand how the trained Transformer uses learned character patterns to produce text.

### 5.11 Experimenting with Model Settings

An additional experiment was conducted to investigate the effect of training duration on model performance.

Two separate Transformer models with the same architecture were trained:

| Experiment | Training Epochs |
|---|---:|
| Main experiment | 10 |
| Second experiment | 3 |

Both models used the same dataset, training configuration and architecture.

The training and validation losses were compared to investigate the effect of using different numbers of training epochs.

Since each experiment uses a separately initialized model, differences in the results may also be influenced by random initialization.

### 5.12 Saving the Trained Model and Results

The trained model and experimental outputs were saved in the `results/` directory.

The saved files include:

- `tiny_transformer.pth` – Trained Transformer model parameters.
- `vocabulary.json` – Character vocabulary required to interpret model token IDs.
- `model_config.json` – Model architecture and training configuration.
- `training_history.csv` – Training and validation loss history.
- `experiment_comparison.csv` – Comparison of the two training experiments.
- `generated_story.txt` – Main generated story.
- `generated_prompt_examples.txt` – Generated outputs from additional prompts.
- `training_loss.png` – Training and validation loss graph.
- `experiment_comparison.png` – Experimental validation loss comparison graph.

These outputs allow the trained model and its experimental results to be retained for future reference.

## 6. Results

The Transformer language model was trained using the selected TinyStories dataset.

### Model Evaluation Results

| Evaluation Metric | Result |
|---|---|
| Final Training Loss | Update after execution |
| Final Validation Loss | Update after execution |
| Perplexity | Update after execution |

### Training Progress

The training and validation loss values were recorded across all 10 epochs.

The generated training loss graph provides a visual representation of the model's learning progress.

**Evidence:** `results/training_loss.png`

### Story Generation Results

The trained model was tested using different starting prompts.

The generated outputs demonstrated the model's ability to predict subsequent characters and produce text based on patterns learned from the training dataset.

However, the generated stories may contain spelling mistakes, grammatical errors, repeated words and incomplete ideas. These limitations are expected from a relatively small character-level Transformer trained on a limited dataset.

**Evidence:**
- `results/generated_story.txt`
- `results/generated_prompt_examples.txt`

## 7. Experiment Comparison

Two training experiments were conducted using the same Transformer architecture.

The main experiment used 10 epochs, while the second experiment used 3 epochs.

| Experiment | Epochs | Training Loss | Validation Loss |
|---|---:|---:|---:|
| Main experiment | 10 | Update after execution | Update after execution |
| Second experiment | 3 | Update after execution | Update after execution |

The comparison was used to investigate how training duration influences the model's prediction performance.

The results should be interpreted alongside the fact that both models were initialized independently.

**Evidence:** `results/experiment_comparison.csv` and `results/experiment_comparison.png`

## 8. Performance Mechanisms

Several mechanisms were implemented to improve the stability and effectiveness of the Transformer training process.

**Causal Masking:** Prevents the model from accessing future characters during next-character prediction, ensuring that predictions are based only on previous context.

**Multi-Head Self-Attention:** Allows the model to learn different relationships between characters through multiple attention heads.

**Positional Embeddings:** Provide information about the order and position of characters within a sequence.

**Residual Connections:** Allow information to flow through the Transformer blocks and support the training of deeper neural networks.

**Layer Normalization:** Helps stabilize the activations within the Transformer architecture.

**Dropout:** Randomly disables selected activations during training to reduce overfitting.

**AdamW Optimizer:** Uses adaptive parameter updates and weight decay to support model optimization.

**Gradient Clipping:** Restricts excessively large gradients to help maintain stable training.

**Mini-Batch Training:** Processes sequences in manageable batches rather than processing the entire dataset in a single training operation.

## 9. Observations

During this practical, I gained an understanding of how a Transformer language model learns patterns from text and uses those patterns to generate new content.

Character-level tokenization provided a straightforward method of converting text into numerical representations without requiring a pretrained tokenizer. However, the model needed to learn word structures and sentence patterns from individual characters.

The training and validation loss graphs provided useful information about the model's learning progress. The comparison between the two experiments also demonstrated the importance of training duration when developing language models.

The generated stories demonstrated that the model could learn basic character sequences and produce text from starting prompts. However, the outputs were not always grammatically correct or logically consistent.

These limitations highlight the challenges of developing language models with limited training data, a relatively small architecture and a restricted number of training iterations.

## 10. Learning Outcomes

After completing this practical, I gained practical experience in:

- Loading and exploring text datasets using Hugging Face.
- Understanding character-level tokenization.
- Converting text into numerical representations.
- Preparing input and target sequences for next-character prediction.
- Understanding token and positional embeddings.
- Implementing masked multi-head self-attention.
- Developing Transformer blocks using PyTorch.
- Applying feed-forward networks, residual connections and layer normalization.
- Initializing and training a Transformer language model.
- Using AdamW optimization and gradient clipping.
- Monitoring training and validation loss.
- Evaluating language models using validation loss and perplexity.
- Generating text using a trained language model.
- Comparing models trained for different numbers of epochs.
- Saving trained model parameters and experimental outputs.

## 11. Conclusion

This practical provided hands-on experience in developing a small Transformer language model from scratch using PyTorch and the TinyStories dataset.

The implementation covered the complete language modelling workflow, from dataset loading and preprocessing to Transformer architecture development, training, evaluation and text generation.

The experiments demonstrated how a Transformer learns character-level patterns from text and how different training durations can influence its prediction performance.

Although the generated stories were not always grammatically correct or logically consistent, the practical provided a valuable understanding of the fundamental mechanisms behind Transformer-based language models.

Overall, this laboratory strengthened my practical knowledge of neural network implementation, language model training, model evaluation and text generation.

## 12. Repository Files

The Lab 03 directory contains the implementation notebook and supporting experimental outputs.

```text
Lab-03-Tiny-Transformer/
│
├── README.md
│
├── notebooks/
│   └── LabSheet_3.ipynb
│
└── results/
    ├── tiny_transformer.pth
    ├── vocabulary.json
    ├── model_config.json
    ├── training_history.csv
    ├── experiment_comparison.csv
    ├── generated_story.txt
    ├── generated_prompt_examples.txt
    ├── training_loss.png
    └── experiment_comparison.png
```

The notebook contains the complete implementation, while the results directory stores the trained model, evaluation outputs, generated text and experimental visualisations.