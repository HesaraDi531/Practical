# Lab Sheet 4 – Fine-Tuning an LLM using LoRA

**Module:** PUSL 3205 – Large Language Models and Generative AI  
**Practical:** Lab Sheet 4 – Fine-Tuning an LLM using LoRA  
**Domain:** Computer-shop question answering  
**Model:** `Qwen/Qwen2.5-0.5B-Instruct`

## 1. Introduction

This practical demonstrates how to fine-tune a small instruction-tuned Large Language Model (LLM) for a specific domain using Low-Rank Adaptation (LoRA). A computer-shop question-answering dataset was prepared and used to adapt the pre-trained model to provide concise answers about computer products, components, accessories and basic troubleshooting.

The original base model and the fine-tuned model were evaluated using the same eight unseen questions. Their responses were saved and compared to observe the effect of domain-specific fine-tuning.

## 2. Objectives

- Understand the process of fine-tuning a pre-trained instruction-tuned language model.
- Prepare a domain-specific question-and-answer dataset.
- Apply Parameter-Efficient Fine-Tuning (PEFT) using LoRA.
- Train the model using supervised fine-tuning.
- Compare the responses of the original and fine-tuned models using unseen questions.
- Save the trained LoRA adapter and export the experimental results.

## 3. Tools and Technologies

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| Google Colab | Notebook execution and GPU environment |
| PyTorch | Deep-learning framework |
| Hugging Face Transformers | Loading the tokenizer and pre-trained language model |
| Hugging Face Datasets | Preparing the training dataset |
| PEFT | Applying LoRA adapters |
| Hugging Face Trainer | Managing the supervised fine-tuning process |
| Pandas | Organising and exporting experimental results |
| Matplotlib | Visualising training loss |
| OpenPyXL | Exporting results to an Excel workbook |

## 4. Model and Hardware Configuration

The experiment used the `Qwen/Qwen2.5-0.5B-Instruct` instruction-tuned model.

The notebook was executed in Google Colab with the following reported environment:

| Configuration | Recorded result |
|---|---|
| GPU | NVIDIA Tesla T4 |
| GPU memory | 14.56 GB |
| CUDA availability | Available |
| CUDA version | 13.0 |
| PyTorch | 2.14.1+cu130 |
| GPU compute capability | 7.5 |
| Training precision | FP16 |
| BF16 | Disabled |

FP16 was used for CUDA training and BF16 was disabled to avoid the BF16 matrix-operation compatibility error encountered with the Tesla T4 GPU. The notebook used the Hugging Face `Trainer` with PEFT LoRA for the training workflow.

## 5. Dataset Preparation

A computer-shop question-and-answer dataset containing **40 training examples** was created. The examples cover computer products, hardware components, storage devices, memory, processors, graphics cards, monitors, accessories and basic computer-related topics.

The examples were formatted as conversational instructions containing a system message, a user question and an assistant answer. During tokenization, prompt tokens were masked from the loss calculation so that training focused on the assistant's answer tokens.

### Evaluation dataset

A separate set of **8 unseen questions** was prepared. These questions were not copied from the training questions. The same eight questions were submitted to both the original model and the fine-tuned model to support a consistent qualitative comparison.

## 6. Fine-Tuning Method

LoRA was applied through the PEFT library. Instead of updating every parameter in the original language model, the training process updated a smaller set of adapter parameters.

The recorded configuration was:

| Parameter | Value |
|---|---:|
| LoRA rank (`r`) | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| Bias | None |
| Epochs | 3 |
| Batch size per device | 1 |
| Gradient accumulation steps | 4 |
| Learning rate | 0.0002 |
| Maximum sequence length | 512 |
| Training precision | FP16 |
| BF16 | Disabled |

The LoRA target modules were `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj` and `down_proj`.

## 7. Experimental Results

### Training results

The notebook completed the three-epoch fine-tuning process successfully.

| Metric | Recorded result |
|---|---:|
| Training examples | 40 |
| Number of epochs | 3 |
| Training runtime | 46.71 seconds |
| Final training loss | 1.3604 |
| Trainable parameters | 4,399,104 |
| Total model parameters | 498,431,872 |
| Trainable parameter percentage | 0.8826% |

The training-loss graph was generated and saved as `training_loss.png`.

### Base model and fine-tuned model comparison

Both models were tested with the same eight unseen questions. The fine-tuned model generally returned shorter and more direct answers focused on the computer-shop domain.

Some observations from the comparison:

- For the SSD question, the fine-tuned model gave a concise recommendation that an SSD is suitable for faster boot times.
- For the CPU cooler question, the fine-tuned model provided a direct explanation of its role in preventing processor overheating.
- For the RAM compatibility question, the fine-tuned model returned a shorter answer focused on compatibility and performance.
- For the Wi-Fi troubleshooting question, the fine-tuned model suggested checking network settings, Wi-Fi connectivity and possible interference.
- For the 650W power-supply question, the fine-tuned model gave a more reasonable general response than the base model's incorrect categorical claim. However, actual PSU suitability depends on the complete PC configuration.

The comparison is qualitative. No automatic accuracy score or human evaluation score was calculated in this experiment.

## 8. Output Files

The notebook generated and exported the following results:

| File | Description |
|---|---|
| `baseline_results.csv` | Answers generated by the original base model |
| `fine_tuned_results.csv` | Answers generated by the fine-tuned model |
| `model_comparison.csv` | Side-by-side comparison of both models |
| `Lab4_Complete_Results.xlsx` | Excel workbook containing the dataset, model answers, comparison and training logs |
| `training_loss.png` | Visualisation of the training loss |
| `adapter_config.json` | Configuration of the saved LoRA adapter |
| `adapter_model.safetensors` | Trained LoRA adapter weights |
| `tokenizer.json` and related tokenizer files | Tokenizer files saved with the adapter |

The notebook also created `Lab4_Complete_Outputs.zip`, containing the exported results and saved adapter.

## 9. Issues and Compatibility Handling

During implementation, a BF16 matrix-operation error occurred because the Tesla T4 GPU has compute capability 7.5, which does not support the relevant BF16 operation used by the earlier training workflow.

The final notebook addressed this by:

- Using FP16 for CUDA training.
- Explicitly disabling BF16.
- Using Hugging Face `Trainer` with PEFT LoRA rather than the TRL chunked-log-probability training path associated with the reported error.
- Releasing the base model from GPU memory before loading the model for fine-tuning.

The final notebook completed training, generated the fine-tuned answers, saved the adapter and exported the result files. Some package compatibility warnings appeared during installation, but they did not prevent the recorded experiment from completing.

## 10. Learning Outcomes

Through this practical, I gained experience in loading an instruction-tuned language model, preparing a domain-specific conversational dataset, applying LoRA through PEFT and carrying out supervised fine-tuning. I also learned how to configure training precision according to GPU compatibility, evaluate the base and fine-tuned models using unseen questions, inspect training loss and save the trained adapter for later use.

## 11. Limitations and Future Improvements

The experiment used a relatively small dataset of 40 examples and only eight unseen evaluation questions. Therefore, the results demonstrate the fine-tuning workflow but do not establish that the fine-tuned model is consistently more accurate across all computer-shop questions.

Future improvements could include expanding the dataset with more varied and carefully verified examples, increasing the size of the unseen evaluation set, and conducting a structured human evaluation of correctness, relevance and clarity. Additional testing should also check whether the model avoids unsupported or overly general technical advice.

## 12. Conclusion

This practical successfully demonstrated domain-specific supervised fine-tuning of `Qwen/Qwen2.5-0.5B-Instruct` using PEFT LoRA. The model was trained on 40 computer-shop question-and-answer examples for three epochs, achieving a recorded final training loss of 1.3604. The base and fine-tuned models were compared using eight unseen questions, and the trained adapter, model responses, comparison data, training-loss graph and Excel results were saved.

The experiment provided practical experience in parameter-efficient LLM fine-tuning and highlighted the importance of selecting a training precision compatible with the available GPU hardware.
