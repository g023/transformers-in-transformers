# Hierarchical Transformer for Text Generation using Inter-Layer Transformers - HILT (https://github.com/g023)

This project implements a novel Hierarchical Transformer model for text generation using PyTorch. Unlike standard transformers, this architecture incorporates **Inter-Layer Transformers (ILTs)** that learn patterns and interactions between consecutive transformer layers. This hierarchical approach enables the model to capture more complex dependencies and representations, improving performance on language modeling tasks such as autoregressive text generation.

The model is trained on text data to generate coherent sequences and supports features like temperature-controlled sampling for controlling creativity in generated text.

## Features

### Core Architecture
- **Hierarchical Attention Mechanisms**: Inter-Layer Transformers learn relationships between transformer layers for enhanced representation learning.
- **Custom Transformer Architecture**: Built with standard components like multi-head attention, feed-forward networks, and positional encoding, extended with hierarchical elements.

### Enhanced Data Handling & Tokenization
- **Standard GPT-2 BPE Tokenizer**: Uses pre-trained GPT-2 tokenizer from HuggingFace for robust subword handling.
- **No Training Required**: Leverages battle-tested tokenizer used in GPT-2 and many modern language models.
- **~50k Vocabulary**: Standard vocabulary size provides excellent coverage for English text.

### Advanced Training Features
- **Mixed-Precision Training (AMP)**: Automatic mixed-precision training with `torch.cuda.amp` for faster training and reduced GPU memory usage.
- **Learning Rate Schedulers**: Support for cosine annealing and step decay schedulers to improve convergence.
- **Gradient Accumulation**: Enable larger effective batch sizes on limited GPU memory.
- **Gradient Clipping**: Prevents gradient explosion during training.

### Metrics & Evaluation
- **Perplexity Computation**: Tracks perplexity (exp(loss)) during training and validation as a standard language model metric.
- **Gradient Norm Tracking**: Monitors average gradient norms for debugging and stability analysis.
- **Comprehensive Logging**: Real-time training progress with loss, perplexity, learning rate, and gradient norms.

### Checkpointing & Resuming
- **Enhanced Checkpointing**: Saves complete training state including model, optimizer, scheduler, scaler, epoch, and metrics.
- **Resume Training**: Full support for resuming interrupted training with all state preserved.
- **Best Model Tracking**: Automatically saves best-performing model based on validation loss.

### Generation & Inference
- **Text Generation with Sampling Control**: Supports temperature-based sampling for deterministic or creative outputs.
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs with automatic device detection.
- **Configurable Parameters**: Extensive command-line arguments for tuning model size, training hyperparameters, and generation settings.

## Requirements

- Python 3.7 or higher
- PyTorch (tested with 1.9+)
- HuggingFace Transformers
- NumPy (included with PyTorch)

## Installation

Install the required dependencies using pip:

```bash
pip install torch transformers
```

## Data Preparation

The model expects plain text files for training. `data.txt` contains sample text . You can use any UTF-8 encoded text file as training data.

## Tokenization

The model uses the standard **GPT-2 BPE tokenizer** from HuggingFace. This is automatically loaded when training or doing inference - no tokenizer training required!

To see tokenizer information:

```bash
python build_vocab.py
```

The GPT-2 tokenizer has:
- Vocabulary size: ~50,260 tokens
- Pre-trained on a large corpus
- Robust handling of English text and code
- Special tokens: `<PAD>`, `<BOS>`, `<EOS>`

## Training

Train the model using the `train.py` script:

```bash
python train.py --text_file data.txt --batch_size 32 --seq_len 128 --epochs 10 --lr 5e-4
```

### Training Arguments

**Data & Model:**
- `--text_file`: Path to the training text file (required)
- `--model_dim`: Model dimension (default: 256)
- `--max_tokens`: Maximum number of tokens to process from text file (default: 100000)

**Training Hyperparameters:**
- `--batch_size`: Batch size for training (default: 32)
- `--seq_len`: Maximum sequence length (default: 128)
- `--epochs`: Number of training epochs (default: 10)
- `--lr`: Learning rate (default: 5e-4)
- `--clip_grad`: Gradient clipping value (default: 1.0)

**Optimization & Efficiency:**
- `--gradient_accumulation_steps`: Gradient accumulation steps for larger effective batch sizes (default: 1)
- `--no_amp`: Disable mixed-precision training (enabled by default on CUDA)
- `--scheduler`: Learning rate scheduler type - `cosine`, `step`, or `none` (default: `cosine`)

**Checkpointing:**
- `--resume`: Resume training from checkpoint path (e.g., `best_model.pt`)

### Training Process

The script will:
- Load the standard GPT-2 BPE tokenizer (downloads automatically on first use)
- Train the model with validation, displaying:
  - Training loss and perplexity
  - Validation loss and perplexity
  - Average gradient norms
  - Current learning rate
- Save the best model based on validation loss as `best_model.pt`
- Support resuming from any checkpoint with full training state restoration

### Example Training Outputs

```
Epoch 1/10 Summary:
  Train Loss: 4.2341 | Train Perplexity: 68.75
  Val Loss: 4.1023 | Val Perplexity: 60.42
  Avg Grad Norm: 2.3456 | Learning Rate: 0.000500
✓ Saved best model with val loss: 4.1023, perplexity: 60.42
```

## Inference

Generate text using the trained model:

```bash
python inference.py --model_path best_model.pt --prompt "Hello world" --max_len 100 --temperature 1.0
```

### Inference Arguments

- `--model_path`: Path to the trained model file (required, e.g., `best_model.pt`)
- `--prompt`: Starting text prompt (default: "The quick brown")
- `--max_len`: Maximum length of generated text (default: 50)
- `--temperature`: Sampling temperature (default: 1.0; lower values for more deterministic output, higher for more creative)
- `--model_dim`: Model dimension, must match training (default: 256)

### Temperature Guide

- **0.5-0.7**: More focused, coherent, and deterministic output
- **0.8-1.0**: Balanced creativity and coherence (recommended)
- **1.1-1.5**: More creative and diverse, may be less coherent
- **> 1.5**: Very random and experimental output

## Model Architecture

The Hierarchical Transformer extends the standard Transformer architecture with hierarchical learning mechanisms. It consists of the following key components:

### Core Components

- **PositionalEncoding**: Adds sinusoidal positional embeddings to input token embeddings to encode sequence order.
- **MultiHeadAttention**: Implements self-attention with multiple heads, allowing the model to attend to different parts of the input simultaneously.
- **FeedForward**: A position-wise feed-forward network with GELU activation and dropout for non-linear transformations.
- **TransformerLayer**: A standard transformer block combining multi-head attention, feed-forward, and layer normalization with residual connections.

### Hierarchical Elements

- **InterLayerTransformer (ILT)**: A small transformer that operates on the outputs of consecutive main transformer layers. It treats the layer outputs as a sequence and learns inter-layer patterns, providing modulation signals to enhance the main layers' representations.
- **HierarchicalTransformer**: The main model class that stacks multiple `TransformerLayer`s, interspersed with `ILT`s between them. Each ILT takes a window of layer outputs (typically 2 consecutive layers) and applies attention to learn hierarchical dependencies, then modulates the next layer's output.

### Inter-Layer Transformers (ILTs)

The Inter-Layer Transformers (ILTs) are a key innovation in this hierarchical architecture. Unlike traditional transformers that focus solely on token-level dependencies within a sequence, ILTs introduce a meta-level of learning by modeling interactions between the layers themselves.

Each ILT takes the outputs from two consecutive transformer layers and treats them as a sequence of "layer representations." It then applies self-attention to learn how these layer outputs relate to each other, capturing patterns such as how early layers process basic syntax while later layers handle semantics.

The learned attention patterns generate a modulation signal, which is added to the output of the next layer, enhancing its representations. This allows the model to dynamically adjust layer behaviors based on the input, leading to more adaptive and context-aware processing.

ILTs are lightweight transformers themselves, with fewer parameters than the main layers, ensuring efficiency. They are trained end-to-end with the rest of the model, learning to optimize the hierarchical flow of information.

This approach enables the model to capture not just intra-layer dependencies but inter-layer synergies, potentially leading to better generalization, reduced overfitting, and improved performance on complex language tasks.

The ILTs are active and functional during both training and inference, integrated into the model's forward pass to learn and apply hierarchical patterns in real-time.

### Training Assistance with ILTs: Curriculum Learning

To further enhance the ILTs' role in training, the model implements **curriculum learning** for ILT modulation. This technique gradually increases the influence of ILT modulation signals during training, allowing the main transformer layers to first learn fundamental token-level patterns before incorporating complex inter-layer hierarchical refinements.

#### How It Works
- **Modulation Scale**: Each ILT generates a modulation signal that is added to the output of the next main layer. The strength of this modulation is controlled by a `modulation_scale` parameter (default: 0.1).
- **Curriculum Schedule**: During training, `modulation_scale` starts low (0.01) in early epochs and linearly increases to the full value (0.1) by the final epoch. This is computed as:
  ```
  modulation_scale = MODULATION_SCALE_START + (current_epoch / total_epochs) * (MODULATION_SCALE_END - MODULATION_SCALE_START)
  ```
- **Benefits**:
  - **Stabilizes Training**: Early epochs focus on basic layer learning without heavy ILT interference, reducing instability.
  - **Improves Convergence**: Gradual introduction of hierarchical patterns helps the model build knowledge incrementally, potentially leading to better final performance.
  - **Reduces Overfitting**: By allowing layers to establish strong foundations first, the model may generalize better to unseen data.
- **Implementation**: The `modulation_scale` is passed as a parameter to `HierarchicalTransformer.forward()`. In `train.py`, it's calculated per epoch and applied during training. Validation uses the full scale (0.1) for consistency.

This curriculum approach is simple, requires no additional components, and leverages the existing ILT architecture to make training more effective and robust.

### ILT Hyperparameters

The `InterLayerTransformer` (ILT) components have several hyperparameters that control their architecture and behavior. These are defined as global constants at the top of `model.py` for easy tuning. Below is a breakdown of each hyperparameter, including its default value, what it controls, and how changing it might affect the model.

1. **`ILT_DIM` (default: 128)**  
   - **What it does**: The internal dimensionality of the ILT's representations, projected from the main model's `d_model` (default 256).  
   - **Impact**: Lower values reduce parameters and computation but may limit pattern capture. Higher values increase expressiveness but add cost.

2. **`NHEAD_ILT` (default: 2)**  
   - **What it does**: Number of attention heads in each ILT transformer layer.  
   - **Impact**: Fewer heads simplify the model; more heads allow diverse attention patterns but increase computation. Must divide `ILT_DIM` evenly.

3. **`N_LAYERS_ILT` (default: 2)**  
   - **What it does**: Number of stacked transformer layers in the ILT.  
   - **Impact**: Fewer layers reduce depth and computation; more layers allow complex dependencies but add parameters.

4. **`ILT_DROPOUT` (default: 0.1)**  
   - **What it does**: Dropout probability specifically for ILT layers.  
   - **Impact**: Lower values reduce regularization; higher values help generalization. Separate from main model dropout for independent tuning.

5. **`MODULATION_SCALE_START` (default: 0.01)**  
   - **What it does**: Initial modulation scale for curriculum learning, controlling ILT influence at training start.  
   - **Impact**: Lower values prioritize main layer learning early; higher values introduce ILT effects sooner.

6. **`MODULATION_SCALE_END` (default: 0.1)**  
   - **What it does**: Final modulation scale for curriculum learning, setting full ILT influence by training end.  
   - **Impact**: Determines maximum ILT impact; tune based on desired hierarchical strength.

**Additional Notes**: The ILT uses `d_ff = ILT_DIM * 4` internally. Tune these global constants in `model.py` based on dataset and monitor metrics like perplexity.

### Key Features

- **Causal Masking**: Ensures autoregressive generation by masking future tokens during training and inference.
- **Special Tokens**: Includes `<PAD>` (padding), `<BOS>` (beginning of sequence), and `<EOS>` (end of sequence) for proper sequence handling.
- **Standard GPT-2 Tokenizer**: Uses pre-trained tokenizer with ~50k vocabulary, no custom training required.
- **Embedding and Language Modeling Head**: Embeddings for input and linear head for next-token prediction.

### Global Hyperparameters

The model uses several global constants defined at the top of `model.py` for easy configuration and tuning:

- **`DROPOUT` (default: 0.1)**: Dropout probability applied throughout the main model components (attention, feed-forward, etc.) for regularization.
- **`ILT_DROPOUT` (default: 0.1)**: Separate dropout probability for Inter-Layer Transformers, allowing independent tuning of ILT regularization.
- **`ILT_DIM` (default: 128)**: Internal dimensionality of ILT representations.
- **`NHEAD_ILT` (default: 2)**: Number of attention heads in ILT layers.
- **`N_LAYERS_ILT` (default: 2)**: Number of transformer layers in each ILT.
- **`MODULATION_SCALE_START` (default: 0.01)**: Initial ILT modulation scale for curriculum learning.
- **`MODULATION_SCALE_END` (default: 0.1)**: Final ILT modulation scale for curriculum learning.

These constants can be modified directly in `model.py` to experiment with different configurations without changing the code structure.

## Files

- `train.py`: Main training script with mixed-precision, BPE tokenization, metrics tracking, and enhanced checkpointing.
- `inference.py`: Script for generating text from a trained model using standard GPT-2 tokenizer.
- `model.py`: Defines the HierarchicalTransformer architecture and its components.
- `tokenizer_utils.py`: Wrapper for standard GPT-2 BPE tokenizer from HuggingFace.
- `build_vocab.py`: Script to display information about the standard GPT-2 tokenizer.
- `data.txt`: Sample training data (Shakespeare excerpts).
- `best_model.pt`: Best-performing model checkpoint with full training state.

## Checkpoint Structure

Enhanced checkpoints now contain complete training state:

```python
{
    'epoch': int,                      # Current epoch number
    'model_state_dict': dict,          # Model weights
    'optimizer_state_dict': dict,      # Optimizer state
    'scheduler_state_dict': dict,      # LR scheduler state (if used)
    'scaler_state_dict': dict,         # AMP scaler state (if used)
    'best_loss': float,                # Best validation loss so far
    'train_loss': float,               # Training loss from this epoch
    'val_loss': float,                 # Validation loss from this epoch
    'perplexity': float,               # Validation perplexity
    'config': dict                     # Training configuration
}
```

This allows for seamless resumption of training with all state preserved.

## Usage Tips

### Training Optimization
- **Mixed-Precision Training**: Enabled by default on CUDA devices, use `--no_amp` to disable if encountering issues.
- **Gradient Accumulation**: Use `--gradient_accumulation_steps 4` or higher to simulate larger batch sizes on limited GPU memory.
- **Learning Rate Scheduling**: Cosine annealing (default) often works best; try `--scheduler step` for step decay.
- **Batch Size and Sequence Length**: Adjust based on GPU memory; start with smaller values and increase gradually.

### Model Architecture
- **Model Size**: Increase `model_dim` and number of layers in `model.py` for better performance, but monitor GPU memory usage.
- **Vocabulary Size**: Larger vocabularies (15k-30k) can improve quality but increase memory usage.

### Data & Training
- **Data Quality**: Use diverse, high-quality text data similar to your target domain for optimal results.
- **Training Duration**: Monitor perplexity; training can continue until perplexity plateaus.
- **Validation Split**: Current 90/10 train/val split can be adjusted in `train.py`.

### Generation
- **Temperature Tuning**: Values < 1.0 produce more focused, predictable text; > 1.0 encourages creativity and variety.
- **Prompt Engineering**: Longer, more specific prompts generally produce better results.

### Checkpointing
- **Resume Training**: Use `--resume best_model.pt` to continue from the best checkpoint.

## Performance Metrics

The training process tracks several key metrics:

- **Loss**: Cross-entropy loss on training and validation sets
- **Perplexity**: exp(loss), measures how well the model predicts; lower is better
- **Gradient Norm**: Average gradient magnitude; useful for detecting training instability
- **Learning Rate**: Current learning rate from scheduler

## Troubleshooting

- **CUDA Availability**: Ensure PyTorch detects your GPU with `torch.cuda.is_available()`. Install CUDA toolkit if needed.
- **Out of Memory**: 
  - Reduce `--batch_size` or `--seq_len`
  - Enable gradient accumulation with `--gradient_accumulation_steps`
  - Reduce `--model_dim`
  - Disable mixed-precision with `--no_amp` (not recommended)
- **Inference Errors**: Verify that `tokenizer.json` and the model file exist and were generated from compatible training runs.
- **Poor Generation Quality**: 
  - Train for more epochs
  - Use lower temperature (0.7-0.8)
  - Increase model dimensions
  - Use more/better training data
- **Training Instability**: 
  - Check gradient norms (should be < 10)
  - Reduce learning rate
  - Increase gradient clipping (`--clip_grad`)
- **Tokenizer Issues**: Delete `tokenizer.json` and retrain to ensure compatibility with current data.</content>


```
# last training session on a text file 
python train.py --text_file stories_1m.txt --resume best_model.pt --epochs 200 --batch_size 90
python inference.py --model_path best_model.pt --prompt "There was" --use_beam_search --beam_width 10
python inference.py --model_path best_model.pt --prompt "There was" --temperature 0 --use_cache
python inference.py --model_path best_model.pt --prompt "There was" --top_p 0.45
python inference.py --model_path best_model.pt --prompt "There was" --temperature 0.1
python inference.py --model_path best_model.pt --prompt "There was" --temperature 0.45 --use_cache

--

python train.py --text_file stories_1M_paras.txt --resume best_model.pt --epochs 25 --batch_size 18 --model_dim 256 --lr 2e-4 --seq_len 512 --gradient_accumulation_steps 512


```