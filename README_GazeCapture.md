# GazeCapture One-Shot Training

This document explains how to use the GazeCapture dataset for one-shot training in the VAE-GAZE project.

## Overview

The GazeCapture one-shot training feature allows you to:

- Randomly select a person from the GazeCapture dataset
- Use multiple samples from that person for more stable fine-tuning
- Control the number of samples used for training
- Specify a particular person ID for targeted training

## Features

### 1. Random Person Selection

- Automatically selects a random person from the dataset
- Ensures the selected person has sufficient samples
- Provides reproducible results with fixed random seed

### 2. Sample Control

- Specify the number of samples to use from the selected person
- Randomly samples from available data if `num_samples` < total samples
- Uses all available samples if `num_samples` > total samples

### 3. Person-Specific Training

- Target a specific person by providing their ID
- Useful for testing or reproducing results
- Ensures consistent training on the same person

## Usage

### Basic Usage

1. **Random person selection:**

```bash
python oneshot_training.py --config configs/finetune/finetune_gazecapture.yaml --use_gazecapture
```

2. **Specify a particular person:**

```bash
python oneshot_training.py --config configs/finetune/finetune_gazecapture.yaml --use_gazecapture --person_id "person_001"
```

3. **Control sample count:**

```bash
python oneshot_training.py --config configs/finetune/finetune_gazecapture.yaml --use_gazecapture --num_samples 50
```

### Configuration File

Create a configuration file `configs/finetune/finetune_gazecapture.yaml`:

```yaml
oneshot_params:
  # Enable GazeCapture dataset usage
  use_gazecapture: true

  # GazeCapture specific parameters
  gazecapture_params:
    hdf_path: "/path/to/GazeCapture.h5"
    person_id: null  # null for random selection
    num_samples: 30  # number of samples per person
    random_person: true  # enable random person selection

  # Training parameters
  num_steps: 500
  learning_rate: 5e-5
  batch_size: 1

  # Fine-tuning strategy
  finetune_strategy: "selective"  # selective, encoder_only, cross_attention_only, minimal

  # Output settings
  save_dir: "./outputs/gazecapture_oneshot"
  checkpoint_path: "./checkpoints/pretrained_model.pth"

# Loss parameters
loss_params:
  l1_loss: 1.0
  mask_l1_loss: 2.0  # Enhanced eye region reconstruction
  vgg_loss: 0.1
```

## Fine-tuning Strategies

### 1. Selective Fine-tuning (Recommended)

```yaml
finetune_strategy: "selective"
```

- Fine-tunes cross-attention layers + output layers + partial upsampling
- Balanced performance and efficiency
- Good for general-purpose fine-tuning

### 2. Encoder-Only Fine-tuning

```yaml
finetune_strategy: "encoder_only"
```

- Fine-tunes only encoder parts (downsampling blocks)
- Good for feature extraction optimization
- Moderate parameter count

### 3. Cross-Attention Only

```yaml
finetune_strategy: "cross_attention_only"
```

- Fine-tunes only cross-attention layers
- Focused on gaze conditioning
- Minimal parameter count

### 4. Minimal Fine-tuning

```yaml
finetune_strategy: "minimal"
```

- Fine-tunes only output layers
- Fastest training
- Good for quick adaptation

## Advanced Features

### 1. Gaze MLP Fine-tuning

You can optionally fine-tune the gaze MLP network:

```yaml
oneshot_params:
  finetune_gaze_mlp: true
  gaze_mlp_layers: ["last"]  # ["last", "all"]
```

### 2. Mask Region Loss

Enhanced eye region reconstruction with mask-specific loss:

```yaml
loss_params:
  mask_l1_loss: 2.0  # Weight for mask region L1 loss
```

### 3. Multiple Sample Training

Using multiple samples provides more stable training:

```yaml
gazecapture_params:
  num_samples: 50  # Use 50 samples per person
```

## Testing

Run the test script to verify functionality:

```bash
python test_gazecapture_oneshot.py
```

This will test:

- Random person selection
- Specific person selection
- Data consistency
- Batch generation

## Output

The training will generate:

- Fine-tuned model checkpoint
- Training loss curve
- Intermediate results (if enabled)
- Person information log

Example output:

```
🎯 Starting VAE-GAZE One-Shot Training with GazeCapture Dataset
📁 HDF5 file: /path/to/GazeCapture.h5
👤 Person ID: Random
📊 Number of samples: 30
🎲 Randomly selected person: person_123
📊 Found 45 total samples for person person_123
🎯 Randomly selected 30 samples from 45 available
✅ Successfully loaded 30 samples for person person_123
```

## Comparison with Single Image Training

| Feature | Single Image | GazeCapture |
|---------|-------------|-------------|
| Data Source | Single image + mask | Multiple samples per person |
| Training Stability | Lower (single sample) | Higher (multiple samples) |
| Person Diversity | Fixed | Random selection |
| Sample Control | N/A | Configurable |
| Use Case | Quick testing | Production training |

## Tips

1. **Sample Count**: Use 20-50 samples for stable training
2. **Learning Rate**: Lower learning rate (5e-5) for multiple samples
3. **Fine-tuning Strategy**: Use "selective" for balanced performance
4. **Mask Loss**: Enable mask_l1_loss for better eye region reconstruction
5. **Random Selection**: Use random person selection for diverse training

## Troubleshooting

### Common Issues

1. **Person not found**: Ensure the person ID exists in the dataset
2. **Insufficient samples**: Reduce `num_samples` or select a different person
3. **Memory issues**: Reduce batch size or number of samples
4. **HDF5 file not found**: Check the file path in configuration

### Debug Mode

Enable detailed logging by setting the log level:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Example Workflow

1. **Setup**: Configure your GazeCapture dataset path
2. **Test**: Run the test script to verify setup
3. **Train**: Start training with random person selection
4. **Evaluate**: Check the generated results and loss curves
5. **Iterate**: Adjust parameters based on results

This feature provides a more robust and flexible approach to one-shot training compared to single image training.
