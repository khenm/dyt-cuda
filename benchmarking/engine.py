import os
import torch
from datasets import load_dataset
import logging
import time
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Config, Wav2Vec2ForPreTraining
from .dyt_wav2vec import patch_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_librispeech_batch(batch_size=4, max_secs=10):
    try: 
        dataset = load_dataset("librispeech_asr", "clean", split="train.100", streaming=True)
        batch = []
        target_len = 16000 * max_secs # 16kHz

        iter_data = iter(dataset)
        while len(batch) < batch_size:
            sample = next(iter_data)
            audio = sample['audio']['array']

            audio_tensor = torch.tensor(audio).float()
            if audio_tensor.size(0) > target_len:
                audio_tensor = audio_tensor[:target_len]
            else:
                padding = target_len - audio_tensor.size(0)
                audio_tensor = torch.cat([audio_tensor, torch.zeros(padding)])
            
            batch.append(audio_tensor)
        
        return torch.stack(batch)
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        return torch.randn(batch_size, 16000 * max_secs)

def run_benchmark(backend='torch', batch_size=4, seq_secs=5, steps=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Wav2Vec2Config(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        vocab_size=32
    )

    model = Wav2Vec2ForPreTraining(config)

    if backend in ['cuda', 'triton', 'torch']:
        model = patch_model(model, backend=backend)
    else:
        logger.info("Using default torch backend")
    
    model.to(device)
    model.train()

    inputs = get_librispeech_batch(batch_size, max_secs=seq_secs).to(device)

    # wav2vec2 down by 320
    feature_len = model.wav2vec2._get_feat_extract_output_lengths(inputs.shape[1])
    mask_time_indices = torch.randint(0, 2, (inputs.shape[0], feature_len), device=device, dtype=torch.bool)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # warmup
    logger.info("Warm up epochs...")
    for _ in range(5):
        loss = model(inputs, mask_time_indices=mask_time_indices).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    torch.cuda.synchronize()

    logger.info(f"Benchmark {backend.upper()} for {steps} steps...")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_loss = 0.0

    start_event.record()
    for _ in range(steps):
        outputs = model(inputs, mask_time_indices=mask_time_indices)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    
    end_event.record()
    torch.cuda.synchronize()

    elapsed_ms = start_event.elapsed_time(end_event)

    # stats
    seconds = elapsed_ms / 1000
    samples_processed = batch_size * steps
    audio_seconds_processed = samples_processed * seq_secs
    throughput = audio_seconds_processed / seconds
    avg_loss = total_loss / steps

    logger.info(f"Benchmark {benchmark.upper()} completed in {seconds:.2f} seconds")
    logger.info(f"Throughput: {throughput:.2f} audio seconds per second")

    max_mem = torch.cuda.max_memory_allocated() / 1e9
    logger.info(f"Peak VRAM: {max_mem:.2f} GB")
    logger.info("="*40)

    return {
        "Backend": backend,
        "Throughput (GB/s)": throughput,
        "Peak VRAM (GB)": max_mem,
        "Avg Loss": avg_loss
    }

def plot_performance(df, metric_name='Avg Loss', save_path='./results/dyt_benchmark_results.png'):
    sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
    unique_backends = df['Backend'].unique()
    palette = {}

    for backend in unique_backends:
        name = backend.lower()
        if 'cuda' in name:
            palette[backend] = '#FF1F5B'
        elif 'triton' in name:
            palette[backend] = '#00CD6C'
        elif 'torch' in name:
            palette[backend] = '#FFC61E'
        else:
            palette[backend] = '#A0B1BA'
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), constrained_layout=True)

    def add_labels(ax, is_loss=False):
        for p in ax.patches:
            height = p.get_height()
            if height > 0:
                ax.annotate(
                    f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0, 5), textcoords='offset points',
                    fontsize=12, fontweight='bold', color='black'
                )

        sns.barplot(data=df, x='Backend', y='Throughput (GB/s)', ax=axes[0], palette=palette, hue='Backend', legend=False)
        axes[0].set_title("Training Speed\n(↑)", fontweight='bold', pad=15)
        axes[0].set_ylabel("Audio Seconds / Sec")
        axes[0].set_xlabel("")
        add_labels(axes[0])

        sns.barplot(data=df, x='Backend', y='Peak VRAM (GB)', ax=axes[1], palette=palette, hue='Backend', legend=False)
        axes[1].set_title("Peak Memory Footprint\n(↓)", fontweight='bold', pad=15)
        axes[1].set_ylabel("Peak VRAM (GB)")
        axes[1].set_xlabel("")
        add_labels(axes[1])

        is_loss = 'loss' in metric_name.lower()
        direction = "(↓)" if is_loss else "(↑)"
    
        sns.barplot(data=df, x='Backend', y=metric_name, ax=axes[2], palette=palette, hue='Backend', legend=False)
        axes[2].set_title(f"Model Stability: {metric_name}\n{direction}", fontweight='bold', pad=15)
        axes[2].set_ylabel(metric_name)
        axes[2].set_xlabel("")

        if is_loss:
            min_val = df[metric_name].min()
            max_val = df[metric_name].max()
            margin = (max_val - min_val) * 2
            if margin == 0: margin = 0.1 * max_val

        add_labels(axes[2], is_loss=is_loss)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")