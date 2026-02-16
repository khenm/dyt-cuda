import argparse
import sys
import os
import pandas as pd
import matplotlib.pyplot as platform
import seaborn as sns 
import logging 
from benchmarking.engine import run_benchmark, plot_performance

sys.path.append(os.path.join(os.getcwd(), 'benchmarking'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('PRETRAIN')

def main():
    parser = argparse.ArgumentParser(description="DyT Wav2Vec2 Benchmarking & Pre-training")
    
    parser.add_argument('--backend', type=str, default='all', 
                        choices=['all', 'baseline', 'torch', 'triton', 'cuda'],
                        help="Select specific backend or 'all' to run comparison.")
    
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size per step.")
    parser.add_argument('--max-secs', type=int, default=5, help="Audio duration per sample in seconds.")
    parser.add_argument('--steps', type=int, default=50, help="Number of steps to benchmark.")
    parser.add_argument('--output', type=str, default='results/dyt_benchmark_results.png', help="Path to save the results plot.")

    args = parser.parse_args()

    if args.backend == 'all':
        backends = ['baseline', 'torch', 'triton', 'cuda']
    else:
        backends = [args.backend]

    logger.info(f"Starting benchmarking... Config: Batch={args.batch_size}, Secs={args.max_secs}, Steps={args.steps}")
    results = []

    for backend in backends:
        try: 
            logger.info(f"\n{"="*10} Running Backend: {backend.upper()} {"="*10}")
            metrics = run_benchmark(
                backend=backend,
                batch_size=args.batch_size,
                seq_secs=args.max_secs,
                steps=args.steps
            )

            display_names = {
                'baseline': 'LayerNorm',
                'torch': 'DyT (PyTorch)',
                'triton': 'DyT (Triton)',
                'cuda': 'DyT (CUDA/Ours)'
            }

            if metrics:
                if "Backend" not in metrics:
                    metrics["Backend"] = display_names.get(backend, backend)
                else:
                    metrics["Backend"] = display_names.get(metrics["Backend"], metrics["Backend"])

                results.append(metrics)

        except Exception as e:
            logger.error(f"FAIL: Backend {backend} crashed: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        logger.warning("No results collected. Exiting.")
        return 

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    if len(results) > 0:
        logger.info("Generating plot...")
        try: 
            metric_col = 'Avg Loss' if 'Avg Loss' in df.columns else df.columns[-1]
            plot_performance(df, metric_name=metric_col, save_path=args.output)
        except Exception as e:
            logger.error(f"Plotting failed: {e}")

if __name__ == '__main__':
    main()