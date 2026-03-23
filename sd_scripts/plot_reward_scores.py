import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os
import glob


def load_results(path):
    """Load JSONL results from a file or all rank files in a directory."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, '[0-9]*.json')))
        files = [f for f in files if '_scores' not in os.path.basename(f)]
    else:
        files = [path]

    results = []
    for f in files:
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    results.append(json.loads(line))
    return results


def main():
    parser = argparse.ArgumentParser(description='Visualize reward model evaluation results')
    parser.add_argument('--input_files', nargs='+', required=True,
                        help='JSONL files or directories (each = one dataset)')
    parser.add_argument('--labels', nargs='+', default=None,
                        help='Labels for each input (default: filename/dirname)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output image path (e.g. ./plots/result.png)')
    args = parser.parse_args()

    if args.labels and len(args.labels) != len(args.input_files):
        raise ValueError("Number of labels must match number of input files")

    datasets = []
    for i, path in enumerate(args.input_files):
        results = load_results(path)
        label = args.labels[i] if args.labels else os.path.basename(path.rstrip('/'))

        rewards = [r['reward_score'] for r in results]
        yes_scores = [r['yes_score'] for r in results]
        no_scores = [r['no_score'] for r in results]
        # ground truth: 1 if answer is "1>2", else 0
        gt_labels = [1 if r.get('answer', '1>2') == '1>2' else 0 for r in results]

        datasets.append({
            'label': label,
            'rewards': np.array(rewards),
            'yes_scores': np.array(yes_scores),
            'no_scores': np.array(no_scores),
            'gt_labels': np.array(gt_labels),
            'count': len(results),
        })

    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    for ds in datasets:
        tag = f"{ds['label']} (n={ds['count']})"

        if len(np.unique(ds['yes_scores'])) > 1:
            sns.kdeplot(ds['yes_scores'], ax=axes[0], label=tag, fill=True, alpha=0.2, warn_singular=False)

        if len(np.unique(ds['no_scores'])) > 1:
            sns.kdeplot(ds['no_scores'], ax=axes[1], label=tag, fill=True, alpha=0.2, warn_singular=False)

        if len(np.unique(ds['rewards'])) > 1:
            sns.kdeplot(ds['rewards'], ax=axes[2], label=tag, fill=True, alpha=0.2, warn_singular=False)

        # ROC curve: reward_score as prediction, answer as ground truth
        fpr, tpr, _ = roc_curve(ds['gt_labels'], ds['rewards'])
        roc_auc = auc(fpr, tpr)
        axes[3].plot(fpr, tpr, label=f"{ds['label']} (AUC={roc_auc:.4f})")

    axes[0].set_title('Yes Score Distribution')
    axes[0].set_xlabel('Yes Score')
    axes[0].legend()

    axes[1].set_title('No Score Distribution')
    axes[1].set_xlabel('No Score')
    axes[1].legend()

    axes[2].set_title('Reward Distribution')
    axes[2].set_xlabel('Reward Score')
    axes[2].axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='threshold=0.5')
    axes[2].legend()

    axes[3].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[3].set_title('ROC Curve')
    axes[3].set_xlabel('False Positive Rate')
    axes[3].set_ylabel('True Positive Rate')
    axes[3].legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    plt.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved plot to {args.output}")


if __name__ == '__main__':
    main()
