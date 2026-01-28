#!/bin/bash
# Helper script to run experiments

if [ $# -eq 0 ]; then
    echo "Usage: ./run_experiment.sh <experiment_name>"
    echo ""
    echo "Available experiments:"
    echo "  analysis              - ID3 training size analysis"
    echo "  compare               - ID3 vs C4.5 comparison"
    echo "  evolution             - ID3 → C4.5 → XGBoost"
    echo "  modern                - All 4 algorithms"
    echo "  large_dataset         - Production-scale datasets"
    exit 1
fi

source venv/bin/activate

case "$1" in
    analysis)
        python src/experiments/analysis.py
        ;;
    compare)
        python src/experiments/compare_algorithms.py
        ;;
    evolution)
        python src/experiments/evolution_comparison.py
        ;;
    modern)
        python src/experiments/modern_comparison.py
        ;;
    large_dataset)
        python src/experiments/large_dataset_comparison.py
        ;;
    *)
        echo "Unknown experiment: $1"
        exit 1
        ;;
esac
