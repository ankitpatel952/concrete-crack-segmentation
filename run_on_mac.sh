#!/bin/bash
set -e

PROJECT_DIR="/Users/ankitpatel/.gemini/antigravity/scratch/concrete-crack-segmentation"
cd "$PROJECT_DIR"

echo "=========================================================="
echo " Machine Learning in Crack Detection of Concrete Bridges  "
echo " Setting up local run on Apple Silicon (MacBook Air M4)   "
echo "=========================================================="

# 1. Create Virtual Environment if not present
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    /usr/bin/python3 -m venv venv
fi

# 2. Activate Virtual Environment
source venv/bin/activate

# 3. Upgrade pip and install PyTorch with Apple Silicon MPS acceleration
echo "Installing/verifying required libraries for Apple Silicon..."
pip install --upgrade pip
pip install torch torchvision numpy opencv-python scipy scikit-image matplotlib

# 4. Run the full pipeline on Apple Silicon M4 GPU
echo "Running complete training, metric quantification & figure generation..."
python3 run_local.py

echo ""
echo "=========================================================="
echo " Run complete! Generated figures in $PROJECT_DIR:"
echo "   - fig2_qualitative_results.png / .pdf"
echo "   - fig3_baseline_comparison.png / .pdf"
echo "=========================================================="
