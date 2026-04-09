#!/bin/bash
# Run all evaluations for Audio-Visual Speech Recognition Models

echo "=========================================="
echo "Running All Evaluations"
echo "=========================================="

# Audio-Only Model
echo ""
echo "1/6: Audio-Only - Clean Environment"
python3 evaluate.py --mode audio --environment clean --batch_size 16

echo ""
echo "2/6: Audio-Only - Noisy Environment"
python3 evaluate.py --mode audio --environment noisy --batch_size 16

# Video-Only Model
echo ""
echo "3/6: Video-Only - Clean Environment"
python3 evaluate.py --mode video --environment clean --batch_size 16

echo ""
echo "4/6: Video-Only - Noisy Environment"
python3 evaluate.py --mode video --environment noisy --batch_size 16

# Fusion Model
echo ""
echo "5/6: Fusion - Clean Environment"
python3 evaluate.py --mode fusion --environment clean --batch_size 16

echo ""
echo "6/6: Fusion - Noisy Environment"
python3 evaluate.py --mode fusion --environment noisy --batch_size 16

echo ""
echo "=========================================="
echo "All evaluations complete!"
echo "=========================================="

# Generate comparison table
echo ""
echo "Generating comparison table..."
python3 generate_comparison_table.py

echo ""
echo "✅ Done! Check results/ directory for all outputs."
