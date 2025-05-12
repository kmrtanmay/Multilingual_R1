
#!/bin/bash

# Make all scripts executable
chmod +x launch_distributed_training.sh

# Create log directory
mkdir -p logs

# Get current timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

# Run the training script and capture logs
echo "Starting training run at $(date)"
echo "Logs will be saved to ${LOG_FILE}"

# Run the training with output captured to log file
./launch_distributed_training.sh 2>&1 | tee ${LOG_FILE}

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "Training completed successfully!"
else
    echo "Training failed with exit code ${PIPESTATUS[0]}"
    echo "Check ${LOG_FILE} for details"
fi