import matplotlib.pyplot as plt
import re
import os

# Sahi filenames
log_files = ['finetuning_log.txt', 'finetuning_2log.txt', 'finetuning_3log.txt']
train_acc = []

# Data extraction
for file_path in log_files:
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                # Regex to find accuracy (assuming format is 'Acc: 98.5')
                match = re.search(r'Acc:\s*([\d.]+)', line)
                if match:
                    train_acc.append(float(match.group(1)))

if not train_acc:
    print("Error: No data found! Check if 'Acc:' exists in your logs.")
else:
    # Plotting Histogram
    plt.figure(figsize=(10, 6))
    
    # Bins ko 10-20 ke beech rakhein for better visualization
    n, bins, patches = plt.hist(train_acc, bins=15, color='#4CAF50', edgecolor='black', alpha=0.7)
    
    plt.title('Distribution of Model Accuracy - IIT(BHU) Project', fontsize=14)
    plt.xlabel('Accuracy Range (%)', fontsize=12)
    plt.ylabel('Frequency (Number of Steps)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 100% Accuracy mark
    plt.axvline(100, color='red', linestyle='dashed', linewidth=2, label='SOTA Goal (100%)')
    plt.legend()

    # Save and Show
    plt.savefig('accuracy_histogram.png')
    print("Success! Histogram saved as 'accuracy_histogram.png'")
    plt.show()