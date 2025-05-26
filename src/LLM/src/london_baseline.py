# TODO: [part d]
# Calculate the accuracy of a baseline that simply predicts "London" for every
#   example in the dev set.
# Hint: Make use of existing code.

import argparse
import utils

def main():
    accuracy = 0.0

    # Compute accuracy in the range [0.0, 100.0]
    ### YOUR CODE HERE ###
    filepath = "../birth_dev.tsv"
    with open(filepath, encoding='utf-8') as fin:
        lines = [x.strip().split('\t') for x in fin]
    total = len(lines) # Number of examples
    correct = sum(1 for line in lines if line[1].lower() == "london")
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Total: {total}, Correct: {correct}")
    ### END YOUR CODE ###

    return accuracy

if __name__ == '__main__':
    accuracy = main()
    with open("london_baseline_accuracy.txt", "w", encoding="utf-8") as f:
        f.write(f"{accuracy}\n")
