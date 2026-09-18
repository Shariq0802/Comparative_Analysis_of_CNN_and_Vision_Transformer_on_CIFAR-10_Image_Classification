# Comparative Analysis of CNN and Vision Transformer on CIFAR-10

Compares CNN and Vision Transformer (ViT) architectures for
image classification, using 3 iterative versions of each model.

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 32x32 colour images, 10 classes
(50,000 train / 10,000 test), loaded via `torchvision`.

## What was done

- **CNN v1 -> v3:** baseline CNN, then added BatchNorm/Dropout/augmentation, then residual
  (ResNet-style) blocks with a cosine LR schedule.
- **ViT v1 -> v3:** baseline ViT, then larger patch size and wider embeddings, then greater depth
  and cosine LR scheduling.
- Each iteration is trained and evaluated on CIFAR-10, with accuracy/loss curves and confusion
  matrices, and compared against the previous version.
- A final comparison covers accuracy, convergence speed, parameter count, training time, and
  generalisation.

## Key results

| Model  | Test Accuracy | Parameters | Training Time |
|--------|--------------|------------|----------------|
| CNN v1 | 75.96%       | 620K       | -              |
| CNN v2 | 82.60%       | 621K       | -              |
| CNN v3 | 92.51%       | 1.92M      | 1385s          |
| ViT v1 | 70.58%       | 208K       | -              |
| ViT v2 | 69.50%       | 822K       | -              |
| ViT v3 | 68.93%       | 1.22M      | 1422s          |

## Conclusion

CNNs outperformed ViTs by a wide margin (92.51% vs 68.93%) on this small dataset. CNNs converge
faster and generalise well once regularised; ViTs plateaued around 69-71% regardless of
architectural changes, because they lack CNNs' built-in locality/translation-invariance biases and
CIFAR-10 (50,000 images, 32x32 resolution) is too small for ViTs to learn those patterns from
scratch. ViTs are expected to close the gap with larger datasets or pretrained weights.

## Contents

- `Task_2.ipynb` - full notebook: data prep, all 6 model iterations, training,
  evaluation, and comparative analysis.

## Note
- There is already code to take in CIFAR 10 data set directly. Anyways, the link is at the top in read me.
