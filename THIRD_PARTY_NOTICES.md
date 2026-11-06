# Third-party notices

## CTNet

The convolutional–Transformer backbone in `model_ctnet.py` is adapted from CTNet.

> Zhao W, Jiang X, Zhang B, Xiao S, Weng S. CTNet: a convolutional transformer network for
> EEG-based motor imagery classification. Scientific Reports 14, 20237 (2024).
> https://doi.org/10.1038/s41598-024-71118-7
> https://github.com/snailpt/CTNet

MIT License

Copyright (c) 2024 snailpt

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Baseline implementations

`baseline_models.py` re-implements EEGNet (Lawhern et al., 2018), ShallowConvNet and
DeepConvNet (Schirrmeister et al., 2017) and Conformer (Song et al., 2022). The
implementations follow the published architectures; please cite the original papers when
using them.

## SEED dataset

The SEED dataset is **not** included in this repository. It is distributed by the Institute
of Brain Functional Genomics, Shanghai Jiao Tong University, under its own licence
(https://bcmi.sjtu.edu.cn/home/seed/). Users must obtain it themselves.
