# HSIC-LK package
This package implements the HSIC with learnable kernels for statistical independence tests, as proposed in our paper AISTATS 2024. 

We provide installation instructions and an example code below showing how to use our tests. We also provide a demo notebook.

## Requirements

The requirements for the packages are:
- `python 3.7+`
  - `numpy`
  - `scipy`
  - `cupy`
  - `pytorch`

## HSIC with learnable kernels

**Statistical Independence Test.** Given the tensor $X$ of shape $(n, d_x)$ and $Y$ of shape $(n, d_y)$, our test returns $0$ if the samples $X$ and $Y$ are independence, and $1$ otherwise.
For details, check out the [demo.ipynb]

**HSIC with Fixed Kernels.** We provide the naive implementation for HSIC with fixed kernels. Gaussian kernels as well as Laplace kernels are implemented, and for more choices of kernels, a simple modification of the code can be attempted. 

**Learnable Kernels.** We provide options for learnable kernels. 

```python
# import modules

```
## Examples.

```python
# import modules

```

## Bibtex

```

```

## License

MIT License (see [LICENSE.md](LICENSE.md)).
