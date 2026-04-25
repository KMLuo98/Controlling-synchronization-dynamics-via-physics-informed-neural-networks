# Hyperparameter Recommendations

This code accompanies the paper *Controlling synchronization dynamics via physics-informed neural networks*.

## Main Point

For the synchronization-control examples, the qualitative effect is usually not very sensitive to small hyperparameter changes. In practice, moderate changes in width, depth, learning rate, and collocation sampling mostly affect convergence speed rather than the final synchronization trend.

## Suggested Ranges

### `N = 10`

- hidden width: `32-48`
- depth: `3-4`
- optimizer steps: `1500-3000`
- collocation samples: `1000-2000`
- control bound: `3-5`

### `N = 50`

- hidden width: `48-80`
- depth: `4-5`
- optimizer steps: `3000-6000`
- collocation samples: `2000-5000`
- control bound: `4-6`

### `N = 100`

- hidden width: `64-128`
- depth: `4-6`
- optimizer steps: `5000-10000`
- collocation samples: `4000-8000`
- control bound: `5-8`

## Practical Advice

- Increase `w_R` first if the target window is not reached.
- Increase `n_f` next if the learned control becomes noisy.
- Increase width and depth only after the loss weights and collocation density are reasonable.
- Use the target interval `[t*, T]` directly instead of multiple fragmented windows unless the paper explicitly studies switching targets.
