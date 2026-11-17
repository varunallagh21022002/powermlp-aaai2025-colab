# PowerMLP: 20 viva questions with model answers

Reference: Qiu et al., *PowerMLP: An Efficient Version of KAN*, AAAI 2025. DOI: https://doi.org/10.1609/aaai.v39i19.34210

These answers distinguish the paper's claims from this project's reduced experiment.

## 1. What problem does this paper address?
KANs learn spline functions on edges, but spline evaluation adds computation. The paper proposes PowerMLP to retain strong function representation while using simpler matrix operations and powered ReLU activations.

## 2. How do MLP, KAN, and PowerMLP differ?
An MLP combines affine transformations with fixed node activations. A KAN learns univariate functions on edges and sums their outputs. PowerMLP combines a powered-ReLU affine path with a learned projection of a fixed SiLU basis function.

## 3. What is the exact PowerMLP hidden-layer equation?
The equation is h_next = alpha × SiLU(h) + ReLU(W × h + b)^k. W, b, and alpha are trainable. SiLU and powered ReLU have no trainable parameters in this implementation. The last layer is only W_out × h + b_out.

## 4. What does the power k mean?
It is a positive integer applied after ReLU: sigma_k(z) = max(0,z)^k. We use k=3, matching the main cubic setting in the paper. k changes the nonlinear function family and can also affect gradient magnitudes.

## 5. Why does the model contain a basis branch?
The branch alpha × SiLU(h) provides another path for information and gradients. The paper's ablations associate it with more stable training, particularly as depth increases. This does not guarantee that every run or configuration improves.

## 6. Is the basis branch the same as an ordinary residual identity connection?
No. It first applies SiLU elementwise and then uses a trainable, bias-free projection. The projection also handles different input and output widths. An ordinary identity residual would simply add the input when dimensions agree.

## 7. What is SiLU, and what is its derivative?
SiLU(x) = x × sigmoid(x). Its derivative is sigmoid(x) + x × sigmoid(x) × (1 - sigmoid(x)). Both terms matter when differentiating the basis branch. We use PyTorch autograd and verify gradients numerically.

## 8. Why can powered ReLU represent spline functions?
The paper connects B-splines to linear combinations of shifted truncated power functions. A shifted powered ReLU is such a function. This connection supports a nonrecursive representation, although the constructive network may need different width or depth.

## 9. What does the theoretical relationship between KAN and PowerMLP establish?
The paper gives representation results with specified network-size transformations. On bounded domains, the model families can represent each other under those constructions. The global function-space result differs. It does not mean equally sized trained networks always have identical accuracy.

## 10. Why is the final layer linear?
The paper defines the output layer as affine. This allows unrestricted real-valued regression outputs, including negative values. Applying ReLU at the end would change the specified architecture and restrict predictions.

## 11. What exactly did you implement?
I implemented the PowerMLP equation in PyTorch, small and large model shapes, a ReLU MLP comparison, a fixed-grid spline KAN comparison, and two ablations. The experiment learns f(x,y)=x exp(-y), the target used in the paper's ablation section. It is a reduced empirical study rather than the complete benchmark suite.

## 12. What data and splits did you use?
The code generates independent inputs uniformly in [-1,1]^2 and computes exact targets. Each seed uses 1,024 training, 256 validation, and 1,024 test examples. All compared models use the same data for a given seed. This domain and split are project choices, not a claim of matching the authors' full protocol.

## 13. Why separate validation data from test data?
Validation loss selects the checkpoint. The test set measures that selected model only afterward. Selecting weights, hyperparameters, or the best seed using test performance would bias the reported generalization estimate.

## 14. Which metrics do you report?
RMSE is the square root of average squared prediction error. MAE averages absolute errors. R-squared compares residual error with predicting the test-set mean. I also report parameter count, training time, and a warmed-up forward/backward timing measurement. Lower RMSE and MAE are better.

## 15. How did you make the model comparison fair?
The main models have similar parameter budgets, use identical data splits, and receive the same optimizer settings and update counts. Large PowerMLP has 689 parameters, MLP 709, and KAN 735. This controls several factors, but shared hyperparameters need not be optimal for every model, and the KAN implementation differs from the authors' baseline.

## 16. How do you calculate PowerMLP's parameter count?
A hidden layer with d inputs and m outputs has dm affine weights, m biases, and dm basis-projection weights: 2dm+m parameters. The final layer has dm+m. For [2,32,8,1], the total is 160+520+9=689.

## 17. What are the ablations, and what can they tell us?
One removes the SiLU basis branch; the other replaces ReLU cubed with ordinary ReLU while retaining the basis branch. They test the effect of each component under our settings. Removing the basis also reduces parameters, so it is not a perfectly capacity-matched comparison. Our experiment does not reproduce the paper's entire depth and width sweeps.

## 18. Did your implementation verify the paper's approximately 40-times speedup?
No. That figure is the authors' aggregate observation under their hardware and protocols. Our code measures its own runtime and uses a custom vectorized fixed-grid KAN. Hardware, batches, grid handling, implementation, and convergence criteria affect ratios. I would only quote our measured ratio with those conditions stated.

## 19. How did you verify correctness and reproducibility?
Tests compare the forward pass with the equation, perform double-precision finite-difference gradient checks for inputs and parameters, verify the paper's parameter counts, and compare spline basis values with SciPy. Fixed seeds, version metadata, configuration JSON, raw per-seed CSV results, histories, and checkpoints make the experiment traceable. Cross-device results can still differ.

## 20. What are the limitations and next steps?
The empirical evaluation covers one smooth regression target and a limited seed budget. It omits the full special-function suite, knot theory, and classification tasks. Next steps are the official pykan baseline, the authors' exact data and optimization protocol, more seeds, depth/width sweeps, and representative classification tasks. The paper also notes limits on complex vision and long-text tasks without specialized architectures.

## Before the viva
Run the Colab notebook once and read the newly generated results. Be ready to explain why the smallest error, fastest update, and fewest parameters need not belong to the same model. Describe results as measured observations rather than universal claims.
