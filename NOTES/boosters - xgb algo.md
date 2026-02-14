
Inputs: 
- Training Data $D = {(X,y)}$ 
- Loss function $L$
- HPs - number of trees K, learning rate $\eta$ , max_depth $md$, min_child_weight $mH$

- Regression Loss: 
	- $l(y, \hat{y}) = (y - \hat{y})^2$
	- Gradient: $g = \hat{y} - y$
	- Hessian: $h = 1$

- classification loss:
	- $l(y,\hat{y}) = -[y log(p) + (1-y) log(1-p)$ where $p = 1 / (1 + e^{-\hat{y}})$
	- Gradient: $g = p - y$
	- Hessian: $h = p ( 1-p)$

Output: ensemble model predictions

Steps:
1. Base prediction: $y_0 = arg min (L)$
	1. mean (regression)
	2. log-odds of event rate (classification)

2. For $t$ from 1 to K:
	1. Compute gradients g = 1st derivative of $L(y,y_{(t-1)})$ w.r.t.  $y_{(t-1)}$
	2. Compute hessians h = 2nd derivative of $L(y,y_{(t-1)})$ w.r.t.  $y_{(t-1)}$

	3. **Function RecursiveBuildTree(D, g, h, d):**
		1. $G = \sum g_i, H = \sum h_i$
		2. Leaf weight formula **(WHY?)** $$w = - G / (H + \lambda)$$
		3. If d (current_depth) >= md (max_depth): return Leaf weight

		4. **Function FindBestSplit(D, g, h):**
			1. $G = \sum g_i, H = \sum h_i$
			2. $Gain^* = -\infty, j^* = null, v^* = null$

			3. For each feature $j$:
				1. sort D by feature values
				2. For each split candidate v:
					1. create partitions $D_L$ and $D_R$
					2. compute $G_L, G_R, H_L, H_R$
					3. check constraints: If $H_L$ or $H_R$ is < $mH$: continue
					4. gain formula: **(WHY?)** $$\text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma$$
					5. If $Gain^* < Gain : Gain^* = Gain, j^* = j, v^* = v$
					6. Repeat for each split candidate v
				3. Repeat for each feature j

			4. Return $(j^*, v^*, Gain^*)$
		
		5. Partition D based on $j^* <= v^*$, create two leaves for this node
		6. Recursively build tree for left leaf
		7. Recursively build tree for right leaf
		8. Return final tree object (parent node)

	4. Update prediction: $y_t = y_{(t-1)} + \eta * f_t(X)$ 
	5. Repeat for each tree $t$

3. Return final prediction $$y_K = y_0 + \sum_{t=1}^{K} \eta * f_t (X)$$