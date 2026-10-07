**Inputs:**

- Training Data $D = {(X,y)}$ 

- Loss function $L$
  
    - Regression Loss: 
    	- $l(y, \hat{y}) = (y - \hat{y})^2$
    	- Gradient: $g = \hat{y} - y$
    	- Hessian: $h = 1$

    - classification loss:
    	- $l(y,\hat{y}) = -[y log(p) + (1-y) log(1-p)$ where $p = 1 / (1 + e^{-\hat{y}})$
    	- Gradient: $g = p - y$
    	- Hessian: $h = p ( 1-p)$

- HPs 
  - number of trees: $K$ 
  - learning rate: $\eta$
  - max_depth: $md$
  - min_child_weight: $mH$


**Output:** ensemble model predictions

**Steps:**

1. Base prediction: $y_0 = arg min (L)$
	- mean (regression)
	- log-odds of event rate (classification)

2. For $t$ from 1 to K:
	
	1. Compute gradients g = 1st derivative of $L(y,y_{(t-1)})$ w.r.t.  $y_{(t-1)}$
	
	2. Compute hessians h = 2nd derivative of $L(y,y_{(t-1)})$ w.r.t.  $y_{(t-1)}$

	3. **Function RecursiveBuildTree(D, g, h, d):**
		
		1. $G = \sum g_i, H = \sum h_i$
		
		2. Leaf weight formula: $w = - G / (H + \lambda)$ **(WHY?)**
		
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
					
					4. gain formula: $\text{Gain} = \dfrac{1}{2}\left[\dfrac{G_L^2}{H_L + \lambda} + \dfrac{G_R^2}{H_R + \lambda} - \dfrac{G^2}{H + \lambda}\right] - \gamma$  **(WHY?)**
					
					1. If $Gain^* < Gain : Gain^* = Gain, j^* = j, v^* = v$
					
					2. Repeat for each split candidate v
				
				1. Repeat for each feature j

			1. Return $(j^*, v^*, Gain^*)$
		
		1. Partition D based on $j^* <= v^*$, create two leaves for this node
		
		2. Recursively build tree for left leaf
		
		3. Recursively build tree for right leaf
		
		4. Return final tree object (parent node)

	1. Update prediction: $y_t = y_{(t-1)} + \eta * f_t(X)$ 
	
	2. Repeat for each tree $t$

1. Return final prediction: $y_K = y_0 + \sum_{t=1}^{K} \eta * f_t (X)$