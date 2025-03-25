# model-based-DL
 
Attempt to implement the LISTA algorithm from Gregor and LeCun (2010)

# Background on convex optimization

Based on lectures of 

### The problem

We want to solve the optimization problem 

$\min_{x\in X} f(x)$ 

for some convex function $f$ over a convex set. 

## Gradient descent

### The problem
We want to solve the optimization problem 

$\min_{x} f(x)$ 

for some convex function $f$ (we ignore the condition of $x\in X$ for now). 

### The algorithm

For some arbitrary $x_0$ we iterate over

$x_{t+1} = x_t - \eta \nabla f(x_t)$

### Smooth function (in convex opt)

**Definition:** 

A convex function $f$ is called **$\beta$ - smooth** if it’s differential and it’s gradient is Lipschitz continuous with parameter $\beta$: 

$||\nabla f(x)- \nabla f(y)||_2 \le \beta ||x-y ||_2$

**Motivation:** 

We want “self-tuning” of the step size, i.e. 

$||\nabla f(x)||\to 0$ as $x\to x^*$

and so we don’t hop around the minimum.

This condition gives us 

1. We can fit a quadratic function above $f$ at any point $x$.
2. The hessian of $f$ is bounded from above by $\beta \cdot \text{Identity}$.

**Examples:**

1. $f(x)=|x|$ is not smooth (because of $x=0$).
2. $f(x)=x^TQx+q^Tx+c$ is $\beta$-smooth for any $\beta\ge 2\lambda_{\max}(Q)$ (the largest eigenvalue of $Q$) - direct application of the proposition on the hessian above). 

### Convergence

Gradient descent guaranteed to improve (for small enough $\eta$, in particular $\eta < \frac{2}{\beta}$). We get convergence rate of  $O(\frac{1}{\epsilon})$ iteration for error $\epsilon$.

## Sub-gradient method

### Subdifferential

For a convex function $f$ we can define its subdifferential - 

A **subderivative** **of a convex function $f$  at a point $x$ is a vector $k$  s.t. for all $y$,

$f(y)≥f(x)+<k,y-x>$ 

Since $f$ is convex, the set of subderivatives at $x_0$ (called the **subdifferential** of $f$ at $x_0$ and denoted $\partial f(x_0)$) is an interval $[a,b]$, where $a$ and $b$ are the one-sided “derivatives”.

In the other direction - $f$ is convex if for each $x_0$ there exist $k$ s.t. the inequality holds.

**Example**

$f(x)=|x|_1$, is not differentiable in $x=0$ but is in any other $x$. So 

$$
\partial f(x)=  
\begin{cases}
\{-1\} & x < 0 \\
\{1\}  & x > 0 \\
[-1,1] & x = 0
\end{cases}=  
\begin{cases}
\{\text{sign}(x)\} & x \not= 0 \\
[-1,1] & x = 0
\end{cases}
$$

### The algorithm

In particular, if $f$ is differentiable at $x_0$ then the subdifferential of $f$ at $x_0$ is the singleton $\{f’(x_0)\}$. 

Thus the sub-gradient method is a generalization of gradient descent:

 $x_{t+1} = x_t - \eta k_t, k_t\in \partial f(x_t)$

### Convergence

What can we say about this method?

1. This method is not a descent method - not improving at every step.
2. It does converge to $\min f$ (under assumptions):
    
    We assume $f$ is convex, but also doesn’t have any infinite (or “too large”) (sub-)gradients:
    
    **Assumption:** For all $x$ and $k_x\in \partial f(x)$, $||k_x||_2 \le G$. 
    
    (this is Lipschitz assumption).
    
    In addition, from definition, for each $x$ and $k_x\in \partial f(x)$ we have 
    
    $f(y)\ge f(x)+<k_x,y-x >$  for all $y$
    
    These two give us convergence to the minimum of $f$.
    
3. It converge with error $\epsilon$ in $O(\frac{1}{\epsilon^2})$ iterations.

## Proximal gradient method

### Projection

Until now we ignored the condition that $x\in X$. In GD (or the subgradient method), we can take a gradient step outside of $X$. 

One way to solve this is by projecting the result onto $X$ at each iteration:

![Screenshot 2025-01-02 at 20.29.14.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/1196a9e1-72f8-42a7-8e09-ee7395f8c36f/bae357c5-2dce-4dfd-9871-915277e8cf63/Screenshot_2025-01-02_at_20.29.14.png)

We get a **projected gradient descent algorithm**:

$*x_{t+1} = \text{Proj}_X(x_t - \eta \nabla f(x_t))*$

or similarly, a **projected subgradient method**:

$x_{t+1} = \text{Proj}_X(x_t  - \eta k_t), k_t\in \partial f(x_t)$

What can we say about the algorithm after this change?

**Convergence**

By repeating the same proof for subgradient method convergence with minor changes, we can prove convergence of the projected subgradient method, at the same rate (under the assumption that the subgradients are bounded from above). 

### Projection as proximal operator

Note that 

$$
\min_x f(x) \text{ s.t. } x\in X \iff \min_x f(x) + I_X(x) 
$$

where $I_X(x)=\infty\cdot(1_{x\not\in X})$ is the indicator of the set $X$.

So projected GD can be replaced with subgradient algorithm on the composite function $f(x)+I_X(x)$. But this will take $O(\frac{1}{\epsilon^2})$ iterations. Projected GD will take less, we’ll show this on a more general case.

### The proximal operator

We can write 

$$
\text{Proj}_X(u)
&=\arg\min_x\frac{1}{2}||x-u||_2^2 \text{  s.t. }x\in X \\ 
&= \arg\min_x I_X(x)+\frac{1}{2}||x-u||_2^2
$$

i.e. we approximate $u$ while still staying in $X$. So projected GD can be written as two optimization problems, and since we have an “easy” form for the projection, we can use it inside the GD:

$x_{t+1} = \text{Proj}_X(x_t - \eta \nabla f(x_t))$

We can now generalize this to some convex function $h$:

$\text{Prox}_h(u)= \arg\min_x h(x)+\frac{1}{2}||x-u||_2^2$

and to add some tradeoff constant $\eta$:

$\text{Prox}_{\eta h}(u)= \arg\min_x h(x)+\frac{1}{2\eta}||x-u||_2^2$

This is the approximation of the min point of $h$, while staying close to $u$, and the tradeoff between the two tasks is governed by $\eta$. Again if $h=I_X$, than this is the projection operator. 

**The proximal gradient method**

We can now use this inside GD to find the min of a composite function $\min_x f(x) + h(x)$:

$x_{t+1} = \text{Prox}_{\eta h}(x_t - \eta \nabla f(x_t))$

Specifically, if $f$ is $\beta$ smooth, we want to take $\eta=\frac{1}{\beta}$.

In case $f$ is not smooth, we can use the subgradient:

$x_{t+1} = \text{Prox}_{\eta h}(x_t - \eta k_t), k_t\in \partial f(x_t)$

**Convergence**

This is particularly useful where $f,h$ are convex, $f$ is smooth and $h$ is not, but has an easy proximal operator (see example below). In this case, the proximal gradient method gives convergence with error $\epsilon$ in $O(\frac{1}{\epsilon})$ - faster than subgradient method for this composite.

### Example - norm 1

What is the proximal operator of $h(x)=||x||_1$? First, we can maximize for each $x_i$ separately. We get:

$$
\begin{align*} 
x = \text{Prox}_{\eta h}(u) 
&\iff \forall_i, x_i = \arg\min_{x_i} |x_i| +\frac{1}{2\eta}(u_i - x_i)^2 \\
&\iff 
\begin{cases}
1 -\frac{1}{\eta}(u_i - x_i)= 0, & x_i > 0 \\
- 1 -\frac{1}{\eta}(u_i - x_i)= 0, & x_i < 0 \\
x_i = 0 & \text{else}
\end{cases} \\
&\iff 
x_i = \begin{cases}
u_i - \eta, & x_i > 0 \\
u_i + \eta, & x_i < 0\\
0 & \text{else}
\end{cases}\\
&\iff 
x_i = \begin{cases}
u_i - \eta, & u_i > \eta \\
u_i + \eta, & u_i < -\eta \\
0 & \text{else}
\end{cases}\\
&\iff 
x_i = \text{sign}(u_i)\cdot \max(|u_i|-\eta, 0) 
\end{align*}
$$

This function is a soft thresholding of $\eta$ - everything bigger than $\eta$ or smaller than $-\eta$ is pulled toward 0 by $\eta$, and everything within $\eta$ of 0 becomes 0: 

![Screenshot 2025-01-06 at 14.49.47.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/1196a9e1-72f8-42a7-8e09-ee7395f8c36f/37ad16f3-a34a-4d01-8479-988afd7154ab/Screenshot_2025-01-06_at_14.49.47.png)

This is the general behavior of the proximal - pulling the points toward the minimum, or toward the boundary of the domain:

![https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf](https://prod-files-secure.s3.us-west-2.amazonaws.com/1196a9e1-72f8-42a7-8e09-ee7395f8c36f/fdce85b8-4f55-4216-8d88-95ed174a53cf/Screenshot_2025-01-06_at_14.46.53.png)

https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf

## Summary

For an optimization problem of the form:

$\min_x f(x) = g(x) + h(x)$

where g is smooth and h is not smooth but has an “easy” proximal operator,

we have the following options:

1. Sub-gradient method: 
    
    $x_{t+1} = x_t - \eta k_t, k_t\in \partial f(x_t)$
    
    which gives convergence with error $\epsilon$ in $O(\frac{1}{\epsilon^2})$ iterations.
    
2. Proximal gradient method:
    
    $x_{t+1} = \text{prox}_{\eta h} (x_t - \eta \nabla g(x_t))$
    
    which gives convergence with error $\epsilon$ in $O(\frac{1}{\epsilon})$ iterations.
    
3. Proximal gradient method + acceleration (momentum):
    
    $z_{t+1} = \text{prox}_{\eta h} (x_t - \eta \nabla g(x_t))$
    
    $x_{t+1} = (1-\gamma_t) z_{t+1} + \gamma_t z_{t}$
    
    which gives convergence with error $\epsilon$ in $O(\frac{1}{\sqrt{\epsilon}})$ iterations. 
    

# Lasso - sparse coding

## The problem

The Lasso problem (L1 regularized regression) has the following form:

 $x\in \R^p$ vector (p is large, i.e. an image), but we observe $x$ only through (noisy) linear measurements:  $y_i = a_i ^T x + \epsilon_i$ 

How many measurements do we need to reconstruct $x$? 

Without any prior knowledge on $x$, we need $n>p$ measurements ($n=p$ if there’s no noise). 

But suppose we know $x$ has only $s$ non-zero entries. Then we need much less measurements (under some technical assumptions - $n ~ s \log \frac{p}{\epsilon^2}$). So we assume $n<p$.

Under these technical assumptions we can write the problem as an optimization problem:

$\hat{x}=\arg\min_x f(x) =||Ax-y||_2^2+\lambda ||x||_1$

$f$  is convex but not smooth:

Convex: a sum of convex functions. 

Not smooth: The first term ($g(x)=||Ax-y||_2^2$) is smooth but the second ($h(x)=||x||_1$) is not, so $f$ is not smooth. 

Therefore we can apply sub-gradient method but not gradient descent. 

## Sub-gradient method for LASSO

$x_{t+1} = x_t - \eta k_t, k_t\in \partial f(x_t)$

What is the sub-differential of $f$? 

Equal to $\partial g + \lambda \partial h$. 

$g$ is smooth and have a derivative everywhere: 

$\partial g(x)=\{2A^T(Ax-y)\}$

but $h$ is non differentiable if some entry $x_i=0$. It’s sub-differential is:

$\partial h(x) =\{z\in \R^n \mid z_i=\text{sign}(x_i) \text{ if } x_i \not = 0, \text{ otherwise } z_i\in[-1 ,1 ] \}$

Overall we get an **update rule**:

$x_{t+1} = x_t - \eta (2A^T(Ax_t-y) + \lambda z), z\in \partial h(x_t)$

If $g$ is $\beta$-smooth, we need $\eta=\frac{1}{\beta}$ to assure convergence. Specifically, norm 2 is smooth with the largest eigenvalue of $A^TA$,  $\beta = \lambda_{\max}$.  So we have  

$x_{t+1} = x_t - \frac{2}{\lambda_{\max}} A^TAx_t + \frac{2}{\lambda_{\max}} A^Ty - \frac{\lambda}{\lambda_{\max}}   z), z\in \partial h(x_t)$

Setting $S=I - \frac{2}{\beta}A^TA$ and $W_e=\frac{2}{\beta}A^T$, we get: 

$x_{t+1} = S\cdot x_t + W_e \cdot y - \frac{\lambda}{\lambda_{\max}} \cdot \text{sign}(x_t)$

where the sign function is defined per entry and equals 0 when $x=0$ .

## Proximal gradient method for LASSO - ISTA

Lasso is an optimization problem of the form:

$\min_x f(x) = g(x) + h(x)$

where g is smooth and h is not smooth but has an “easy” proximal operator.

We can thus use the update rule:

$x_{t+1} = \text{prox}_{\eta h} (x_t - \eta \nabla g(x_t))$

If $g$ is $\beta$-smooth, we need $\eta=\frac{1}{\beta}$ (to assure convergence). Specifically, norm 2 is smooth with the largest eigenvalue of $A^TA$,  $\beta = \lambda_{\max}$.  So we have 

$$
\begin{align*}
x_{t+1} 
&= \arg\min_{x} \lambda ||x||_1 + \frac{\beta}{2}||x - (x_t - \frac{1}{\beta}\nabla g(x_t))||_2^2 \\

&= \arg\min_{x} ||x||_1 + \frac{1}{2\frac{\lambda}{\beta}}||x - (x_t - \frac{1}{\beta}\nabla g(x_t))||_2^2 \\

&=\text{Prox}_{\frac{\lambda}{\beta}||\cdot||_1}(x_t - \frac{1}{\beta}\nabla g(x_t)) \\
&=\text{Prox}_{\frac{\lambda}{\beta}||\cdot||_1}(x_t - \frac{2}{\beta}A^T(Ax_t-y))\\
&=\text{Prox}_{\frac{\lambda}{\beta}||\cdot||_1}((I - \frac{2}{\beta}A^TA)x_t + \frac{2}{\beta}A^Ty))
\end{align*}
$$

for $\text{Prox}_{\alpha||\cdot||_1}(u)_i = \text{sign}(u_i)\cdot\max(|u_i|-\alpha_i, 0)$.

Setting $S=I - \frac{2}{\beta}A^TA$ and $W_e=\frac{2}{\beta}A^T$, we get:

$$
\begin{align*}
x_{t+1} 
&=\text{Prox}_{\frac{\lambda}{\beta}||\cdot||_1}((S\cdot x_t + W_e\cdot y))
\end{align*}
$$

Taking $\beta=2\lambda_{\max}$ gives us the definitions in [REF].

## Proximal gradient method + acceleration for LASSO - FISTA

Gregor, K., & LeCun, Y. (2010, June). Learning fast approximations of sparse coding. In Proceedings of the 27th international conference on international conference on machine learning (pp. 399-406).
