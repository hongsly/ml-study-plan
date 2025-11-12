# Day 15 Quick Reference: A/B Testing + Statistics Fundamentals

**Week 3, Day 1** | Focus: Regularization as prior, A/B testing design, distributions, CLT vs LLN

---

## **1. Regularization as Bayesian Prior**

### **Frequentist vs Bayesian**:
- **Frequentist (MLE)**: Maximize P(D|θ) (likelihood)
- **Bayesian (MAP)**: Maximize P(θ|D) = P(D|θ) × P(θ) / P(D) (posterior)

### **MAP Estimation**:
```
Maximize P(θ|D) = P(D|θ) × P(θ)

Take log: maximize [log P(D|θ) + log P(θ)]
         = maximize [log likelihood + log prior]

Multiply by -1 (flip max → min):
         = minimize [-log P(D|θ) - log P(θ)]
         = minimize [loss + regularization]

where:
  loss = -log P(D|θ)  (negative log-likelihood)
  regularization = -log P(θ)  (negative log-prior)
```

**Key insight**: Maximizing posterior = minimizing (loss + regularization)

---

### **L2 Regularization = Gaussian Prior**

**Assume**: Parameters θ ~ N(0, τ²)

**Prior density**:
```
P(θ) = (1/√(2πτ²)) exp(-θ²/(2τ²))

log P(θ) = -θ²/(2τ²) + constant

To minimize -log P(θ):
  -log P(θ) = θ²/(2τ²) + constant
            = λθ²  where λ = 1/(2τ²)
```

**Result**: L2 regularization term = λ||θ||²

**Connection to MAP**:
```
minimize [loss + regularization]
= minimize [-log P(D|θ) + λθ²]
= maximize [log P(D|θ) - λθ²]
= maximize [log P(D|θ) + log P(θ)]  (MAP)
```

**Relationship**: Larger τ (wider prior, less confident) → smaller λ (less regularization)

---

### **L1 Regularization = Laplace Prior**

**Assume**: Parameters θ ~ Laplace(0, b)

**Prior density**:
```
P(θ) = (1/(2b)) exp(-|θ|/b)

log P(θ) = -|θ|/b + constant

To minimize -log P(θ):
  -log P(θ) = |θ|/b + constant
            = λ|θ|  where λ = 1/b
```

**Result**: L1 regularization term = λ||θ||₁

**Connection to MAP**:
```
minimize [loss + regularization]
= minimize [-log P(D|θ) + λ|θ|]
= maximize [log P(D|θ) - λ|θ|]
= maximize [log P(D|θ) + log P(θ)]  (MAP)
```

**Why L1 promotes sparsity**: Laplace distribution has sharp peak at 0, pushing small weights to exactly 0

---

## **2. A/B Testing Design**

### **Overall Loop**:
```
Research → Hypothesis + Metrics → Implement Variations → Run Test → Analyze
```

### **Metrics**:

**North Star Metrics**:
- Encapsulate core business value
- Example: Revenue per user, user retention, customer lifetime value
- Hard to move, but aligned with business goals

**Tactical Metrics**:
- Easy to measure, immediate feedback
- Example: Click-through rate, session duration, signup rate
- May not align with long-term core value

**Best Practice**: Each test should have **one primary metric** to avoid multiple comparisons problem.

---

### **Segmentation**:

**When**: Divide users into groups (pre or post randomization)

**Why**:
- Better interpretation of results
- Avoid pitfalls (novelty effect, Simpson's paradox)
- Understand heterogeneous treatment effects

**Example**: Segment by new vs returning users before analyzing results

---

## **3. Common A/B Testing Pitfalls**

### **Novelty Effect**:
**Problem**: Users behave differently simply because feature is new

**Solutions**:
1. Run test long enough (2-4 weeks minimum)
2. Wait a while after launch before measuring
3. Segment by new vs returning users (new users less affected)

---

### **Multiple Comparisons Problem**:
**Problem**: Testing multiple metrics or peeking at results inflates false positive rate

**Example**: Test 20 metrics at α=0.05 → expect 1 false positive even if no real effect

**Solutions**:
1. Choose **one primary metric** before the test
2. If must test multiple, use Bonferroni correction: α' = α/k (k = number of tests)
3. Don't peek! Decide sample size beforehand

---

### **Selection Bias**:
**Problem**: Treatment applied to non-random subset of users

**Examples**:
- Only applying to users in certain regions
- Technical issues causing inconsistent randomization
- Users self-selecting into treatment

**Solution**: True randomization with validation checks

---

### **Simpson's Paradox**:
**Problem**: Aggregated results contradict segmented results due to hidden confounding variable

**Example**:
```
Aggregated: B > A (60% vs 55%)
Segmented:
  - New users: A > B (70% vs 65%)
  - Returning users: A > B (40% vs 35%)
```

**Why**: B happens to get more "easy" new users, A gets more "hard" returning users

**Solution**: Always segment by key user characteristics and examine distributions

---

## **4. Common Distributions**

### **Binomial Distribution**

**Use**: Number of successes in n fixed independent trials

**PMF**:
```
P(X = k | n, p) = C(n,k) p^k (1-p)^(n-k)

where C(n,k) = n! / (k!(n-k)!)
```

**Parameters**:
- n: number of trials
- p: probability of success
- k: number of successes (0 to n)

**Moments**:
- Mean: μ = np
- Variance: σ² = np(1-p)

**Variance intuition**: Maximized at p=0.5 (most uncertainty), minimized at p=0 or p=1

**Example**: Flip 100 coins (n=100, p=0.5) → expect 50±5 heads

---

### **Geometric Distribution**

**Use**: Number of trials until first success (or failures before first success)

**PMF** (trials until success):
```
P(X = k | p) = (1-p)^(k-1) · p

k = 1, 2, 3, ...
```

**Mean**: μ = 1/p

**Example**: System fails with p=0.05 per hour. P(survives exactly 10 hours before failing) = 0.95^9 × 0.05

**Relationship**: Geometric is discretized version of Exponential (divide continuous time into intervals)

---

### **Poisson Distribution**

**Use**: Number of events in fixed time/space interval, given average rate λ

**PMF**:
```
P(X = k | λ) = e^(-λ) λ^k / k!

k = 0, 1, 2, ...
```

**Parameter**: λ = average number of events

**Mean**: μ = λ
**Variance**: σ² = λ

**Example**: Call center receives 12 calls/hour on average. For 30-minute window:
- λ = 6 (scale by time)
- P(X=k) = e^(-6) × 6^k / k!
- Expected calls = 6

**Relationships**:
- Binomial → Poisson (when n large, p small, np=λ)
- Time between Poisson events ~ Exponential(λ)

---

### **Exponential Distribution**

**Use**: Time between events in Poisson process

**PDF**:
```
f(x | λ) = λ e^(-λx)  for x ≥ 0
```

**Mean**: μ = 1/λ
**Variance**: σ² = 1/λ²

**Relationship**: If events occur as Poisson(λ), waiting time ~ Exponential(λ)

---

## **5. CLT vs LLN**

### **Law of Large Numbers (LLN)**

**Statement**: As sample size → ∞, sample mean → population mean

**Math**: x̄ₙ →^P μ as n → ∞

**Use**: Estimate population mean from sample mean

**Example**: Survey 10,000 customers to estimate average satisfaction. LLN guarantees x̄ ≈ μ.

---

### **Central Limit Theorem (CLT)**

**Statement**: As sample size → ∞, sampling distribution of x̄ → Normal(μ, σ²/n)

**Math**: √n(x̄ₙ - μ) →^D N(0, σ²) as n → ∞

**Use**: Apply normal distribution tools (confidence intervals, hypothesis tests) even when data isn't normal

**Key insight**: CLT is what makes most of statistics work! Real data rarely normal, but CLT says "sample means will be normal"

---

### **CLT Concrete Examples**

**Example 1: Confidence Intervals**
- Survey 50 customers (satisfaction 1-5, skewed distribution)
- Individual scores NOT normal
- But x̄ ~ N(μ, σ²/50) by CLT
- Can compute 95% CI = x̄ ± 1.96(s/√50) using z-scores

**Example 2: A/B Testing**
- Testing conversion rates (0 or 1, Bernoulli, NOT normal)
- With n=1000 per group, proportion p̂ approximately normal by CLT
- Can use z-test: z = (p̂₁ - p̂₂) / SE

**Example 3: Quality Control**
- Widget weights have unknown (maybe bimodal) distribution
- Take 30 widgets/day, compute daily average
- Daily averages follow normal distribution by CLT
- Can set control limits: μ ± 3σ

---

### **Difference Summary**

| Aspect | LLN | CLT |
|--------|-----|-----|
| **What it says** | x̄ → μ (gets close) | x̄ ~ Normal (predictable distribution) |
| **Use** | Estimate population mean | Quantify uncertainty (CI, p-values) |
| **Gives you** | Point estimate | Interval estimate + hypothesis tests |
| **Requires** | Large n | Large n (n≥30 rule of thumb) |
| **Works on** | Any distribution (finite mean) | Any distribution (finite variance) |

---

## **6. Interview Q&A**

### **Q: Explain the connection between L2 regularization and Bayesian priors.**
**A**: "L2 regularization corresponds to placing a Gaussian prior N(0, τ²) on the parameters. In the Bayesian framework, we maximize the posterior P(θ|D) = P(D|θ)P(θ), which in log space becomes maximizing [log P(D|θ) + log P(θ)].

When we multiply by -1 to convert to minimization (standard in optimization), we get: minimize [-log P(D|θ) - log P(θ)], which is minimize [loss + regularization]. The log of a Gaussian prior gives log P(θ) = -θ²/(2τ²), so -log P(θ) = θ²/(2τ²) = λθ², which is exactly the L2 penalty term.

The regularization strength λ = 1/(2τ²) is inversely proportional to the prior variance. Larger τ means wider prior (less confident about parameters being small) which corresponds to smaller λ (less regularization). This probabilistic interpretation shows that regularization encodes our prior belief about parameter magnitudes."

---

### **Q: You're running an A/B test and the aggregated results show B > A, but when segmented by user type, A > B in both segments. What's happening?**
**A**: "This is Simpson's Paradox, caused by a hidden confounding variable affecting the group distributions. For example, if treatment B happened to receive more 'easy' new users with naturally higher conversion rates, while A received more 'hard' returning users with lower baseline rates, the aggregated comparison is misleading. The segmented results are more trustworthy because they compare like-with-like within each user type. To prevent this, I'd ensure proper randomization and always check the balance of key covariates across treatment groups. The solution is to either report stratified results or use statistical methods that adjust for these imbalances."

---

### **Q: When would you use CLT vs LLN?**
**A**: "LLN tells us that our sample mean converges to the population mean—it's about point estimation. I'd reference this when justifying why taking a large sample gives a good estimate of the true mean. CLT goes further: it tells us the distribution of the sample mean is normal, which lets us quantify uncertainty. I'd use CLT when I need to:
1. Construct confidence intervals (x̄ ± z·σ/√n)
2. Perform hypothesis tests (z-tests, t-tests)
3. Do A/B testing with non-normal data

For example, in A/B testing with binary outcomes, individual data points are Bernoulli (not normal), but CLT guarantees the sample proportions follow a normal distribution with large n, allowing us to use standard statistical tests. Without CLT, most of practical statistics wouldn't work."

---

### **Q: Why does the binomial variance have the form np(1-p)?**
**A**: "Start with a single Bernoulli trial with variance p(1-p). This form makes intuitive sense: variance is maximized at p=0.5 (maximum uncertainty) and minimized at p=0 or p=1 (deterministic outcomes). For a binomial random variable, we're summing n independent Bernoulli trials. Since variance of independent random variables adds, Var(X₁ + ... + Xₙ) = n·Var(Bernoulli) = n·p(1-p). For example, flipping 100 fair coins: mean = 50, variance = 100×0.5×0.5 = 25, so standard deviation = 5. We'd expect 50±10 heads (within 2σ) about 95% of the time."

---

### **Q: What's the most important pitfall to avoid in A/B testing?**
**A**: "Multiple comparisons problem—it's insidious because it's tempting to measure everything. If I test 20 metrics at α=0.05, I expect one false positive even with no real effect. The discipline is to choose one primary metric before the test starts. Secondary metrics are fine for exploratory analysis, but decisions should be based on the primary metric. If I truly must test multiple hypotheses, I'd use Bonferroni correction (divide α by number of tests) or FDR control methods. Similarly, 'peeking' at results and stopping early when p<0.05 inflates the false positive rate—you're effectively running multiple tests. The solution is to pre-specify sample size and wait until you reach it."

---

## **7. Common Pitfalls**

❌ **"L2 regularization always improves model performance"**
✅ L2 is a form of prior. If your prior is wrong (parameters aren't actually small), regularization hurts. It's a bias-variance tradeoff.

❌ **"The MAP = minimize(loss + reg) connection only works for convex problems"**
✅ The probabilistic interpretation (maximize posterior = minimize loss + reg) holds for **any** model, convex or not! Convexity only affects whether gradient descent finds global vs local optima. Neural networks use the same MAP framework, just with non-convex loss landscapes.

❌ **"Geometric distribution: P(X=k) = (1-p)^k · p for k successes"**
✅ Two definitions exist: (1) trials until success: (1-p)^(k-1)·p, mean=1/p, or (2) failures before success: (1-p)^k·p, mean=(1-p)/p. Be clear which!

❌ **"Binomial and Poisson are unrelated"**
✅ Poisson is limiting case of Binomial when n→∞, p→0, np=λ fixed. Also: time between Poisson events ~ Exponential.

❌ **"CLT says data becomes normal with large n"**
✅ CLT says **sample mean** becomes normal. Individual data can stay non-normal forever.

❌ **"A/B test shows B > A with p=0.03, so B is better"**
✅ Statistical significance ≠ practical significance. Check effect size! Also consider: novelty effects, multiple comparisons, Simpson's paradox.

❌ **"P-value is probability the null hypothesis is true"**
✅ P-value = P(observe this data | H₀ true). NOT P(H₀ true | data). Common misinterpretation!

---

## **8. Key Formulas**

### **Regularization as Prior**:
```
MAP: maximize [log likelihood + log prior]
   = minimize [-log likelihood - log prior]
   = minimize [loss + regularization]

where:
  loss = -log P(D|θ)
  regularization = -log P(θ)

L2: Gaussian prior N(0, τ²) → regularization = λθ², λ = 1/(2τ²)
L1: Laplace prior Laplace(0, b) → regularization = λ|θ|, λ = 1/b
```

### **Distributions**:
```
Binomial: P(X=k) = C(n,k) p^k (1-p)^(n-k)
  Mean: np, Variance: np(1-p)

Geometric: P(X=k) = (1-p)^(k-1) p
  Mean: 1/p

Poisson: P(X=k) = e^(-λ) λ^k / k!
  Mean: λ, Variance: λ

Exponential: f(x) = λe^(-λx)
  Mean: 1/λ, Variance: 1/λ²
```

### **CLT**:
```
x̄ ~ N(μ, σ²/n) as n → ∞

95% CI: x̄ ± 1.96 × (σ/√n)
```

---

## **9. Study Tips**

- **Regularization = Prior**: Think probabilistically. λ controls strength of prior belief.
- **Sign flip is key**: Multiply by -1 to convert maximize(log posterior) → minimize(loss + reg). This works for any model (convex or not).
- **A/B Testing**: One primary metric, watch for Simpson's paradox, don't peek!
- **Distribution relationships**: Binomial ↔ Poisson, Geometric ↔ Exponential
- **CLT vs LLN**: LLN gives point estimate, CLT gives distribution → enables inference
- **Variance formulas**: Often derive from summing independent RVs (variances add!)
- **Simpson's Paradox**: Always segment and check covariate balance

---

## **Day 15 Study Outcome**

**Topics Mastered**:
- ✅ Regularization as Bayesian Prior (L2→Gaussian, L1→Laplace)
- ✅ A/B Testing Design (metrics, segmentation, pitfalls)
- ✅ Common Distributions (Binomial, Geometric, Poisson, Exponential)
- ✅ CLT vs LLN (applications and differences)

**Knowledge Check**: 92.0% (A-)

**Weak Areas**:
- Binomial variance derivation (75%)
- CLT concrete examples (70%)

**Next**: Week 3 Day 2 - PyTorch Basics (shifted from original Week 3 Day 1 plan)
