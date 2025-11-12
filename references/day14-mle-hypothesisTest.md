# Day 14 Quick Reference: MLE + Hypothesis Testing

**Week 2, Day 7** | Focus: MLE derivations, hypothesis testing (t-test, z-test, chi-square)

---

## **1. MLE (Maximum Likelihood Estimation)**

### **Concept**:
Find parameter values that maximize the probability of observing the data.

**Likelihood function**: L(θ|D) = P(D|θ)

**MLE approach**:
1. Write likelihood: L(θ) = ∏ᵢ P(xᵢ|θ)
2. Take log: log L(θ) (easier to differentiate)
3. Take derivative: d(log L)/dθ
4. Set to 0 and solve for θ

---

## **2. MLE for Exponential Distribution**

### **PDF**:
```
p(x|λ) = λe^(-λx)  for x ≥ 0
```

### **Derivation**:
```
L(λ|D) = ∏ᵢ λe^(-λxᵢ) = λⁿ e^(-λ Σxᵢ)

log L(λ) = n log(λ) - λ Σxᵢ

d(log L)/dλ = n/λ - Σxᵢ = 0

λ̂ = n / Σxᵢ = 1/x̄
```

**Result**: λ̂ = 1/sample_mean

### **Intuition**:
Exponential models "time until event". If average wait time is 3 minutes, rate λ = 1/3 events per minute.

---

## **3. MLE for Gaussian Distribution**

### **PDF**:
```
p(x|μ,σ²) = (1/√(2πσ²)) e^(-(x-μ)²/(2σ²))
```

### **MLE Estimators**:
```
μ̂ = x̄ = Σxᵢ / n  (sample mean)

σ̂² = Σ(xᵢ - μ̂)² / n  (sample variance)
```

**Note**: Biased estimator for σ² (use n-1 for unbiased). MLE gives n.

---

## **4. Hypothesis Testing Framework**

### **General Steps**:
1. **State hypotheses**: H₀ (null) vs H₁ (alternative)
2. **Choose significance level**: α (typically 0.05)
3. **Calculate test statistic**: Represents signal/noise ratio
4. **Find p-value**: P(observe this statistic | H₀ is true)
5. **Decision**: If p < α, reject H₀

---

## **5. Z-test**

### **When to Use**:
- Population standard deviation σ **known**
- OR large sample (n ≥ 30) where sample std approximates σ well
- Testing population mean μ

### **Assumptions**:
1. Continuous random variable
2. Normal distribution (or large n for CLT)
3. **Known population std**
4. Independent observations

### **Test Statistic**:
```
z = (x̄ - μ₀) / (σ / √n)

where:
  x̄ = sample mean
  μ₀ = hypothesized population mean
  σ = population std (known)
  n = sample size
```

### **P-value**: Use standard normal CDF N(0,1)

---

## **6. T-test**

### **When to Use**:
- Population standard deviation σ **unknown**
- Small to medium sample (n < 30 preferred, but works for any n)
- Testing population mean μ

### **Why T-distribution?**
When σ is unknown, we estimate it from data → adds uncertainty. T-distribution has **wider tails** than normal to account for this extra uncertainty.

**Key insight**: As n→∞, t-distribution → normal distribution (CLT)

### **Assumptions**:
1. Continuous random variable
2. Normal distribution (especially important for n < 30)
3. **Unknown population std** (estimated from sample)
4. Independent observations

---

## **7. Three Types of T-tests**

### **One-sample t-test**:
Test if population mean equals μ₀.

**H₀**: μ = μ₀

```
t = (x̄ - μ₀) / (s / √n)

where s = sample std = √(Σ(xᵢ - x̄)² / (n-1))

DOF = n - 1
```

---

### **Unpaired two-sample t-test** (independent samples):
Test if two independent populations have equal means.

**H₀**: μ₁ = μ₂

```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)

DOF = n₁ + n₂ - 2
```

**Example**: Comparing drug vs placebo groups.

---

### **Paired two-sample t-test** (related samples):
Test if two related populations have equal means (e.g., before/after treatment).

**H₀**: μ_diff = 0

```
t = d̄ / (s_d / √n)

where:
  d̄ = mean of differences (after - before)
  s_d = std of differences
  n = number of pairs

DOF = n - 1
```

**Example**: Blood pressure before vs after treatment (same patients).

---

## **8. Z-test vs T-test Comparison**

| Aspect | Z-test | T-test |
|--------|--------|--------|
| **Population σ** | Known | Unknown (estimated from sample) |
| **Sample size** | Large (n≥30) preferred | Any size (especially n<30) |
| **Distribution** | Normal N(0,1) | T-distribution (df = n-1) |
| **Tails** | Standard width | Wider tails (more conservative) |
| **Assumptions** | Normality or large n | Normality (critical for small n) |
| **P-value** | Normal CDF | T-distribution CDF |
| **Harder to reject H₀** | No | Yes (wider tails) |

**Rule of thumb**: If σ unknown, use t-test. For n≥30, t and z converge anyway.

---

## **9. Chi-square Test**

### **Purpose**:
Test if categorical data/frequencies differ significantly from expectation.

### **Chi-square Distribution**:
Sum of squared independent N(0,1) random variables.

---

### **Type 1: Goodness of Fit**
Test if observed data fits an alleged distribution.

**H₀**: Data follows the specified distribution

**Example**: Die rolled 60 times. Expected: 10 of each face. Observed: {8, 12, 9, 11, 10, 10}. Does die follow uniform distribution?

---

### **Type 2: Test of Independence (Contingency Table)**
Test if two categorical variables are independent.

**H₀**: Variables are independent (no relationship)

**Example**: Does gender affect voting preference?

|        | Candidate A | Candidate B |
|--------|-------------|-------------|
| Male   | 30          | 20          |
| Female | 25          | 25          |

---

### **Test Statistic**:
```
χ² = Σᵢ (Oᵢ - Eᵢ)² / Eᵢ

where:
  Oᵢ = observed count in category i
  Eᵢ = expected count in category i
```

### **Degrees of Freedom**:
- **Goodness of fit**: dof = k - 1 (k = number of categories)
- **Contingency table**: dof = (rows - 1) × (cols - 1)

### **P-value**: Use chi-square distribution CDF with dof

### **Decision**: If p < α, reject H₀

---

## **10. Normal Distribution Calculations**

### **Standard Normal**: N(0, 1)

**Z-score**:
```
z = (x - μ) / σ
```

Converts any normal N(μ, σ²) to standard normal N(0, 1).

### **CDF (Cumulative Distribution Function)**:
Φ(z) = P(X ≤ z)

### **Common Calculations**:
```
P(X > z) = 1 - Φ(z)
P(a < X < b) = Φ(b) - Φ(a)
P(X < -z or X > z) = 2(1 - Φ(z))  [two-tailed]
```

### **Key Z-score Values**:
```
z = 1.645 → 95% percentile (5% in upper tail)
z = 1.96  → 97.5% percentile (2.5% in upper tail, two-tailed α=0.05)
z = 2.576 → 99.5% percentile (0.5% in upper tail, two-tailed α=0.01)
```

### **Example**:
For N(0,1), P(X > 1.5) = 1 - Φ(1.5) ≈ 1 - 0.933 = 0.067 (6.7%)

---

## **11. Interview Q&A**

### **Q: Derive MLE for exponential distribution.**
**A**: "The exponential pdf is λe^(-λx). For n samples, the likelihood is λⁿ e^(-λΣxᵢ). Taking log gives n log(λ) - λΣxᵢ. Taking derivative with respect to λ: n/λ - Σxᵢ = 0. Solving for λ gives λ̂ = n/Σxᵢ = 1/x̄. This makes intuitive sense: if average time between events is 5 minutes, the rate is 1/5 = 0.2 events per minute."

---

### **Q: When do you use t-test vs z-test?**
**A**: "Use t-test when population standard deviation is unknown and you estimate it from the sample. Use z-test when population std is known, or for large samples (n≥30) where sample std approximates population std well. The key difference is that t-distribution has wider tails to account for the extra uncertainty from estimating σ. For small samples (n<30), this matters a lot—t-test is more conservative and less likely to incorrectly reject the null hypothesis. As sample size increases, the t-distribution converges to normal, so the choice matters less for large n."

---

### **Q: How do you use chi-square test for a contingency table?**
**A**: "Chi-square test of independence checks if two categorical variables are related. For example, testing if gender affects product preference. The null hypothesis is independence—no relationship between variables.

I calculate expected counts for each cell assuming independence: Expected = (row total × column total) / grand total. Then compute the chi-square statistic: χ² = Σ(Observed - Expected)²/Expected across all cells.

Degrees of freedom = (rows - 1) × (columns - 1). Compare the χ² statistic to the chi-square distribution with this df. If p < 0.05, reject the null—the variables are not independent, there's a significant relationship."

---

### **Q: You have 25 samples. Should you use z-test or t-test?**
**A**: "I'd use t-test. With n=25, we're below the n≥30 rule of thumb, and in practice we rarely know the true population standard deviation. The t-test with dof=24 accounts for the uncertainty in estimating σ from the sample by using a distribution with wider tails than the normal. This makes it harder to reject the null hypothesis, which is appropriate given our limited data. If I incorrectly used a z-test, I'd underestimate uncertainty and potentially find false significance."

---

## **12. Common Pitfalls**

❌ **"If I have n=100, I should still use z-test since n is large"**
✅ If σ is unknown, use t-test. With n=100, t and z give nearly identical results anyway, but t-test is technically correct.

❌ **"MLE always gives unbiased estimators"**
✅ MLE for Gaussian σ² uses n in denominator (biased). Unbiased version uses n-1. MLE prioritizes maximizing likelihood, not unbiasedness.

❌ **"Chi-square test tells me which categories are different"**
✅ Chi-square only tells you there's a significant difference somewhere. Post-hoc tests needed to identify specific categories.

❌ **"P-value tells me the probability H₀ is true"**
✅ P-value = P(observe this data | H₀ is true). It's NOT P(H₀ is true | data). Common misinterpretation!

❌ **"Gaussian MLE σ̂² uses n-1 denominator"**
✅ MLE uses n. The n-1 version is the unbiased estimator, not the MLE.

---

## **13. Key Formulas**

### **MLE**:
```
Exponential: λ̂ = 1/x̄
Gaussian: μ̂ = x̄, σ̂² = Σ(xᵢ - μ̂)²/n
```

### **Hypothesis Tests**:
```
Z-test: z = (x̄ - μ₀) / (σ/√n)
One-sample t-test: t = (x̄ - μ₀) / (s/√n), df = n-1
Unpaired t-test: t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂), df = n₁+n₂-2
Paired t-test: t = d̄ / (s_d/√n), df = n-1
Chi-square: χ² = Σ(Oᵢ - Eᵢ)²/Eᵢ
```

### **Normal Distribution**:
```
Z-score: z = (x - μ)/σ
P(X > z) = 1 - Φ(z)
```

---

## **15. Study Tips**

- **MLE intuition**: "Which parameter makes this data most likely?"
- **T vs Z**: If σ unknown → t-test (almost always in practice)
- **T-distribution**: Wider tails = harder to reject H₀ = more conservative
- **Chi-square**: Only for categorical data (counts/frequencies)
- **Independence matters**: DW test catches time series issues before they break your inference
- **DOF rules**: one-sample (n-1), two-sample (n₁+n₂-2), paired (n-1), chi-square contingency ((r-1)(c-1))

---

## **Day 14 Study Outcome**

**Topics Mastered**:
- ✅ MLE derivations (exponential, Gaussian)
- ✅ T-test vs Z-test (when to use, assumptions)
- ✅ Three types of t-tests
- ✅ Chi-square test (goodness of fit, independence)
- ✅ Normal distribution calculations

**Knowledge Check**: 86.5% (B+/A-)

**Next**: Week 3 Day 1 - A/B Testing Design + Statistics Fundamentals (PDF/CDF/PMF, CLT vs LLN)
