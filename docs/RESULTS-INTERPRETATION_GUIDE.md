# APGI Framework Results Interpretation Guide

This guide explains how to interpret the results from APGI Framework falsification tests and statistical analyses.

## Table of Contents

1. [Understanding Falsification](#understanding-falsification)
2. [Test Result Categories](#test-result-categories)
3. [Statistical Measures](#statistical-measures)
4. [Effect Sizes](#effect-sizes)
5. [Confidence Intervals](#confidence-intervals)
6. [P-Values and Significance](#p-values-and-significance)
7. [Result Visualization](#result-visualization)
8. [Common Patterns](#common-patterns)
9. [Troubleshooting Results](#troubleshooting-results)

## Understanding Falsification

Falsification in the APGI Framework tests whether the Allostatic Precision-Gated Ignition (APGI) model can be distinguished from alternative models or null hypotheses.

### Key Concepts

- **Falsification**: Evidence that the APGI model does not hold
- **Non-falsification**: Lack of evidence against the APGI model
- **Statistical Power**: Ability to detect true effects
- **Type I Error**: False positive (incorrectly rejecting APGI)
- **Type II Error**: False negative (failing to reject incorrect alternative)

## Test Result Categories

### Primary Falsification Test

Tests the core APGI equation against baseline predictions.

#### Interpretation

- **Falsified**: APGI equation fails to predict observed behavior
- **Not Falsified**: APGI equation is consistent with observations
- **Inconclusive**: Insufficient data or statistical power

### Secondary Tests

#### Consciousness Without Ignition

- **Purpose**: Tests if consciousness can occur without precision-weighted prediction error
- **Falsification**: Evidence that consciousness occurs without ignition
- **Interpretation**: Suggests alternative mechanisms for consciousness

#### Threshold Insensitivity

- **Purpose**: Tests if consciousness threshold is sensitive to experimental manipulations
- **Falsification**: Threshold remains unchanged despite interventions
- **Interpretation**: May indicate hard-coded thresholds or measurement issues

#### Soma-Bias Test

- **Purpose**: Tests if somatic markers disproportionately influence decision-making
- **Falsification**: Decisions biased toward somatic gain/loss patterns
- **Interpretation**: Supports somatic marker hypothesis over pure rationality

## Statistical Measures

### Key Metrics

| Metric | Range | Interpretation |
| ------ | ----- | -------------- |
| **Effect Size** | 0.0 - ∞ | Magnitude of observed effect |
| **Cohen's d** | 0.0 - ∞ | Standardized difference between means |
| **R²** | 0.0 - 1.0 | Proportion of variance explained |
| **AIC/BIC** | -∞ - ∞ | Model fit quality (lower is better) |

### Effect Size Guidelines

- **Small**: d = 0.2 (negligible practical impact)
- **Medium**: d = 0.5 (moderate practical impact)
- **Large**: d = 0.8 (strong practical impact)
- **Very Large**: d ≥ 1.0 (substantial practical impact)

## Effect Sizes

### Interpretation Framework

#### Small Effects (d = 0.2)

- May be statistically significant with large samples
- Practical importance depends on context
- Could be measurement error or minor systematic bias

#### Medium Effects (d = 0.5)

- Clearly detectable differences
- May have practical significance
- Worth further investigation

#### Large Effects (d = 0.8)

- Strong evidence of meaningful differences
- Likely to have practical implications
- High confidence in results

### Domain-Specific Considerations

#### Neural Signals

- Small effects may be important in high-precision measurements
- Consider signal-to-noise ratios
- Account for baseline variability

#### Behavioral Data

- Medium effects are typically meaningful
- Consider learning effects and fatigue
- Account for individual differences

#### Simulation Results

- Focus on stability and convergence
- Consider computational precision limits
- Evaluate against theoretical predictions

## Confidence Intervals

### Understanding CIs

Confidence intervals provide a range of plausible values for the true population parameter.

#### 95% Confidence Interval Interpretation

- **Narrow CI**: Precise estimate, high confidence in true value
- **Wide CI**: Imprecise estimate, more uncertainty
- **CI excludes zero**: Statistically significant effect
- **CI includes zero**: Effect may be due to chance

### Practical Guidelines

#### CI Width Assessment

- **Very Narrow (< 0.1)**: High precision, strong confidence
- **Narrow (0.1 - 0.3)**: Good precision, reliable estimate
- **Moderate (0.3 - 0.5)**: Acceptable precision
- **Wide (> 0.5)**: Low precision, consider more data

#### Decision Making

- Use CI width to assess reliability
- Consider CI overlap between conditions
- Account for multiple comparisons

## P-Values and Significance

### P-Value Interpretation

P-values indicate the probability of observing the data (or more extreme) assuming the null hypothesis is true.

#### Significance Levels

- **p < 0.001**: Very strong evidence against null hypothesis
- **p < 0.01**: Strong evidence against null hypothesis
- **p < 0.05**: Moderate evidence against null hypothesis
- **p ≥ 0.05**: No statistical evidence against null hypothesis

### Common Misinterpretations

#### What P-Values DON'T Mean

- P-value is NOT the probability that the null hypothesis is true
- P-value does NOT indicate the size or importance of an effect
- P-value is NOT the probability of replication

#### What P-Values DO Mean

- Probability of observed data given null hypothesis
- Strength of evidence against null hypothesis
- Should be considered alongside effect sizes and confidence intervals

### Multiple Testing Correction

When conducting multiple tests, p-values should be corrected for multiple comparisons

- **Bonferroni**: Divide α by number of tests
- **Holm-Bonferroni**: Step-down correction
- **False Discovery Rate (FDR)**: Control expected proportion of false positives

## Result Visualization

### Common Plot Types

#### Time Series Plots

- Show parameter evolution over time
- Identify convergence patterns
- Detect oscillations or instabilities

#### Statistical Charts

- Box plots for distribution comparison
- Bar charts with error bars
- Scatter plots for correlations

#### Network Diagrams

- Model component relationships
- Information flow visualization
- Parameter interaction networks

#### Heat Maps

- Parameter sensitivity analysis
- Correlation matrices
- Cross-validation results

### Interpretation Guidelines

#### Visual Patterns to Look For

- **Convergence**: Parameters stabilizing over time
- **Oscillations**: May indicate instability or feedback loops
- **Outliers**: Potential measurement errors or unusual conditions
- **Clusters**: Natural grouping in parameter space

## Common Patterns

### Result Patterns and Interpretations

#### Consistent Falsification Across Tests

- **Interpretation**: Strong evidence against APGI model
- **Action**: Consider model modifications or alternative theories

#### Inconsistent Results

- **Interpretation**: Measurement issues or context-dependent effects
- **Action**: Check experimental conditions and data quality

#### Small but Significant Effects

- **Interpretation**: Subtle but reliable differences
- **Action**: Investigate practical significance and mechanisms

#### Large Effects in Simulation but Not Behavior

- **Interpretation**: Model discrepancy with real-world data
- **Action**: Validate simulation parameters and assumptions

### Replication Considerations

- **Single Study**: Preliminary evidence, needs replication
- **Multiple Studies**: Stronger evidence if consistent
- **Meta-Analysis**: Quantitative synthesis across studies

## Troubleshooting Results

### Common Issues

#### Unexpected Results

1. **Check Data Quality**: Outliers, missing values, measurement errors
2. **Verify Parameters**: Correct model implementation
3. **Review Assumptions**: Statistical and theoretical assumptions
4. **Consider Context**: Experimental conditions and participant characteristics

#### Statistical Issues

1. **Low Power**: Small sample sizes, high variability
2. **Multiple Testing**: Uncorrected p-values
3. **Assumption Violations**: Non-normal data, heteroscedasticity
4. **Confounding Variables**: Uncontrolled experimental factors

#### Model Issues

1. **Parameter Ranges**: Check if parameters are within valid ranges
2. **Implementation Errors**: Verify mathematical correctness
3. **Boundary Conditions**: Test extreme parameter values
4. **Numerical Stability**: Check for computational errors

### Diagnostic Steps

#### Data Validation

```python
# Check data distributions
import numpy as np
from scipy import stats

# Normality test
statistic, p_value = stats.shapiro(data)

# Outlier detection
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
outliers = data[(data < Q1 - 1.5*IQR) | (data > Q3 + 1.5*IQR)]
```

#### Model Validation

- Verify parameter bounds
- Check mathematical consistency
- Validate against known test cases

#### Statistical Validation

- Check test assumptions
- Verify effect size calculations
- Confirm confidence interval methods

### Getting Help

#### Documentation Resources

- [APGI Equations](APGI-Equations.md)
- [Parameter Specifications](APGI-Parameter-Specifications.md)
- [Testing Guide](TESTING.md)

#### Technical Support

- Check error logs for specific issues
- Review system requirements
- Contact development team for complex issues

#### Best Practices

- Always report effect sizes with p-values
- Include confidence intervals
- Document analysis methods
- Consider practical significance
- Plan for replication studies

---

For technical details on the APGI model, see [APGI Equations](APGI-Equations.md).
For experimental design guidance, see [Experiments Guide](EXPERIMENTS.md).
