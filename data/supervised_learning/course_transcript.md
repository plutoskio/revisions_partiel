# Supervised Learning Course Transcript

## 01 Logistic Regression

### Slide 1

Lecture 1: Logistic Regression & Linear Classiﬁcation
Alexis BOGROFF
Course on Classiﬁcation Problems @Albert School
January 20, 2026
January 20, 2026 1 / 30

### Slide 2

Machine Learning Workﬂow Overview
• Deﬁne the Problem: Clearly specify the objective and collect quality,
representative data.
• Data Splitting: Divide the data into training and test sets (prevent data leakage).
• Data Preparation: Preprocess data through cleaning, normalization, encoding, and
feature engineering.
• Model Training: Use the correct algorithms to learn patterns from training data.
• Model Evaluation: Validate the model using metrics (Accuracy, Log-Loss).
• Deployment and Monitoring: Deploy the model into production.
January 20, 2026 2 / 30

### Slide 3

Introduction to Classiﬁcation Problems
• Classiﬁcation is a supervised learning task where the output variable is categorical .
• Examples:
◦ Spam detection (Binary: Yes/No).
◦ Churn prediction (Binary: Churn/Stay).
◦ Handwritten digit recognition (Multi-class: 0-9).
January 20, 2026 3 / 30

### Slide 4

What does Classiﬁcation Data look like?
The Structure of a Supervised Dataset
• Features ( X ): The input variables (columns) describing the observation.
• Target ( Y ): The label we want to predict. In classiﬁcation, Y is discrete.
Example: Titanic Dataset (Binary Classiﬁcation)
Age ( X 1 ) Fare ( X 2 ) Class ( X 3 ) Survived? ( Y )
22 7.25 3 No (0)
38 71.28 1 Yes (1)
26 7.92 3 Yes (1)
35 53.10 1 Yes (1)
...
...
...
...
Goal: Learn a function f such that f (Age , Fare , Class) ≈ Survived.
January 20, 2026 4 / 30

### Slide 5

What is Logistic Regression?
A misnomer: Despite its name, it is a Classiﬁcation algorithm, not a Regression one.
• Role: It predicts the probability that an observation belongs to a speciﬁc category
(Class 0 or Class 1).
• Type: It is a Linear Model . It assumes a linear relationship between input features
and the log-odds of the outcome.
• Status in AI: It is considered the ”Hello World” of classiﬁcation. Simple, fast, but
surprisingly powerful.
”If you can’t beat Logistic Regression with your fancy Deep Learning model, use
Logistic Regression.”
January 20, 2026 5 / 30

### Slide 6

Real-World Use Cases
Where is Logistic Regression actually used in the industry?
1. Banking & Finance
• Credit Scoring: Will this
customer default on their
loan?
• Why LogReg?
Regulations require
explainability . You must
explain why a loan was
refused.
2. Medicine
• Diagnosis: Based on
symptoms, does the
patient have Disease X?
• Risk Factors:
Calculating Odds Ratios
(e.g., ”Smoking increases
risk by 5x”).
3. Marketing
• Churn Prediction: Will
this subscriber cancel
their subscription next
month?
• Ad Click: Will the user
click on this banner?
(CTR).
January 20, 2026 6 / 30

### Slide 7

Strategic Choice: Why use Logistic Regression?
Why choose a simple linear model over complex ones (Random Forest, Neural
Networks)?
Feature Logistic Regression Complex Models (e.g. XGBoost)
Interpretability High (White Box) Low (Black Box)
Training Speed Very Fast Slower
Data Needed Small datasets ok Needs massive data
Performance Good on simple data Best on complex data
Decision Rule: Start with Logistic Regression as a Baseline . Only switch to complex
models if the performance gain is signiﬁcant and worth the loss of interpretability.
January 20, 2026 7 / 30

### Slide 8

Limit: The Linearity Constraint
Logistic Regression draws a straight line (or plane) to separate classes.
When it works:
• Classes are ”linearly separable”.
• Example: Separating huge tumors
(Malignant) from small ones (Benign).
When it fails (The XOR Problem):
• If classes are arranged in a circle or a
checkerboard pattern.
• A single straight line cannot separate
them.
• Solution: Feature Engineering (add x
2
)
or Non-linear models (Trees, Neural
Nets).
January 20, 2026 8 / 30

### Slide 9

Attempting to Model Probability Linearly
The Challenge: How can we predict a binary probability P using a linear combination
of features?
Hypothesis (Attempt #1): Let’s try to equate P directly to the linear equation.
Let’s test: P ( Y = 1 | X ) = β 0 + β 1 X 1 + ··· + β p X p
Why this is a ”Dead End”:
1. Mathematical Impossibility (Range Mismatch):
◦ The linear term β
T
X goes from −∞ to + ∞ .
◦ A probability must stay in [0 , 1].
◦ Result: For extreme X , the model predicts P = − 0 . 5 or P = 1 . 2. It breaks.
2. Behavioral mismatch:
◦ A straight line assumes constant impact. In reality, probabilities ”saturate” (diminishing
returns).
Conclusion: We cannot model P directly. We need to transform it into an unbounded
value ﬁrst.
January 20, 2026 9 / 30

### Slide 10

The Logistic Model: Sigmoid Function
• We model the probability P ( Y = 1 | X )
using the sigmoid function σ ( z ):
σ ( z ) =
1
1 + e − z
• Where z is the linear combination of
features (the logit):
z = β 0 + β 1 X 1 + ··· + β p X p
Key property: σ (0) = 0 . 5
January 20, 2026 10 / 30

### Slide 11

Why Sigmoid? Comparison with other functions
Why don’t we use functions popular in Deep Learning like ReLU or Tanh ?
• Sigmoid: Ranges in [ 0 , 1 ].
• Tanh (Hyperbolic Tangent): Ranges in
[− 1 , 1 ].
◦ Problem: A probability cannot be negative ( P < 0
makes no sense).
• ReLU (Rectiﬁed Linear Unit): Ranges in
[ 0 , + ∞ ).
◦ Formula: f ( x ) = max(0 , x ).
◦ Problem: Probabilities cannot exceed 1 ( P > 1
makes no sense).
Only Sigmoid maps strictly to
probability space.
January 20, 2026 11 / 30

### Slide 12

Theoretical Coherence: The Canonical Link
Beyond the range [0 , 1], Sigmoid is the mathematically consistent choice for Linear
Classiﬁcation.
• 1. Smoothness (Diﬀerentiability): Unlike the ”Step Function” (Perceptron)
which jumps from 0 to 1, Sigmoid is smooth. This allows us to compute gradients
for optimization.
• 2. The Log-Odds assumption: If we assume that the Log-Odds are linear:
log
(
p
1 − p
)
= β
T
X
Then, by inverting this equation, we mathematically obtain the Sigmoid function:
p =
1
1 + e − β T X
Conclusion: Sigmoid is not arbitrary; it is the natural inverse of the Linear
Log-Odds.
January 20, 2026 12 / 30

### Slide 13

Understanding Odds and Log-Odds
To understand the coeﬃcients β , we need to look at the Odds :
• Probability: P ∈ [0 , 1]
• Odds:
P
1− P
∈ [0 , + ∞ ) (How much more likely is success than failure?)
• Log-Odds (Logit): log
(
P
1− P
)
∈ (−∞ , + ∞ )
The Linear Link:
log
(
P ( Y = 1)
P ( Y = 0)
)
= β 0 + β 1 X 1 + ··· + β p X p
Logistic Regression is a Linear Regression on the Log-Odds.
January 20, 2026 13 / 30

### Slide 14

Deriving the Sigmoid Function (1/2)
How do we get from the Log-Odds back to the Probability P ?
1. Start with the Linear Assumption:
log
(
P
1 − P
)
= β 0 + β 1 X 1 + ...
  
Let’s call this z
⇒ log
(
P
1 − P
)
= z
2. Remove the Logarithm (Exponentiate both sides):
P
1 − P
= e
z
3. Isolate P (Cross-multiply):
P = e
z
(1 − P )
P = e
z
− P · e
z
January 20, 2026 14 / 30

### Slide 15

Deriving the Sigmoid Function (2/2)
4. Group terms with P on the left:
P + P · e
z
= e
z
P (1 + e
z
) = e
z
5. Solve for P:
P =
e
z
1 + e z
6. Final Polish (Divide by e
z
): To get the standard ”Sigmoid” form, we multiply top
and bottom by e
− z
:
P =
1
e − z + 1
= σ ( z )
Conclusion: The Sigmoid is the inverse of the Logit.
January 20, 2026 15 / 30

### Slide 16

Decision Boundary: The Geometry
• The decision boundary is the set of points where the model is uncertain ( P = 0 . 5).
• Mathematically, σ ( z ) = 0 . 5 ⇐⇒ z = 0.
β 0 + β 1 X 1 + β 2 X 2 = 0
• This equation represents a Line (in 2D) or a Hyperplane (in higher dimensions).
Prediction Rule:
• If β
T
X > 0 ⇒ P > 0 . 5 ⇒ ˆY = 1
• If β
T
X < 0 ⇒ P < 0 . 5 ⇒ ˆY = 0
January 20, 2026 16 / 30

### Slide 17

Visualizing the Decision Boundary
• Points on the line have 50% probability.
• Points far from the line have probabilities close to 0 or 1.
January 20, 2026 17 / 30

### Slide 18

Interpreting the Coeﬃcients: The ”Odds Ratio”
The superpower of Logistic Regression is Explainability . How do we interpret a weight
β j ?
Math Reminder:
log(Odds) = β 0 + β 1 X 1 + ...
If we increase X 1 by 1 unit:
• The Log-Odds increase by β 1 .
• The Odds are multiplied by e
β 1
(this is called the Odds Ratio ).
January 20, 2026 18 / 30

### Slide 19

Concrete Example: Titanic Interpretation
Imagine we ﬁt a model to predict survival:
Model: β gender male = − 2 . 5 , β age = − 0 . 04
1. Interpreting Gender ( β = − 2 . 5 ):
• Odds Ratio = e
− 2 . 5
≈ 0 . 08.
• ”Being male multiplies the odds of survival by 0.08.”
• (In other words: Men are ≈ 12 x less likely to survive than women, all else equal).
2. Interpreting Age ( β = − 0 . 04 ):
• Odds Ratio = e
− 0 . 04
≈ 0 . 96.
• ”For every extra year of age, the odds of survival decrease by 4%.”
January 20, 2026 19 / 30

### Slide 20

Loss: Can we use Mean Squared Error (MSE)?
The Question: Since ˆ y ∈ [0 , 1] and y ∈{ 0 , 1} , why not use the standard regression
error?
MSE =
1
n
n∑
i =1
(ˆy i − y i )
2
1. It is technically possible:
• The function is diﬀerentiable.
• Gradient descent can run on it.
2. But it is statistically incorrect:
• MSE assumes data follows a Gaussian (Normal)
distribution.
• Binary classiﬁcation instead follows a Bernoulli
distribution ( y ∼ Bernoulli ( p )).
• Maximizing Bernoulli likelihood naturally leads
to Cross-Entropy, not MSE.
Gaussian (MSE) vs Bernoulli
(LogLoss)
January 20, 2026 20 / 30

### Slide 21

Problem 1: The Vanishing Gradient (Saturation)
Why does MSE learn slowly? Let’s look at the gradients (derivative of Loss w.r.t
weights w ).
With Cross-Entropy (Log-Loss):
∇ w L CE = (ˆ y − y ) · x
If prediction is wrong (ˆ y ≈ 1 , y = 0), the
gradient is large ( ≈ 1). The model learns
fast.
With MSE:
∇ w MSE = (ˆ y − y ) · σ
′
( z )
  
Problem !
· x
Since σ
′
( z ) = ˆ y (1 − ˆy ), the gradient goes to
zero if the prediction is conﬁdent (close to 0 or
1), even if it is wrong.
Consequence: If the model is ”conﬁdently wrong” using MSE, the gradient vanishes,
and the model stops learning.
January 20, 2026 21 / 30

### Slide 22

Problem 2: Non-Convexity (Local Minima)
MSE + Sigmoid = Non-Convex
• The combination creates a ”wavy” error
surface.
• Gradient descent can get stuck in local
minima (valleys that are not the lowest
point).
Cross-Entropy + Sigmoid = Convex
• The surface is shaped like a bowl.
• Guaranteed to ﬁnd the global optimum
(minimum error).
January 20, 2026 22 / 30

### Slide 23

Optimization: The Correct Approach (MLE)
• Likelihood function (MLE): We want to maximize the probability of the observed
data.
L (β ) =
n∏
i =1
P ( y i| x i ,β ) =
n∏
i =1
ˆy
y i
i (1 − ˆy i )
1− y i
• Log-Likelihood: Easier to diﬀerentiate (turns products into sums).
ℓ (β ) =
n∑
i =1
[ y i log(ˆ y i ) + (1 − y i ) log(1 − ˆy i )]
• Cost Function (Binary Cross-Entropy):
◦ Maximizing Likelihood ⇔ Minimizing Negative Log-Likelihood.
◦ This is known as the Log-Loss :
J (β ) = −
1
n
ℓ (β ) = −
1
n
n∑
i =1
[ y i log(ˆ y i ) + (1 − y i ) log(1 − ˆy i )]
January 20, 2026 23 / 30

### Slide 24

The Bernoulli Distribution: The ”Compact” Form
The piecewise deﬁnition is clear but hard to manipulate mathematically:
P ( X = k ) =
{
p if k = 1
1 − p if k = 0
The ”Classic” One-Line Formula: To make calculus easier (especially for
Likelihood), we use this compact form:
P ( X = k ) = p
k
(1 − p )
1− k
Veriﬁcation (Why it works):
• Case k = 1: p
1
(1 − p )
1− 1
= p · 1 = p ✓
• Case k = 0: p
0
(1 − p )
1− 0
= 1 · (1 − p ) = 1 − p ✓
This formula uniﬁes both cases, making it perfect for derivatives!
January 20, 2026 24 / 30

### Slide 25

Gradient Descent for Logistic Regression
There is no closed-form solution (like in Linear Regression). We must use iterative
optimization.
• Update Rule: β new = β old − α ∇ J (β )
• Gradient: It turns out the gradient looks very similar to linear regression!
∂ J (β )
∂β j
=
n∑
i =1
(ˆy i − y i ) x ij
• Intuition:
◦ (ˆy i − y i ) is the prediction error.
◦ We adjust weights proportional to the error and the feature value.
January 20, 2026 25 / 30

### Slide 26

Critical Preprocessing: Feature Scaling
Warning: Logistic Regression uses Gradient Descent optimization.
The Problem:
• Imagine Feature A = ”Salary” (Range: 20k -
100k).
• Imagine Feature B = ”Age” (Range: 18 - 80).
• The gradients for Salary will be huge compared
to Age.
• Result: The model will struggle to converge or
will be biased towards larger numbers.
The Solution: Always apply Standardization
(StandardScaler) before training:
X scaled =
X − µ
σ
Unscaled data = Elongated valley
(slow). Scaled data = Round bowl
(fast).
January 20, 2026 26 / 30

### Slide 27

Beyond Binary: Multi-Class Classiﬁcation
What if we have more than 2 classes? (e.g., Digit Recognition 0-9, Iris Species)
Binary Classiﬁcation
• Y ∈{ 0 , 1}
• One decision boundary.
• Uses Sigmoid .
Multi-Class Classiﬁcation
• Y ∈{ 0 , 1 ,..., K − 1}
• Multiple boundaries dividing the space.
• Two main strategies:
1. One-vs-Rest (OvR)
2. Multinomial (Softmax)
January 20, 2026 27 / 30

### Slide 28

Strategy 1: One-vs-Rest (OvR)
Also called One-vs-All .
• Idea: Decompose the problem into K binary problems.
• For class k , we train a binary classiﬁer: ”Is it class k or not?” (vs all others).
• Prediction: Run all K models and pick the class with the highest probability score.
Advantage: Simple to implement with any
binary classiﬁer (Logistic, SVM, etc.).
January 20, 2026 28 / 30

### Slide 29

Strategy 2: Multinomial (Native / Softmax)
Instead of training multiple models, we generalize Logistic Regression directly to
output a probability distribution.
• Softmax Function: Generalization of Sigmoid for K classes.
P ( Y = k | X ) =
e
z k
∑ K
j =1 e
z j
• Properties:
◦ All probabilities are positive and sum to 1 (
∑
P k = 1).
◦ We minimize the Categorical Cross-Entropy loss.
Note: In sklearn , LogisticRegression uses OvR by default but can switch to
multi class=’multinomial’ .
January 20, 2026 29 / 30

### Slide 30

Python Exercise: Mastering sklearn’s LogisticRegression
Objective: Go deep into sklearn to interpret and visualize the model.
1. Data Setup:
◦ Load the iris dataset (keep only 2 classes and 2 features for 2D viz).
◦ Split data into Train/Test.
2. Model & Coeﬃcients:
◦ Fit a LogisticRegression .
◦ Retrieve .coef (β ) and .intercept (β 0 ).
◦ Write the equation of the boundary z = 0.
3. Advanced Visualization (The Goal):
◦ Scatter plot your training data.
◦ Decision Boundary: Plot the straight line separating the classes.
◦ Probability Contours: Use predict proba on a meshgrid to display the probability
gradient (colormap) in the background.
4. Analysis:
◦ Predict a new point. Compare predict() vs predict proba() .
January 20, 2026 30 / 30

## 02 Model Metrics

### Slide 1

Lecture 2: Evaluation Metrics
Alexis BOGROFF
Course on Classiﬁcation Problems @Albert School
January 21, 2026
January 21, 2026 1 / 26

### Slide 2

Introduction: You trained a model... Now what?
The Scenario: You built a Logistic Regression to detect Credit Card Fraud .
• Dataset: 1,000,000 transactions.
• Fraud rate: 0.1% (1,000 frauds, 999,000 legit).
The Result: Your model achieves 99.9% Accuracy .
Is this a good model?
January 21, 2026 2 / 26

### Slide 3

The Accuracy Paradox
The ”Dumb” Model: Imagine a model that simply predicts ”Legit” (0) for
absolutely everyone.
• Correct predictions: 999,000 (all the legit ones).
• Wrong predictions: 1,000 (all the frauds).
• Accuracy:
999 , 000
1 , 000 , 000
= 99 . 9 %.
Conclusion: In the context of Imbalanced Datasets (Fraud, Cancer, Spam),
Accuracy is a misleading metric. It rewards the model for doing nothing. We need
deeper metrics to understand ”How” the model is wrong.
January 21, 2026 3 / 26

### Slide 4

The Confusion Matrix: Dissecting Performance
Instead of a single score, we look at the 4 possible outcomes.
Predicted Class
Positive (1) Negative (0)
Actual Class
Positive (1) True Positive (TP) False Negative (FN)
Negative (0) False Positive (FP) True Negative (TN)
• TP: We predicted Fraud, and it was Fraud. (Success)
• TN: We predicted Legit, and it was Legit. (Success)
• FP: We predicted Fraud, but it was Legit. (False Alarm)
• FN: We predicted Legit, but it was Fraud. (Missed Detection)
January 21, 2026 4 / 26

### Slide 5

Statistical Terminology: Type I vs Type II Errors
In hypothesis testing (and Data Science interviews), errors have speciﬁc names:
Type I Error (False Positive)
• ”False Alarm”.
• Rejection of a true null hypothesis.
• Example: Convicting an innocent
person.
Type II Error (False Negative)
• ”Missed Detection”.
• Failing to reject a false null hypothesis.
• Example: Letting a guilty person go free.
Memorization Tip: The boy who cried wolf caused a Type I error (False Alarm) ﬁrst,
then the village committed a Type II error (Missed the wolf).
January 21, 2026 5 / 26

### Slide 6

Business Impact: The Cost Matrix
Errors are not created equal. In business, FN ̸= FP .
Example: Cancer Detection
• Cost of FP (Type I): Patient stress, cost of extra biopsy. (Moderate)
• Cost of FN (Type II): Patient dies. (Catastrophic)
Example: YouTube Copyright Filter
• Cost of FP: Creator gets angry, appeals. (Customer Service cost)
• Cost of FN: YouTube gets sued by Universal Music. (Legal cost)
As a Data Scientist, you must tune the model to minimize the Total Business Cost ,
not just maximize Accuracy.
January 21, 2026 6 / 26

### Slide 7

Precision (Positive Predictive Value)
Question: ”When the model predicts YES, how often is it right?”
Precision =
TP
TP + FP
• Focus: Quality of the positive predictions.
• High Precision means: Low False Positive rate. You rarely cry wolf.
• Use Case: Email Spam Filter .
◦ User hates checking the Spam folder.
◦ If you mark an email as Spam (Prediction=1), you better be sure.
◦ FP (Deleting a work email) is unacceptable.
January 21, 2026 7 / 26

### Slide 8

Recall (Sensitivity / True Positive Rate)
Question: ”Out of all the real YESs, how many did we ﬁnd?”
Recall =
TP
TP + FN
• Focus: Quantity / Coverage.
• High Recall means: Low False Negative rate. You rarely miss a target.
• Use Case: Terrorist Detection / Ebola Screening .
◦ You cannot aﬀord to let a single case pass through.
◦ FP (Stopping an innocent person) is annoying but acceptable.
◦ FN (Missed terrorist) is catastrophic.
January 21, 2026 8 / 26

### Slide 9

Speciﬁcity (True Negative Rate)
Often overlooked, but crucial in medicine. Question: ”Out of all the Healthy people
(Negatives), how many did we correctly identify as healthy?”
Speciﬁcity =
TN
TN + FP
The Trade-oﬀ:
• High Recall usually implies Lower Speciﬁcity.
• If you arrest everyone (Recall = 100%), your Speciﬁcity is 0% (you arrested all
innocents too).
January 21, 2026 9 / 26

### Slide 10

The Trade-oﬀ: Precision vs Recall
It is usually impossible to maximize both simultaneously.
Scenario: Strict Model
• ”Only predict 1 if probability > 90%”
• Result: High Precision , Low Recall .
• (We miss many, but the ones we ﬁnd
are sure).
Scenario: Lax Model
• ”Predict 1 if probability > 10%”
• Result: Low Precision , High Recall .
• (We ﬁnd everyone, but generate many
false alarms).
January 21, 2026 10 / 26

### Slide 11

Precision vs Speciﬁcity: The Diﬀerence in Direction
Are they the same? No. They answer two completely diﬀerent questions.
Precision (Vertical View)
• Focus: Predictions .
• Denominator: All Predicted Positives
( TP + FP ).
• ”Of all the people I arrested, how many
were guilty?”
Speciﬁcity (Horizontal View)
• Focus: Reality (Negatives) .
• Denominator: All Actual Negatives
( TN + FP ).
• ”Of all the innocent people, how many
did I let go?”
The Golden Rule:
• Recall (aka Sensitivity) is accuracy on the Actual Positives.
• Speciﬁcity is accuracy on the Actual Negatives.
• Precision is the quality of the Positive Prediction.
January 21, 2026 11 / 26

### Slide 12

Intuition: The Nightclub Bouncer
Let’s visualize the diﬀerence with a simple analogy. The Scenario: A nightclub has a
line of 100 people:
• 50 VIPs (Positives).
• 50 Non-VIPs (Negatives).
The ”Lax” Bouncer: He lets absolutely everyone in.
1. Precision (Purity inside)
• Question: ”Of the people inside, are
they mostly VIPs?”
• Calculation:
50 VIPs
100 People inside
= 50 %
• Result: Mediocre (Coin ﬂip).
2. Speciﬁcity (Filtering at the door)
• Question: ”Of the Non-VIPs, did you
stop them?”
• Calculation:
0 Stopped
50 Non-VIPs
= 0 %
• Result: Catastrophic (No ﬁlter).
Conclusion: Precision measures the outcome quality , Speciﬁcity measures the process
quality (the ﬁlter).
January 21, 2026 12 / 26

### Slide 13

Combining them: The F1-Score
We need a single metric to ﬁnd the balance. Why not the arithmetic mean
P + R
2
?
• Example: Precision=1.0, Recall=0.0 (Useless model). Arithmetic Mean = 0.5.
(Misleading)
The Harmonic Mean (F1-Score):
F 1 = 2 ×
Precision × Recall
Precision + Recall
• Properties: If either P or R is low, the F1-score crashes close to 0.
• It forces the model to be good at both .
January 21, 2026 13 / 26

### Slide 14

Deriving the F1-Score
Deﬁnition of Harmonic Mean: The inverse of the average of the inverses .
H =
1
Arithmetic Mean(
1
P
,
1
R
)
Step-by-Step Derivation:
1. Average the inverses:
1
P
+
1
R
2
=
P + R
P · R
2
=
P + R
2 · P · R
2. Take the inverse of the result to get F1:
F 1 =
1
P + R
2 · P · R
=
2 · P · R
P + R
January 21, 2026 14 / 26

### Slide 15

Deep Dive: Why do we invert the numbers?
The Harmonic Mean is mandated by math whenever you average Ratios with a Fixed
Numerator .
1. The Analogy: Speed
Speed =
Distance (Fixed)
Time (Variable)
To average speed over a ﬁxed distance, you must average the Times (the
denominators). Since Time =
D
Speed
, you are averaging the inverses.
2. The F1-Score Context
Precision =
TP
Predicted Volume
Recall =
TP
Actual Volume
• Both metrics share the same goal: capturing TP .
• We want to average the ”Volume required to get 1 TP” (the inverse).
• Result: Averages of inverses → Harmonic Mean.
January 21, 2026 15 / 26

### Slide 16

Advanced: The F-Beta Score
What if Business says: ”Recall is 2x more important than Precision”? We use the
generalized F β score:
F β = (1 + β
2
) ×
Precision × Recall
(β 2 × Precision) + Recall
Strategic Choice of β :
• β = 1 : Standard F1 (Balance).
• β = 0 . 5 : Focus on Precision (e.g., Spam, Recommender Systems).
• β = 2 : Focus on Recall (e.g., Cancer, Fraud).
January 21, 2026 16 / 26

### Slide 17

Deriving F-Beta Score
The Confusion: If β = 2 means ”Recall is important”, why does the formula have
β
2
× Precision?
F β = (1 + β
2
)
Precision × Recall
(β 2 × Precision ) + Recall
The Explanation (Cross-Multiplication): F β is a weighted Harmonic Mean. We
apply the weight β
2
to the inverse of Recall (
1
R
).
Weighted Sum =
1
P
+ β
2
·
1
R
To add these fractions, we must cross-multiply:
1 · R + β
2
· P
P · R
Algebraically, the weight of Recall ends up multiplying Precision because of the common
denominator.
January 21, 2026 17 / 26

### Slide 18

The Decision Threshold
Logistic Regression outputs a probability P ∈ [0 , 1]. We need a binary decision
ˆy ∈{ 0 , 1} .
ˆy = 1 ⇐⇒ P > Threshold
• Default Threshold: 0.5.
• Moving the Threshold:
◦ Increase Threshold (e.g., 0.8) → Harder to predict 1 → Precision ↑ , Recall ↓ .
◦ Decrease Threshold (e.g., 0.2) → Easier to predict 1 → Precision ↓ , Recall ↑ .
We need a way to evaluate the model regardless of the threshold choice.
January 21, 2026 18 / 26

### Slide 19

ROC Curve (Receiver Operating Characteristic)
We plot the performance for ALL possible thresholds .
• X-Axis: False Positive Rate (1 − Speciﬁcity). ”Risk of False Alarm”.
• Y-Axis: True Positive Rate (Recall). ”Ability to Detect”.
• Top-Left Corner: Ideal (TPR=1, FPR=0).
• Diagonal Line: Random Guessing. January 21, 2026 19 / 26

### Slide 20

AUC (Area Under the Curve)
How do we compare two ROC curves? We calculate the Area Under the Curve.
• AUC = 0.5: Random model (no discrimination capacity).
• AUC = 1.0: Perfect model.
• Typical Good Model: 0.8 - 0.9.
Probabilistic Interpretation (Deep Dive): AUC is the probability that the model
ranks a randomly chosen Positive instance higher than a randomly chosen Negative
instance.
AUC = P (score( x
+
) > score( x
−
))
It measures the ranking quality, not the calibration.
January 21, 2026 20 / 26

### Slide 21

The Blind Spot of AUC: Ranking vs Calibration
AUC only cares about Order . It does not care about the Value .
Comparison: Two models predicting the same Fraud ( Y = 1) vs Legit ( Y = 0).
Model Score(Fraud) Score(Legit) Check ( > ) AUC Contribution
Model A (Conﬁdent) 0.95 0.05 Yes +1 Point
Model B (Compressed) 0.51 0.49 Yes +1 Point
The Consequence:
• Both models have the same AUC (Perfect Ranking).
• Model A is well calibrated: 0.95 means ”High Danger”.
• Model B is poorly calibrated: 0.51 looks like a ”Coin Flip”, even though it correctly
identiﬁed the fraud.
AUC tells you who is riskiest. Calibration tells you how risky they are.
January 21, 2026 21 / 26

### Slide 22

ROC vs Precision-Recall (PR) Curve
ROC Curve
• Uses TPR and FPR.
• Robust to imbalance: Does not
change if we add 1M negatives.
• Use when negatives are roughly equal to
positives, or when TN matters.
Precision-Recall Curve
• Uses Precision and Recall.
• Sensitive to imbalance: If class 1 is
rare, Precision drops fast.
• Gold Standard for highly imbalanced
data (Fraud, Anomaly detection).
January 21, 2026 22 / 26

### Slide 23

How to Read a ROC Curve (Beneﬁt vs Cost)
The ROC Curve tells the story of a trade-oﬀ: ”How much noise must I accept to ﬁnd
the signal?”
The Axes:
• Y-Axis (Beneﬁt): True Positive Rate (Recall). We want this to go up fast.
• X-Axis (Cost): False Positive Rate (Risk of False Alarm). We want this to
stay low.
The Ideal Shape (Top-Left):
• Ideally, the curve shoots straight up (ﬁnding positives) before moving right
(making mistakes).
• AUC = 1.0: The model creates a perfect separation. It ﬁnds all positives
before making a single mistake.
The Baseline (Diagonal): Unlike PR, the ”Random” baseline is always the diagonal
( y = x , AUC=0.5). If your curve touches this line, your model provides zero
information.
January 21, 2026 23 / 26

### Slide 24

How to Read a Precision-Recall Curve
Unlike ROC (where we aim Top-Left), here we aim for the Top-Right .
The Narrative: As we move Right (increasing Recall/Coverage), does the
Precision (Quality) stay high, or does it crash?
Key Landmarks:
• Top-Right Corner (1,1): The ”Holy Grail”. Perfect Recall with Perfect
Precision.
• The ”Cliﬀ”: The point where Precision starts to drop signiﬁcantly. We
want to push this cliﬀ as far right as possible.
• The Baseline (Random): A horizontal line at y =
Positives
Total
. (e.g., 0.01 for
Fraud).
Comparison Metric: AUC-PR (Average Precision).
• 1.0 = Perfect.
• Baseline = % of Positives (e.g. 0.01). Beating this is the real test.
January 21, 2026 24 / 26

### Slide 25

From Binary to Multi-Class Metrics
How do we calculate Precision/Recall if we have 3 classes (A, B, C)? We compute the
metric for each class individually (One-vs-Rest), then average them. Two ways to
average:
1. Macro Average
• Calculate F1 for A, F1 for B, F1 for C.
• Take the simple average.
• Eﬀect: Treats all classes equally. Useful
if you care about the minority class
performance.
2. Weighted Average
• Average weighted by the number of true
instances in each class.
• Eﬀect: Dominated by the majority
class. Closer to ”Global Accuracy”.
Trap: In imbalanced multi-class, ’Weighted’ can hide poor performance on rare classes. Use
’Macro’ to be strict.
January 21, 2026 25 / 26

### Slide 26

Exercise: Strategic Evaluation
Dataset: Credit Card Fraud (Highly Imbalanced).
1. Baseline: Train a Logistic Regression. Calculate Accuracy. (Expected: ≈ 99%).
2. Real Metrics: Calculate the Confusion Matrix, Precision, Recall, and F1.
3. Threshold Tuning:
◦ Predict probabilities ( .predict proba ).
◦ Create a loop to calculate F1-score for thresholds from 0.1 to 0.9.
◦ Find the threshold that maximizes the F1-score.
4. Visualization: Plot the ROC Curve and calculate AUC.
5. Business Decision:
◦ Assume Cost(FN) = $ 1000 (Stolen money) and Cost(FP) = $ 10 (Client call).
◦ Find the threshold that minimizes Total Cost.
January 21, 2026 26 / 26

## 03 Decision Trees

### Slide 1

Lecture 3: Decision Trees
Alexis BOGROFF
Course on Classiﬁcation Problems @Albert School
January 28, 2026
January 28, 2026 1 / 30

### Slide 2

Overview
Introduction to Decision Trees
CART Algorithm
January 28, 2026 2 / 30

### Slide 3

What is a Decision Tree?
• A supervised learning algorithm used for classiﬁcation and regression tasks.
• Mimics human decision-making (yes/no).
• Represents data in a tree-like structure.
• Splits are computed recursively.
Introduction to Decision Trees January 28, 2026 3 / 30

### Slide 4

Types of Decision Trees
• Classiﬁcation Trees : Output is a category.
• Regression Trees : Output is a continuous value.
Introduction to Decision Trees January 28, 2026 4 / 30

### Slide 5

Structure & Terminology
To speak the language of Decision Trees, we must name the parts:
• Root Node : The top node representing the entire population. It gets divided into
two or more sets.
• Internal Node (or Decision Node): A sub-node that splits into further sub-nodes.
It represents a decision rule (e.g., Age > 30).
• Branch : A subsection of the entire tree.
• Leaf Node (or Terminal Node): Nodes that do not split. They contain the ﬁnal
output (Class prediction or Regression value).
Introduction to Decision Trees January 28, 2026 5 / 30

### Slide 6

CART Algorithm and Gini Index
• The CART algorithm underlies modern decision tree implementations.
• It uses the Gini Index to measure impurity in a dataset and performs binary splits.
• For classiﬁcation, given a dataset D with c classes:
◦ p i is the probability of an instance belonging to class i :
p i =
| D i|
| D |
Gini ( D ) = 1 −
c∑
i =1
p
2
i
• A lower Gini Index indicates a purer node, meaning it predominantly contains
instances of a single class.
CART Algorithm January 28, 2026 6 / 30

### Slide 7

Gini Split
• For a general computation of Gini Index, when a dataset is split into k child nodes
D j , we compute the weighted sum of the Gini Indices:
Gini split =
k∑
j =1
| D j|
| D |
Gini ( D j )
• Since CART implements a binary split there are only two partitions at each split,
i.e. k = 2, so the formula simplies to:
Gini split =
| D 1|
| D |
Gini ( D 1 ) +
| D 2|
| D |
Gini ( D 2 )
• With | D j| the number of instances in subset D j .
• The optimal split is the one that results in the lowest Gini split .
CART Algorithm January 28, 2026 7 / 30

### Slide 8

Alternative Criterion: Entropy & Information Gain
• While CART uses Gini, other algorithms (ID3, C4.5) use Entropy .
• Entropy measures the disorder or uncertainty in the data.
H ( D ) = −
c∑
i =1
p i log 2 ( p i )
• Information Gain (IG) : We choose the split that reduces Entropy the most.
IG ( D , split) = H ( D ) −
k∑
j =1
| D j|
| D |
H ( D j )
• Comparison :
◦ Gini is computationally faster (no log to compute).
◦ Entropy tends to produce slightly more balanced trees.
◦ In practice: Results are often very similar (95% agreement).
CART Algorithm January 28, 2026 8 / 30

### Slide 9

Example: Computing Gini Index
• Consider a dataset with 10 instances
and two classes: A and B.
• Before splitting:
Gini ( D ) = 1 − ( p
2
A + p
2
B )
= 1 −
( (
5
10
) 2
+
(
5
10
) 2
)
= 1 − (0 . 5
2
+ 0 . 5
2
)
= 1 − (0 . 25 + 0 . 25)
= 0 . 5
Feature X Class
5.1 A
4.9 A
5.5 A
6.1 B
5.8 B
6.2 B
5.7 B
5.9 A
6.3 B
6.5 A
CART Algorithm January 28, 2026 9 / 30

### Slide 10

How Are Thresholds Deﬁned in CART?
Step 1: Sort Unique Values of the Feature
• Given feature values: [5 . 1 , 4 . 9 , 5 . 5 , 6 . 1 , 5 . 8 , 6 . 2 , 5 . 7 , 5 . 9 , 6 . 3 , 6 . 5]
• Sort in ascending order: [4 . 9 , 5 . 1 , 5 . 5 , 5 . 7 , 5 . 8 , 5 . 9 , 6 . 1 , 6 . 2 , 6 . 3 , 6 . 5]
Step 2: Compute Midpoints Between Consecutive Values
• Possible split points:
4 . 9 + 5 . 1
2
,
5 . 1 + 5 . 5
2
,
5 . 5 + 5 . 7
2
,...
= 5 . 0 , 5 . 3 , 5 . 6 , 5 . 75 , 5 . 85 , 6 . 0 , 6 . 15 , 6 . 25 , 6 . 4
Step 3: Evaluate Gini Index for Each Split
• Split dataset at each midpoint and compute Gini split .
• The split that minimizes impurity is chosen.
CART Algorithm January 28, 2026 10 / 30

### Slide 11

Example: Computing Gini a Poor Gini Split
• Let’s consider a split at 5.85
Left node (X ≤ 5 . 85 ) :
• 3 instances of A, 2 of B.
• Gini ( Left ) = 1 − (0 . 6
2
+ 0 . 4
2
)
• = 0 . 48
Right node (X > 5 . 8 ) :
• 2 instances of A, 3 of B.
• Gini ( Right ) = 1 − (0 . 4
2
+ 0 . 6
2
)
• = 0 . 48
Compute weighted Gini Split:
Gini split = (0 . 5 × 0 . 48) + (0 . 5 × 0 . 48) = 0 . 48
• This split does not signiﬁcantly reduce impurity.
CART Algorithm January 28, 2026 11 / 30

### Slide 12

Example: Computing the Best Gini Split
• Let’s consider a split at X = 5 . 6
Left node (X ≤ 5 . 6 ) :
• 3 instances of A, 0 of B.
• Gini ( Left ) = 0 . 0 (pure node)
Right node (X > 5 . 5 ) :
• 2 instances of A, 5 of B.
• Gini ( Right ) = 1 − (0 . 285
2
+ 0 . 714
2
)
• Gini ( Right ) = 0 . 408
Compute weighted Gini Split:
Gini split = (0 . 3 × 0) + (0 . 7 × 0 . 408) = 0 . 286
• This split signiﬁcantly reduces impurity, leading to better classiﬁcation.
CART Algorithm January 28, 2026 12 / 30

### Slide 13

The ”Greedy” Nature of CART
• How does the tree decide the perfect structure?
• Recursive Partitioning : It builds the tree step-by-step, splitting data into smaller
subsets recursively.
• Greedy Algorithm :
◦ At each step, the algorithm makes the locally optimal choice (best split now ).
◦ It does not look ahead to see if a slightly worse split now would lead to a much better
split later.
◦ Consequence: Decision Trees might not ﬁnd the globally optimal tree, but they are fast
and eﬀective.
CART Algorithm January 28, 2026 13 / 30

### Slide 14

Mathematical Formulation for Regression Trees
• Unlike classiﬁcation trees, regression trees predict continuous values.
• Instead of Gini impurity, they minimize Mean Squared Error (MSE) at each split:
MSE ( t ) =
1
| D t|
∑
i∈ D t
( y i − ¯y t )
2
• Where:
◦ y i is the actual value of instance i .
◦ ¯y t is the mean value of instances in node t .
• The best split minimizes the weighted sum of the MSE in child nodes, thereby
reducing prediction error.
CART Algorithm January 28, 2026 14 / 30

### Slide 15

Feature Importance in Regression Trees
• Feature importance is thus based on MSE reduction instead of impurity reduction.
• This method is known as Mean Decrease in Impurity (MDI) . The score is
computed as:
I ( f ) =
∑
t
∆ MSE ( t ) × P ( t )
• Where:
◦ ∆ MSE ( t ) is the reduction in Mean Squared Error at node t .
◦ P ( t ) is the probability of reaching node t .
CART Algorithm January 28, 2026 15 / 30

### Slide 16

Example: Splitting in a Regression Tree
• Consider a dataset predicting Parisian house prices based on square meters (m ² ).
• Let’s split at 55m ²
Size (m ² ) Price (EUR)
45 600,000
50 650,000
55 700,000
65 800,000
70 850,000
• Mean Squared Error (MSE) before the split:
8 , 600 , 000 , 000 EUR
2
• MSE after the split:
500 , 000 , 000 EUR
2
Left Node ( ≤ 55 ) :
• 45m ² : 600,000 EUR
• 50m ² : 650,000 EUR
• Mean Prediction: 625,000 EUR
Right Node ( > 55 ) :
• 65m ² : 800,000 EUR
• 70m ² : 850,000 EUR
• Mean Prediction: 825,000 EUR
CART Algorithm January 28, 2026 16 / 30

### Slide 17

Pre-Pruning
• Decision trees can be pruned to prevent overﬁtting.
• Two main types of pruning exist:
Pre-Pruning (Early Stopping)
• Stops tree growth early before reaching full depth.
• Limits complexity by using hyperparameters like:
◦ max depth : Maximum depth of the tree.
◦ min samples split : Minimum samples needed to split a node.
◦ min samples leaf : Minimum samples required at a leaf.
◦ max leaf nodes : Limits the number of leaf nodes.
• Risk: Can lead to underﬁtting if stopped too early.
CART Algorithm January 28, 2026 17 / 30

### Slide 18

Post-Pruning: Cost Complexity Pruning (CCP)
Post-Pruning
• CCP is a post-pruning technique applied after the tree is fully grown.
• It removes unnecessary branches to reduce complexity.
• The cost complexity function is:
R α ( T ) = R ( T ) + α | T |
where:
◦ R ( T ) is the impurity (e.g., Gini, MSE).
◦ | T | is the number of leaf nodes.
◦ α is the complexity parameter controlling pruning strength.
CART Algorithm January 28, 2026 18 / 30

### Slide 19

Applying CCP in Scikit-Learn
• CCP is not applied by default (‘ccp alpha=0.0‘).
• Steps to apply CCP:
1. Train a full decision tree.
2. Train trees for diﬀerent α values and choose the best one
1
Example Code:
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(ccp alpha=0.01)
clf.fit(X train, y train)
1
To go further with a clever selection of α values, see https://scikit-learn.org/stable/
auto examples/tree/plot cost complexity pruning.html
CART Algorithm January 28, 2026 19 / 30

### Slide 20

Feature Importance in Decision Trees
• Importance scores are assigned to features based on how much they reduce impurity.
• It corresponds to how often the feature was used and how strong was the impurity
reduction each time the feature was used.
• The importance score for a feature f is computed as:
I ( f ) =
∑
t∈ nodes where f is used
∆ Gini ( t ) × P ( t )
Where:
• ∆ Gini ( t ) is the impurity reduction at node t :
∆ Gini ( t ) = Gini ( t ) −
(
| D L|
| D |
Gini ( D L ) +
| D R |
| D |
Gini ( D R )
)
• P ( t ) is the probability of reaching node t :
P ( t ) =
| D t|
| D total |
CART Algorithm January 28, 2026 20 / 30

### Slide 21

Interpreting Feature Importance
• Feature importance helps in feature selection and model interpretability:
• Higher score I ( f ) → Feature is used frequently and signiﬁcantly reduces impurity
→ More important.
• Lower score I ( f ) → Feature contributes little to reducing impurity
→ Less important.
Extracting Feature Importance in Scikit-Learn:
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X train, y train)
importance scores = clf.feature importances
print(importance scores)
CART Algorithm January 28, 2026 21 / 30

### Slide 22

Evaluating Decision Trees
• Classiﬁcation
◦ Use Confusion Matrix to analyze predictions.
◦ Compute accuracy, precision, recall, and F1-score.
• Regression
◦ Use Mean Squared Error (MSE) to measure error.
◦ Use R
2
Score to assess model goodness-of-ﬁt.
CART Algorithm January 28, 2026 22 / 30

### Slide 23

Limitations of Decision Trees
Key Limitations:
• Overﬁtting: Decision trees tend to overﬁt if not pruned or regularized properly.
• Instability: Small changes in data can lead to a completely diﬀerent tree structure.
• Lack of Extrapolation: Decision trees cannot extrapolate beyond the training data
range.
◦ Example: A regression tree trained on house prices between 30m ² and 110m ² cannot
predict for 150m ² accurately.
• Biased towards dominant classes: If class imbalance exists, trees may favor the
majority class.
• High variance: Sensitive to the training set, requiring ensemble methods (Random
Forests, Boosting) for stability.
CART Algorithm January 28, 2026 23 / 30

### Slide 24

Visualizing the Decision Surface
• Orthogonal Boundaries :
◦ Logistic Regression draws diagonal lines
(linear combinations of features)
◦ Decision Trees draw boundaries
perpendicular to the feature axes .
• Creates ”box-like” / ”step-wise”
decision regions.
• Implication : To approximate a diagonal
boundary, a Decision Tree needs many
splits (staircase eﬀect), which increases
complexity.
CART Algorithm January 28, 2026 24 / 30

### Slide 25

Exercise 1: Understanding Decision Tree Splits
Task: Given the following dataset, determine the best split using Gini Index.
Feature X Class
2.5 A
3.0 A
3.5 B
4.0 B
4.5 A
5.0 B
Questions:
• Compute the Gini Index before splitting.
• Evaluate possible splits and compute Gini Split.
• Determine the best split.
CART Algorithm January 28, 2026 25 / 30

### Slide 26

Exercise 2: Constructing a Regression Tree
Task: Build a regression tree using the following data (predict house prices in Paris).
Size (m ² ) Price (EUR)
30 400,000
50 600,000
70 800,000
90 1,000,000
110 1,200,000
Questions:
• Compute the MSE before splitting.
• Find the optimal split by minimizing MSE.
• Predict prices for new data points using your tree.
CART Algorithm January 28, 2026 26 / 30

### Slide 27

Exercise 3: Decision Tree Hyperparameters
Task: Investigate the impact of hyperparameters on decision trees.
• Train a Decision Tree Classiﬁer on a dataset.
• Display the tree structure using sklearn ’s plot tree method.
• Experiment with diﬀerent values for:
◦ max depth
◦ min samples split
◦ min samples leaf
• Observe how tree structure and accuracy change.
Questions:
• How does increasing max depth aﬀect overﬁtting?
• What happens when min samples leaf is too high?
• How do you determine the best hyperparameters?
CART Algorithm January 28, 2026 27 / 30

## 04 Random Forest

### Slide 1

Lecture 4: Random Forests
Alexis BOGROFF
Course on Supervised Learning Problems @Albert School
January 28, 2026
January 28, 2026 1 / 26

### Slide 2

The Philosophy: Wisdom of Crowds
• Core Idea: A large group of
average predictors is smarter than a
single expert.
• Condorcet’s Jury Theorem: ”If it
is more probable that each of the
voters will judge in conformity with
the truth, the more the number of
voters increases, the greater the
probability of the truth of the
decision will be.”
 Even slightly better-than-random classiﬁers achieve
near-perfect accuracy when combined in large
numbers.
Ensemble Learning Foundations January 28, 2026 2 / 26

### Slide 3

Why Ensembles Work: Condorcet’s Jury Theorem
The Theorem relies on two explicit conditions:
1. Competence ( p > 0 . 5 ): Each voter (tree) must have a probability of being correct
greater than random guessing.
2. Independence: Voters must be independent; their errors must be uncorrelated .
3. Shared goal .
The Mathematical Result:
• As the number of voters N →∞ , the probability that the majority vote is correct
approaches 1 .
Implication: We need many trees ( N ) that are diverse (Independent) and decent
( p > 0 . 5).
Ensemble Learning Foundations January 28, 2026 3 / 26

### Slide 4

Weak Learners vs. Strong Learners
• Weak Learner:
◦ A model that performs slightly better than random guessing.
◦ Example: A shallow Decision Tree (Stump), or a tree with high variance.
• Strong Learner:
◦ A model that can achieve arbitrarily high accuracy.
◦ Example: A Random Forest (an ensemble of weak learners).
• Goal of Ensemble Methods: Combine multiple weak learners to build a strong
learner.
Ensemble Learning Foundations January 28, 2026 4 / 26

### Slide 5

Types of Ensemble Methods
• Bagging (Bootstrap Aggregating):
◦ Parallel training. Independent models. Reduces Variance.
◦ Example: Random Forest.
• Boosting:
◦ Sequential training. Models learn from previous mistakes. Reduces Bias.
◦ Example: AdaBoost, XGBoost (Lecture 6).
• Stacking:
◦ Meta-learning. A ”ﬁnal” model learns how to combine predictions from diﬀerent base
models (e.g., Linear Reg + KNN + Tree).
Ensemble Learning Foundations January 28, 2026 5 / 26

### Slide 6

Theory: Why Averaging Reduces Variance?
• Intuition: Errors cancel each other out. If one tree overestimates and another
underestimates, the average is closer to the truth.
• The Math:
◦ If we have B independent random variables (trees) each with variance σ
2
:
Var(Mean) =
σ
2
B
◦ Impact: As B increases, variance shrinks to 0 (if trees are independent).
• The Reality (Trees are correlated):
Var(Forest) = ρσ
2
+
1 − ρ
B
σ
2
• Because trees share training data, they are correlated ( ρ> 0).
• Random Forest Strategy: By using random features, we lower ρ (correlation),
maximizing the variance reduction.
Ensemble Learning Foundations January 28, 2026 6 / 26

### Slide 7

Introduction to Random Forests
• Random Forest is an ensemble learning method.
• It builds multiple decision trees and combines their outputs.
• Used for both classiﬁcation and regression tasks.
• Helps reduce overﬁtting compared to a single decision tree.
• Model from Breiman’s original paper (2001)
Ensemble Learning Foundations January 28, 2026 7 / 26

### Slide 8

Diﬀerence Between a Decision Tree and Random Forest
1. Bootstrapping (Sampling with Replacement)
◦ Generates multiple datasets of the same size as the original.
◦ Each sample contains about 63.2% of unique data points.
2. Training Multiple Decision Trees
◦ Each tree is trained independently on a diﬀerent bootstrap sample.
3. Trees use a Random Set of Feature at Each Split
◦ Instead of using all features, each split considers a random subset .
◦ Classiﬁcation: Uses m =
√
d features.
◦ Regression: Uses m = d / 3 features.
4. Aggregation of Predictions
◦ Classiﬁcation: Uses majority voting among trees.
◦ Regression: Averages predictions from all trees.
Ensemble Learning Foundations January 28, 2026 8 / 26

### Slide 9

Bagging: Bootstrap Aggregating
Bagging stands for B ootstrap Agg regat ing .
• Objective: Reduce variance and avoid overﬁtting.
• The Process:
1. Given a dataset D = { ( x 1 , y 1 ),..., ( x n , y n )} of size n .
2. Create B new datasets D
1
, D
2
,..., D
B
(also of size n ).
3. Sampling with Replacement: An observation can appear multiple times in the same
sample D
b
.
• Result: We obtain B slightly diﬀerent training sets, leading to B diverse decision
trees.
Ensemble Learning Foundations January 28, 2026 9 / 26

### Slide 10

The Mathematics of Bootstrap
Question: If we sample n times with replacement from n items, what is the probability
that a speciﬁc item x i is never picked?
• Probability of NOT picking x i in one draw: 1 −
1
n
• Probability of NOT picking x i in n draws (independent):
P (Not Selected) =
(
1 −
1
n
) n
• As n →∞ , this limit converges to 1 / e :
lim
n→∞
(
1 −
1
n
) n
= e
− 1
≈ 0 . 368
• Conclusion:
◦ ≈ 36.8% of data is left out (Out-of-Bag).
◦ ≈ 63.2% of data is used for training the tree.
Ensemble Learning Foundations January 28, 2026 10 / 26

### Slide 11

Out-of-Bag (OOB) Validation
What do we do with the 36.8% left out?
• For each tree, the unused samples are called Out-of-Bag (OOB) samples.
• They act as a Natural Validation Set . No need for Cross-Validation!
Computing the OOB Score:
1. For each instance x i in the original dataset, identify all trees that did not see x i
during training.
2. Aggregate the predictions of these speciﬁc trees (Majority Vote or Average).
3. Compare this aggregated prediction to the true label y i to calculate the error.
Note: In Scikit-Learn, set oob score=True to compute this automatically.
Ensemble Learning Foundations January 28, 2026 11 / 26

### Slide 12

What is Aggregating
Aggregating
• This aggregation is the mechanism to compute the overall prediction of the model
by combining the independent predictions of each tree.
• It reduces variance over a simple Tree model by averaging multiple independent
models (same than ﬁnancial porfolio risk minimisation).
• Classiﬁcation: Majority Voting.
ˆy = arg max
k
B∑
b =1
1( y b = k )
• Regression: Averaging.
ˆy =
1
B
B∑
b =1
y b
Ensemble Learning Foundations January 28, 2026 12 / 26

### Slide 13

Out-of-Bag (OOB) Validation
• Since we use Bootstrapping, approx. 37% of instances are left out for each tree.
• These are called Out-of-Bag (OOB) samples.
• OOB Score Mechanism:
1. For every instance x i , predict its label using only the trees that did NOT see x i during
training.
2. Aggregate these predictions to get the OOB prediction.
3. Compute accuracy/error on these predictions.
• Advantage: It acts as a free validation set! No need to do Cross-Validation (saving
time).
Ensemble Learning Foundations January 28, 2026 13 / 26

### Slide 14

Decision Tree Construction in Random Forest
Splitting Criteria:
• Reminder of metrics:
◦ For Classiﬁcation: Gini Impurity G ( X ) = 1 −
∑ C
i =1 p
2
i
◦ For Regression: Mean Squared Error MSE =
1
n
∑ n
i =1 ( y i − ¯y )
2
• Random Feature Selection:
◦ Each node considers only a random subset of m features.
◦ This reduces the correlation between trees, improving generalization.
◦ For Classiﬁcation: m =
√
d
◦ For Regression: m =
d
3
Ensemble Learning Foundations January 28, 2026 14 / 26

### Slide 15

The Secret Sauce: Decorrelation
• Problem with simple Bagging:
◦ If there is one very strong feature in the dataset, all trees will likely choose it for the
ﬁrst split.
◦ Result: All trees look similar → Predictions are correlated → Variance is not reduced
eﬀectively.
• Random Forest Solution (Feature Randomness):
◦ At each split, we force the tree to choose from a random subset of features.
◦ This forces trees to consider other features, creating diversity.
◦ Decorrelated trees = Better aggregate performance.
Ensemble Learning Foundations January 28, 2026 15 / 26

### Slide 16

Feature Importance: Mean Decrease in Impurity (MDI)
• Intuition: If a feature is useful, it will be selected often for splits, and these splits
will signiﬁcantly reduce impurity (Gini or MSE).
• Step 1: Single Tree Importance For a single tree T , the importance of feature j
is the sum of impurity decreases weighted by the number of samples at that node:
I j ( T ) =
∑
t∈ T , v ( t )= j
N t
N
· ∆Impurity( t )
• Step 2: Forest Importance (The ”Mean” part) In a Random Forest with B
trees, we average this score:
Importance j =
1
B
B∑
b =1
I j ( T b )
• Normalization: Scores are typically normalized so they sum to 1 (or 100).
Ensemble Learning Foundations January 28, 2026 16 / 26

### Slide 17

Interpreting Feature Importance & Pitfalls
Interpretation:
• Features with high scores ”drive” the prediction.
• Useful for Feature Selection (dropping useless
variables).
Warning: The Bias of MDI
• MDI is biased towards High Cardinality
Features (numerical features with many unique
values or categorical features with many
categories).
• Reason: These features oﬀer many potential split
points, are selected more often by pure chance.
• Alternative: Use Permutation Importance
(model-agnostic) to verify results if high
cardinality features are present.
Ensemble Learning Foundations January 28, 2026 17 / 26

### Slide 18

The Random Forest Algorithm: Step-by-Step
1. Setup: Choose number of trees B (e.g., 100) and feature subset size m .
2. Training Loop: For b = 1 to B :
◦ Draw a bootstrap sample D
b
of size N from the training data.
◦ Grow a Decision Tree T b on D
b
.
◦ At each node split:
▶ Select m features at random from the total d features.
▶ Pick the best split among these m features (Gini/MSE).
▶ Split the node and recurse until a stopping criterion is met (e.g., min leaf size).
3. Prediction:
◦ Pass the new instance x down all B trees.
◦ Regression: Average the outputs: ˆ y =
1
B
∑
ˆy b .
◦ Classiﬁcation: Majority Vote.
Ensemble Learning Foundations January 28, 2026 18 / 26

### Slide 19

Critical Hyperparameters (Scikit-Learn)
• n estimators :
◦ Number of trees in the forest.
◦ Rule: More is better (reduces variance), but slower. Value stabilizes after a point (e.g.,
100 or 500).
• max features :
◦ Size of the random subset of features at each split.
◦ Defaults:
√
d (classiﬁcation) or d (regression). Smaller = more diversity but more bias.
• max depth / min samples leaf :
◦ Controls the complexity of individual trees (Pre-pruning).
Ensemble Learning Foundations January 28, 2026 19 / 26

### Slide 20

Implementation: Parallelization
• Random Forest is ”embarrassingly parallel”.
• Since trees are independent, can be built simultaneously on diﬀerent CPU cores.
• In Scikit-Learn:
◦ Set n jobs=-1 to use all available cores.
◦ Drastically reduces training time compared to Boosting (which is sequential).
Ensemble Learning Foundations January 28, 2026 20 / 26

### Slide 21

Practical Implementation with Scikit-Learn
• Key Hyperparameters:
◦ n estimators : Number of trees (start with 100, then increase).
◦ max features :
▶ "sqrt" : Standard for Classiﬁcation ( m =
√
d ).
▶ "log2" : Alternative for very high-dimensional data ( m = log 2 ( d )).
◦ n jobs : Set to -1 to use all CPU cores (Parallelization).
Python Example
from sklearn.ensemble import RandomForestClassifier
# Initialize with parallel processing
rf = RandomForestClassifier(
n_estimators=100,
max_features=’sqrt’, # or ’log2’
max_depth=None, # Fully grown trees
n_jobs=-1, # Use all cores
random_state=42
)
rf.fit(X_train, y_train)
print(rf.feature_importances_)
Ensemble Learning Foundations January 28, 2026 21 / 26

### Slide 22

Where Random Forest Shines
Conditions for Optimal Performance:
• The data is structured (tabular) .
• There are many features , which may have weak individual predictive power.
• There are complex feature interactions , but deep learning would be overkill.
• The data is of small to medium size .
• The data is noisy (RF handles noise well via bagging).
Ensemble Learning Foundations January 28, 2026 22 / 26

### Slide 23

Random Forest and Dataset Size
Best Suited for Small to Medium-Sized Datasets:
• Scalability and Training Time: Handles moderately large datasets, but
computational and memory demands increase signiﬁcantly with size.
• Interpretability: Ideal for smaller datasets where feature importance scores provide
actionable insights.
• Data Requirements: Performs well with limited data by combining weak signals
into strong predictions, unlike deep learning which requires large datasets.
Caution:
• For very large datasets, consider alternatives like Gradient Boosted Trees (e.g.,
XGBoost, LightGBM) or Deep Learning .
Ensemble Learning Foundations January 28, 2026 23 / 26

### Slide 24

Why Businesses Rarely Use Random Forest in Production
Challenges in Business Adoption:
• Interpretability Requirements: Businesses prefer models like Logistic Regression
that provide clear decision rules.
• Computational Costs: Random Forest is slow for large datasets, making XGBoost
or LightGBM better alternatives.
• Deployment Complexity: Requires more memory and processing power compared
to simpler models.
• Regulatory Concerns: Industries like ﬁnance and healthcare need transparent
models to comply with regulations.
• Preference for Simplicity: Businesses often prioritize models that are easy to
maintain and explain to stakeholders.
Ensemble Learning Foundations January 28, 2026 24 / 26

### Slide 25

Where Random Forest is Most Practical
RF is commonly used in:
• Exploratory Analysis: Identifying important features before using simpler models.
• Research and Prototyping: Initial modeling before transitioning to more
interpretable models for production.
• Backend Systems: Fraud detection, risk assessment, and anomaly detection where
explanations are less critical.
Takeaway: RF is powerful but often replaced by simpler or more eﬃcient models in
production settings.
Ensemble Learning Foundations January 28, 2026 25 / 26

### Slide 26

Comparison with Previous Classiﬁcation Models
Key Comparison:
Aspect Logistic Regression KNN Random Forest
Data Type Num/Categ Num/Categ Num/Categ
Interpretability High Low Medium
Overﬁtting Risk Low if Regularized Depends on K Low
Computational Cost Low High (Inference) Medium (Training)
Handles Missing Data No No Yes
Non-Linearity No Yes Yes
Large Datasets Good Bad Average
Feature Importance Yes No Yes
Ensemble Learning Foundations January 28, 2026 26 / 26

