# Temporal Hierarchical Neural Network with LSTM

## Overview

The integrated model now uses **LSTM (Long Short-Term Memory)** networks to capture temporal patterns in your exogenous variables while maintaining your hierarchical structure: **y_hat = Ax_1 + Bx_2 + Cx_3**

## Key Architecture Changes

### 1. **Temporal Data Processing**
Instead of treating each frame independently, we now create **sequences of 10 frames**:
- Input: `[batch_size, sequence_length=10, features]`
- Each sequence predicts the gold difference at the **last frame**
- Overlapping sequences (stride=3 for training, stride=5 for validation/test)

### 2. **LSTM for Each Exogenous Variable Group**

#### **A: Damage Sequences (x_1)**
```
Input: [batch, 10 timesteps, 6 damage features]
↓
Bidirectional LSTM (32 hidden units)
↓ 
Takes last timestep output: [batch, 64]
↓
MLP: 64 → 32 → 1
↓
damage_score: [batch, 1]
```

#### **B: Vision Sequences (x_2)**  
```
Input: [batch, 10 timesteps, 3 vision features]
↓
Bidirectional LSTM (16 hidden units)
↓
Takes last timestep output: [batch, 32]
↓
MLP: 32 → 16 → 1
↓
vision_score: [batch, 1]
```

#### **G1-G5: Player Stat Group Sequences (for each of 10 players)**

For each player (p_1 to p_10), each stat group processes temporal sequences:

- **G1 (Offensive)**: `[batch, 10, 10 features]` → LSTM(32) → 8-dim embedding
- **G2 (Defensive)**: `[batch, 10, 4 features]` → LSTM(16) → 8-dim embedding
- **G3 (Vampirism)**: `[batch, 10, 4 features]` → LSTM(16) → 8-dim embedding
- **G4 (Resource)**: `[batch, 10, 2 features]` → LSTM(8) → 8-dim embedding
- **G5 (Mobility)**: `[batch, 10, 3 features]` → LSTM(8) → 8-dim embedding

### 3. **Player Score Aggregation (F)**

For each player:
```
Concatenate [g1_emb, g2_emb, g3_emb, g4_emb, g5_emb]: [batch, 40]
↓
MLP: 40 → 32 → 16
↓
player_score: [batch, 16] (16-dimensional player representation)
```

### 4. **Team Aggregation with Attention (D & E)**

```
Team 1 (players 1-5): [batch, 5 players, 16 features]
Team 2 (players 6-10): [batch, 5 players, 16 features]
↓
Self-Attention mechanism (Q, K, V projections)
- Learns which players matter more in current game state
- Allows "fed carry" to dominate team score
↓
Attention pooling + MLP
↓
t1, t2: [batch, 16] each
```

### 5. **Team Interaction (C)**

```
Concatenate [t1, t2]: [batch, 32]
↓
MLP: 32 → 64 → 32 → 1
↓
team_score: [batch, 1]
```

### 6. **Final Combination**

```
Concatenate [damage_score, vision_score, team_score]: [batch, 3]
↓
MLP: 3 → 32 → 1
↓
y_hat: [batch, 1]
```

## Why This Works Better

### **Temporal Understanding**
- **LSTM cells** remember past states and detect trends
- Captures: "Is gold gap widening or closing?"
- Detects: "Did team just take Baron?" (sudden gold spike pattern)
- Understands: "Vision control → objective setup → gold gain" (multi-step causality)

### **No Sigmoid Bottlenecks**
- Removed all sigmoids that compressed output to (0,1)
- LSTM + LeakyReLU allow full range of values
- Gradients flow better through network

### **Cross-Feature Interactions**
- MLPs within each component model interactions (e.g., AD × Attack Speed)
- Attention mechanism models player synergies
- Team interaction MLP captures coordination effects

### **Robustness**
- **Huber Loss**: Less sensitive to outlier gold swings
- **BatchNorm**: Stabilizes training
- **Dropout**: Prevents overfitting
- **Weight Decay**: L2 regularization

## Model Capacity

**Estimated ~15,000-20,000 parameters** including:
- 6 Bidirectional LSTMs (A, B, G1-G5)
- 10 player processing pathways (F applied 10 times)
- 2 attention mechanisms (D, E)
- Multiple MLPs

This is **434x larger** than the original 49-parameter model, but:
- Still much smaller than the 914K parameter full LSTM
- Maintains your interpretable hierarchical structure
- Adds temporal reasoning capabilities

## Training Configuration

- **Optimizer**: Adam (better for LSTM than SGD)
- **Learning Rate**: 0.001
- **Loss**: Huber (delta=1.0)
- **Batch Size**: 64
- **Epochs**: 50
- **Sequence Length**: 10 frames
- **Data**: ~21,500 training sequences from 3,182 matches

## Output

The model predicts gold difference by:
1. Understanding how damage metrics **evolved** over last 10 frames
2. Understanding how vision control **changed** over time  
3. Understanding how each player's stats **progressed** over time
4. Combining these temporal patterns hierarchically

This respects your exogenous variable structure while adding temporal awareness!

