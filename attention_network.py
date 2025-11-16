"""
Attention-based neural networks for defender decision making.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class QKVNetwork(nn.Module):
    """
    Query-Key-Value network for attention mechanism.
    Transforms defender features into query, key, value representations.
    """

    def __init__(self, input_dim: int = 6, context_dim: int = 16):
        """
        Args:
            input_dim: Dimension of input features (defender features)
            context_dim: Dimension of output context vector
        """
        super().__init__()
        self.context_dim = context_dim

        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        # Query, Key, Value heads
        self.query_head = nn.Linear(32, context_dim)
        self.key_head = nn.Linear(32, context_dim)
        self.value_head = nn.Linear(32, context_dim)

    def forward(self, defender_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through QKV network.

        Args:
            defender_features: [batch_size, num_defenders, input_dim]

        Returns:
            queries, keys, values: Each [batch_size, num_defenders, context_dim]
        """
        # Extract shared features
        features = self.feature_net(defender_features)

        # Generate Q, K, V
        queries = self.query_head(features)
        keys = self.key_head(features)
        values = self.value_head(features)

        return queries, keys, values


class AttentionModule(nn.Module):
    """
    Attention module that computes context vectors for each defender.
    """

    def __init__(self, context_dim: int = 16):
        """
        Args:
            context_dim: Dimension of context vectors
        """
        super().__init__()
        self.context_dim = context_dim
        self.scale = np.sqrt(context_dim)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor,
                values: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Compute attention and generate context vectors.

        Args:
            queries: [batch_size, num_defenders, context_dim]
            keys: [batch_size, num_defenders, context_dim]
            values: [batch_size, num_defenders, context_dim]
            mask: Optional mask [batch_size, num_defenders, num_defenders]

        Returns:
            context_vectors: [batch_size, num_defenders, context_dim]
        """
        # Compute attention scores: Q @ K^T / sqrt(d)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale

        # Apply mask if provided (e.g., for dead defenders)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Compute context vectors: attention @ V
        context_vectors = torch.matmul(attention_weights, values)

        return context_vectors


class EvaluationNetwork(nn.Module):
    """
    Evaluation network that scores defender-attacker pairs.
    Takes enriched defender features (original + context) and attacker features.
    """

    def __init__(self, defender_dim: int = 6, context_dim: int = 16,
                 attacker_dim: int = 6, hidden_dim: int = 64):
        """
        Args:
            defender_dim: Dimension of defender features
            context_dim: Dimension of context vector
            attacker_dim: Dimension of attacker features
            hidden_dim: Hidden layer dimension
        """
        super().__init__()

        input_dim = defender_dim + context_dim + attacker_dim

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single score output
        )

    def forward(self, defender_features: torch.Tensor, context_vectors: torch.Tensor,
                attacker_features: torch.Tensor) -> torch.Tensor:
        """
        Compute scores for all defender-attacker pairs.

        Args:
            defender_features: [batch_size, num_defenders, defender_dim]
            context_vectors: [batch_size, num_defenders, context_dim]
            attacker_features: [batch_size, num_attackers, attacker_dim]

        Returns:
            scores: [batch_size, num_defenders, num_attackers]
        """
        batch_size = defender_features.shape[0]
        num_defenders = defender_features.shape[1]
        num_attackers = attacker_features.shape[1]

        # Concatenate defender features with context
        enriched_defenders = torch.cat([defender_features, context_vectors], dim=-1)
        # [batch_size, num_defenders, defender_dim + context_dim]

        # Expand dimensions for broadcasting
        # [batch_size, num_defenders, 1, defender_dim + context_dim]
        enriched_defenders = enriched_defenders.unsqueeze(2)
        # [batch_size, 1, num_attackers, attacker_dim]
        attacker_features = attacker_features.unsqueeze(1)

        # Broadcast and concatenate
        # [batch_size, num_defenders, num_attackers, defender_dim + context_dim]
        enriched_defenders = enriched_defenders.expand(-1, -1, num_attackers, -1)
        # [batch_size, num_defenders, num_attackers, attacker_dim]
        attacker_features = attacker_features.expand(-1, num_defenders, -1, -1)

        # Concatenate defender and attacker features
        # [batch_size, num_defenders, num_attackers, defender_dim + context_dim + attacker_dim]
        combined_features = torch.cat([enriched_defenders, attacker_features], dim=-1)

        # Evaluate all pairs
        # [batch_size, num_defenders, num_attackers, 1]
        scores = self.network(combined_features)

        # Remove last dimension
        scores = scores.squeeze(-1)  # [batch_size, num_defenders, num_attackers]

        return scores


class AttentionAllocationSystem(nn.Module):
    """
    Complete attention-based allocation system.
    Combines QKV network, attention, and evaluation network.
    """

    def __init__(self, defender_dim: int = 6, attacker_dim: int = 6,
                 context_dim: int = 16, hidden_dim: int = 64):
        """
        Args:
            defender_dim: Dimension of defender features
            attacker_dim: Dimension of attacker features
            context_dim: Dimension of context vectors
            hidden_dim: Hidden dimension for evaluation network
        """
        super().__init__()

        self.qkv_network = QKVNetwork(input_dim=defender_dim, context_dim=context_dim)
        self.attention = AttentionModule(context_dim=context_dim)
        self.evaluation = EvaluationNetwork(
            defender_dim=defender_dim,
            context_dim=context_dim,
            attacker_dim=attacker_dim,
            hidden_dim=hidden_dim
        )

    def forward(self, defender_features: torch.Tensor,
                attacker_features: torch.Tensor) -> torch.Tensor:
        """
        Compute allocation scores for all defender-attacker pairs.

        Args:
            defender_features: [batch_size, num_defenders, defender_dim]
            attacker_features: [batch_size, num_attackers, attacker_dim]

        Returns:
            scores: [batch_size, num_defenders, num_attackers]
        """
        # Generate Q, K, V from defender features
        queries, keys, values = self.qkv_network(defender_features)

        # Compute context vectors via attention
        context_vectors = self.attention(queries, keys, values)

        # Evaluate all defender-attacker pairs
        scores = self.evaluation(defender_features, context_vectors, attacker_features)

        return scores
