from pathlib import Path

import torch
import torch.nn as nn

HOMEWORK_DIR = Path(__file__).resolve().parent
INPUT_MEAN = [0.2788, 0.2657, 0.2629]
INPUT_STD = [0.2064, 0.1944, 0.2252]


# THIS IS CLAUDE SONNET'S 4.5 VER 
class MLPPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
    ):
        """
        Args:
            n_track (int): number of points in each side of the track
            n_waypoints (int): number of waypoints to predict
        """
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Claude Sonnet 4.5

        # Calculate input and output dimensions
        # Input: 2 tracks (left + right) * n_track points * 2 coordinates (x, y)
        input_size = 2 * n_track * 2  # = 40 for default n_track=10
        
        # Output: n_waypoints * 2 coordinates (x, y)
        output_size = n_waypoints * 2  # = 6 for default n_waypoints=3
        
        # Define the MLP architecture
        hidden_size = 256
        
        # Claude came up with 4 but i can put more if i want
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Claude Sonnet 4.5

        # Get batch size
        B = track_left.shape[0]
        
        # Concatenate left and right tracks along the track dimension
        # (B, n_track, 2) + (B, n_track, 2) -> (B, 2*n_track, 2)
        x = torch.cat([track_left, track_right], dim=1)
        
        # Flatten the track points into a single feature vector
        # (B, 2*n_track, 2) -> (B, 2*n_track*2)
        x = x.view(B, -1)
        
        # Pass through the MLP network
        # (B, 40) -> (B, 6)
        x = self.network(x)
        
        # Reshape output to waypoint format
        # (B, n_waypoints*2) -> (B, n_waypoints, 2)
        waypoints = x.view(B, self.n_waypoints, 2)
        
        return waypoints


# Claude Sonnet 4.5
class TransformerPlanner(nn.Module):
    def __init__(
        self,
        n_track: int = 10,
        n_waypoints: int = 3,
        d_model: int = 64,
    ):
        super().__init__()

        self.n_track = n_track
        self.n_waypoints = n_waypoints

        # Claude Sonnet 4.5

        # Encode input track points to d_model dimensions
        self.track_encoder = nn.Linear(2, d_model)
        # Learned query embeddings (one per waypoint)
        self.query_embed = nn.Embedding(n_waypoints, d_model)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=256,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=3
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, 2)

    def forward(
        self,
        track_left: torch.Tensor,
        track_right: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Predicts waypoints from the left and right boundaries of the track.

        During test time, your model will be called with
        model(track_left=..., track_right=...), so keep the function signature as is.

        Args:
            track_left (torch.Tensor): shape (b, n_track, 2)
            track_right (torch.Tensor): shape (b, n_track, 2)

        Returns:
            torch.Tensor: future waypoints with shape (b, n_waypoints, 2)
        """
        # Claude Sonnet 4.5
        B = track_left.shape[0]
        
        # Concatenate tracks: (B, n_track, 2) + (B, n_track, 2) -> (B, 2*n_track, 2)
        track_points = torch.cat([track_left, track_right], dim=1)
        
        # Encode to d_model: (B, 20, 2) -> (B, 20, d_model)
        memory = self.track_encoder(track_points)
        
        # Get query embeddings: (n_waypoints, d_model) -> (B, n_waypoints, d_model)
        query_indices = torch.arange(self.n_waypoints, device=track_left.device)
        queries = self.query_embed(query_indices)
        queries = queries.unsqueeze(0).expand(B, -1, -1)
        
        # Apply transformer decoder
        decoder_output = self.transformer_decoder(tgt=queries, memory=memory)
        
        # Project to coordinates: (B, n_waypoints, d_model) -> (B, n_waypoints, 2)
        waypoints = self.output_projection(decoder_output)
        
        return waypoints


class PatchEmbedding(nn.Module):
    def __init__(self, h: int = 96, w: int = 128, patch_size: int = 8, in_channels: int = 3, embed_dim: int = 64):
        """
        Convert image to sequence of patch embeddings using a simple approach

        This is provided as a helper for implementing the Vision Transformer Planner.
        You can use this directly in your ViTPlanner implementation.

        Args:
            h: height of input image
            w: width of input image
            patch_size: size of each patch
            in_channels: number of input channels (3 for RGB)
            embed_dim: embedding dimension
        """
        super().__init__()
        self.h = h
        self.w = w
        self.patch_size = patch_size
        self.num_patches = (h // patch_size) * (w // patch_size)

        # Linear projection of flattened patches
        self.projection = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input images

        Returns:
            (B, num_patches, embed_dim) patch embeddings
        """
        B, C, H, W = x.shape
        p = self.patch_size

        # Reshape into patches: (B, C, H//p, p, W//p, p) -> (B, C, H//p, W//p, p, p)
        x = x.reshape(B, C, H // p, p, W // p, p).permute(0, 1, 2, 4, 3, 5)
        # Flatten patches: (B, C, H//p, W//p, p*p) -> (B, H//p * W//p, C * p * p)
        num_patches = (H // p) * (W // p)
        x = x.reshape(B, num_patches, C * p * p)

        # Linear projection
        return self.projection(x)
    
# Claude Sonnet 4.5
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        """
        A single Transformer encoder block with multi-head attention and MLP.

        You can use the one you implemented in Homework 3.

        Hint: A transformer block typically consists of:
        1. Layer normalization
        2. Multi-head self-attention (use torch.nn.MultiheadAttention with batch_first=True)
        3. Residual connection
        4. Layer normalization
        5. MLP (Linear -> GELU -> Dropout -> Linear -> Dropout)
        6. Residual connection

        Args:
            embed_dim: embedding dimension
            num_heads: number of attention heads
            mlp_ratio: ratio of MLP hidden dimension to embedding dimension
            dropout: dropout probability
        """
        super().__init__()

        # Claude Sonnet 4.5
         
        # Layer norm 1
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # Multi-head self-attention (no dropout)
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.0,
            batch_first=True
        )
        
        # Layer norm 2
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # MLP (no dropout)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, sequence_length, embed_dim) input sequence

        Returns:
            (batch_size, sequence_length, embed_dim) output sequence
        """
        # Claude Sonnet 4.5
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x



# Claude Sonnet 4.5
class ViTPlanner(nn.Module):
    def __init__(
        self,
        n_waypoints: int = 3,
        patch_size: int = 8,
        embed_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        
        self.n_waypoints = n_waypoints
        
        # Normalization (required by template)
        self.register_buffer("input_mean", torch.as_tensor([0.2788, 0.2657, 0.2629]), persistent=False)
        self.register_buffer("input_std", torch.as_tensor([0.2064, 0.1944, 0.2252]), persistent=False)
        
        # 1. Patch embedding - CHANGE dimensions to 96x128
        self.patch_embed = PatchEmbedding(
            h=96, w=128,  # Changed from 64x64!
            patch_size=patch_size, 
            in_channels=3, 
            embed_dim=embed_dim
        )
        num_patches = self.patch_embed.num_patches
        
        # 2. CLS token (same as yours)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 3. Positional embeddings (same as yours)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # 4. Transformer blocks - NO DROPOUT!
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim, 
                num_heads=num_heads, 
                mlp_ratio=mlp_ratio
            )  # Removed dropout parameter!
            for _ in range(num_layers)
        ])
        
        # 5. Output head - REGRESSION not classification!
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, n_waypoints * 2)  # Output 6 values (3 waypoints * 2 coords)
        )
    
    def forward(self, image: torch.Tensor, **kwargs) -> torch.Tensor:
        B = image.shape[0]
        
        # Normalize (required by template)
        x = (image - self.input_mean[None, :, None, None]) / self.input_std[None, :, None, None]
        
        # 1. Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # 2. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # 3. Add positional embeddings
        x = x + self.pos_embed
        
        # 4. Transformer blocks
        for block in self.transformer_layers:
            x = block(x)
        
        # 5. Take [CLS] token
        cls_representation = x[:, 0]  # (B, embed_dim)
        
        # 6. Regression head - output waypoints
        out = self.head(cls_representation)  # (B, 6)
        
        # 7. Reshape to waypoints
        waypoints = out.view(B, self.n_waypoints, 2)  # (B, 3, 2)
        
        return waypoints


MODEL_FACTORY = {
    "mlp_planner": MLPPlanner,
    "transformer_planner": TransformerPlanner,
    "vit_planner": ViTPlanner,
}


def load_model(
    model_name: str,
    with_weights: bool = False,
    **model_kwargs,
) -> torch.nn.Module:
    """
    Called by the grader to load a pre-trained model by name
    """
    m = MODEL_FACTORY[model_name](**model_kwargs)

    if with_weights:
        model_path = HOMEWORK_DIR / f"{model_name}.th"
        assert model_path.exists(), f"{model_path.name} not found"

        try:
            m.load_state_dict(torch.load(model_path, map_location="cpu"))
        except RuntimeError as e:
            raise AssertionError(
                f"Failed to load {model_path.name}, make sure the default model arguments are set correctly"
            ) from e

    # limit model sizes since they will be zipped and submitted
    model_size_mb = calculate_model_size_mb(m)

    if model_size_mb > 20:
        raise AssertionError(f"{model_name} is too large: {model_size_mb:.2f} MB")

    return m


def save_model(model: torch.nn.Module) -> str:
    """
    Use this function to save your model in train.py
    """
    model_name = None

    for n, m in MODEL_FACTORY.items():
        if type(model) is m:
            model_name = n

    if model_name is None:
        raise ValueError(f"Model type '{str(type(model))}' not supported")

    output_path = HOMEWORK_DIR / f"{model_name}.th"
    torch.save(model.state_dict(), output_path)

    return output_path


def calculate_model_size_mb(model: torch.nn.Module) -> float:
    """
    Naive way to estimate model size
    """
    return sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
