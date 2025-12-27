from torchinfo import summary
# You must provide an input_size so it can "run" a pass
summary(model, input_size=(1, BASE_CONFIG["context_length"]), dtypes=[torch.long])

==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
GPTModel                                 [1, 1024, 50257]          --
├─Embedding: 1-1                         [1, 1024, 768]            38,597,376
├─Embedding: 1-2                         [1024, 768]               786,432
├─Dropout: 1-3                           [1, 1024, 768]            --
├─Sequential: 1-4                        [1, 1024, 768]            --
│    └─TransformerBlock: 2-1             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-1               [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-2      [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-3                 [1, 1024, 768]            --
│    │    └─LayerNorm: 3-4               [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-5             [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-6                 [1, 1024, 768]            --
│    └─TransformerBlock: 2-2             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-7               [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-8      [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-9                 [1, 1024, 768]            --
│    │    └─LayerNorm: 3-10              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-11            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-12                [1, 1024, 768]            --
│    └─TransformerBlock: 2-3             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-13              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-14     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-15                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-16              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-17            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-18                [1, 1024, 768]            --
│    └─TransformerBlock: 2-4             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-19              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-20     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-21                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-22              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-23            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-24                [1, 1024, 768]            --
│    └─TransformerBlock: 2-5             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-25              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-26     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-27                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-28              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-29            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-30                [1, 1024, 768]            --
│    └─TransformerBlock: 2-6             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-31              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-32     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-33                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-34              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-35            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-36                [1, 1024, 768]            --
│    └─TransformerBlock: 2-7             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-37              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-38     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-39                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-40              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-41            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-42                [1, 1024, 768]            --
│    └─TransformerBlock: 2-8             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-43              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-44     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-45                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-46              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-47            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-48                [1, 1024, 768]            --
│    └─TransformerBlock: 2-9             [1, 1024, 768]            --
│    │    └─LayerNorm: 3-49              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-50     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-51                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-52              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-53            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-54                [1, 1024, 768]            --
│    └─TransformerBlock: 2-10            [1, 1024, 768]            --
│    │    └─LayerNorm: 3-55              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-56     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-57                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-58              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-59            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-60                [1, 1024, 768]            --
│    └─TransformerBlock: 2-11            [1, 1024, 768]            --
│    │    └─LayerNorm: 3-61              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-62     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-63                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-64              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-65            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-66                [1, 1024, 768]            --
│    └─TransformerBlock: 2-12            [1, 1024, 768]            --
│    │    └─LayerNorm: 3-67              [1, 1024, 768]            1,536
│    │    └─MultiHeadAttention: 3-68     [1, 1024, 768]            2,362,368
│    │    └─Dropout: 3-69                [1, 1024, 768]            --
│    │    └─LayerNorm: 3-70              [1, 1024, 768]            1,536
│    │    └─FeedForward: 3-71            [1, 1024, 768]            4,722,432
│    │    └─Dropout: 3-72                [1, 1024, 768]            --
├─LayerNorm: 1-5                         [1, 1024, 768]            1,536
├─Linear: 1-6                            [1, 1024, 50257]          38,597,376
==========================================================================================
Total params: 163,037,184
Trainable params: 163,037,184
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 967.52
==========================================================================================
Input size (MB): 0.01
Forward/backward pass size (MB): 1261.05
Params size (MB): 652.15
Estimated Total Size (MB): 1913.21
==========================================================================================