
## CNN-small
- Conv1: 1 -> 8 (3x3, pad=1)
- Conv2: 8 -> 16 (3x3, pad=1)
- Conv3: 16 -> 32 (3x3, pad=1)
- FC1: 288 -> 128
- FC2: 128 -> 10
- MaxPool after each conv (2x2)

## CNN-mid
- Conv1: 1 -> 32 (3x3, pad=1)
- Conv2: 32 -> 64 (3x3, pad=1)
- Conv3: 64 -> 128 (3x3, pad=1)
- FC1: 1152 -> 512
- FC2: 512 -> 10
- MaxPool after each conv (2x2)

## CNN-big
- Conv1: 1 -> 64 (3x3, pad=1)
- Conv2: 64 -> 128 (3x3, pad=1)
- Conv3: 128 -> 256 (3x3, pad=1)
- FC1: 2304 -> 1024
- FC2: 1024 -> 512
- FC3: 512 -> 10
- MaxPool after each conv (2x2)