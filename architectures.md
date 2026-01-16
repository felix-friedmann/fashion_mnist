
## CNN-small
- Conv1: 1 &rarr 8 (3x3, pad=1)
- Conv2: 8 &rarr 16 (3x3, pad=1)
- Conv3: 16 &rarr 32 (3x3, pad=1)
- FC1: 288 &rarr 128
- FC2: 128 &rarr 10
- MaxPool after each conv (2x2)

## CNN-mid
- Conv1: 1 &rarr 32 (3x3, pad=1)
- Conv2: 32 &rarr 64 (3x3, pad=1)
- Conv3: 64 &rarr 128 (3x3, pad=1)
- FC1: 1152 &rarr 512
- FC2: 512 &rarr 10
- MaxPool after each conv (2x2)

## CNN-big
- Conv1: 1 &rarr 64 (3x3, pad=1)
- Conv2: 64 &rarr 128 (3x3, pad=1)
- Conv3: 128 &rarr 256 (3x3, pad=1)
- FC1: 2304 &rarr 1024
- FC2: 1024 &rarr 512
- FC3: 512 &rarr 10
- MaxPool after each conv (2x2)