# Amazon Berkeley Objects

The four unmodified original JPEGs in examples/originals are from the Amazon Berkeley Objects dataset, © Amazon.com, Inc. or its affiliates, used under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). [Official dataset and license](https://registry.opendata.aws/amazon-berkeley-objects/).

| Product ID | Split | Native width × height | Original file |
|---|---|---|---|
| B07FFHHG1Q | dev | 898 × 1831 | [90/90408e42.jpg](https://amazon-berkeley-objects.s3.amazonaws.com/images/original/90/90408e42.jpg) |
| B07QCR35JF | dev | 2006 × 2560 | [fe/fe1e14d8.jpg](https://amazon-berkeley-objects.s3.amazonaws.com/images/original/fe/fe1e14d8.jpg) |
| B074H6ZXT5 | test | 700 × 1000 | [45/45dc24b0.jpg](https://amazon-berkeley-objects.s3.amazonaws.com/images/original/45/45dc24b0.jpg) |
| B0752GQ94S | test | 870 × 2560 | [6a/6a9c0e39.jpg](https://amazon-berkeley-objects.s3.amazonaws.com/images/original/6a/6a9c0e39.jpg) |

All four SHA-256 values and URLs are in configs/products.json. Each committed JPEG was checked against a direct download of the listed official original. src/data.py validates the SHA, JPEG format and native dimensions on each load. For the 512² reference it fits the original inside the square using Lanczos downsampling and centers it on white. It refuses undersized sources and never enlarges the original. The public source detail is preserved before applying the separate 256² degradation. Reference tiles, degraded images, restoration outputs and contact sheets are local derivatives.

The earlier thumbnail experiment used ABO's separate small/ objects, whose long edge is 256 pixels, and enlarged them to form references. Its summary is retained only in results/pilot-thumbnail/. It is excluded from the native-resolution comparison.

