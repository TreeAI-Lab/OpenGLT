# Data layout

Counting raw data is from [`AK+'s released data`](https://github.com/LingxiaoShawn/GNNAsKernel/tree/main/data/subgraphcount/raw/), provided at:

```text
data/SubstructureCounting/raw/data.mat
```

TU, OGB, LRGB, ZINC, and AQSOL are downloaded automatically by their released package. 

CardBench, G-CARE, GNCE, and GOOD require download, with official links listed below. Place the remaining artifacts as follows:

```text
data/
|-- CardBench/{single_table,binary_join}/*.npz
|
|-- GCARE/raw/
|   |-- dataset/{aids,yago}/
|   |-- queryset/{aids,yago}/
|   |-- true_cardinality/{aids,yago}/
|
|-- GNCE/raw/
|   |-- <kg>/{star,path}/Joined_Queries.json
|   |-- <kg>/graph/<kg>.nt
|   |-- <kg>/<kg>_embeddings.json
|
|-- GOOD/{GOODHIV,GOODZINC}/
```

`<kg>` is one of `swdf`, `lubm`, `yago`, and `wikidata`.

## Links

| Dataset | Quick download | Official repository |
|---|---|---|
| **TU** | *(Automatically download)* | [TU Dataset collection](https://chrsmrrs.github.io/datasets/docs/datasets/) |
| **OGB** | *(Automatically download)* | [snap-stanford/ogb](https://github.com/snap-stanford/ogb) |
| **LRGB** | *(Automatically download)* | [vijaydwivedi75/lrgb](https://github.com/vijaydwivedi75/lrgb) |
| **ZINC** | *(Automatically download)* | [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) |
| **CardBench** | [Download instructions](https://github.com/google-research/google-research/blob/master/CardBench_zero_shot_cardinality_training/DowloadArtifacts.md) | [CardBench](https://github.com/google-research/google-research/tree/master/CardBench_zero_shot_cardinality_training) |
| **G-CARE** | [Query sets](https://drive.google.com/file/d/1Dlj43rBAOVPAsfzKlYxIbZ9RsqeGM_MN), [data graphs](https://drive.google.com/file/d/1HAgSVE-24NOap6_Q1_twH56Dkb2kPvGU) | [yspark-dblab/gcare](https://github.com/yspark-dblab/gcare) |
| **GNCE** | [Datasets and queries](https://nx36303.your-storageshare.de/s/wMbJJ2JLnkXcSE6) | [TimEricSchwabe/GNCE_publication](https://github.com/TimEricSchwabe/GNCE_publication) |
| **GOOD** | [GOOD-HIV](https://drive.google.com/file/d/1CoOqYCuLObnG5M0D8a2P2NyL61WjbCzo/view?usp=sharing) and [GOOD-ZINC](https://drive.google.com/file/d/1CHR0I1JcNoBqrqFicAZVKU3213hbsEPZ/view?usp=sharing) | [divelab/GOOD](https://github.com/divelab/GOOD) |


> *PS: Note for TU Datasets*
>
> Occasionally, raw files may be downloaded into a nested directory:
>
> ```text
> data/<dataset_name>/raw/<dataset_name>/
> ```
>
> This can cause:
>
> ```text
> TypeError: LocalFileSystem.mv() takes 3 positional arguments but 4 were given
> ```
>
> It can be fixed by moving the files up one level and rerun:
>
> ```bash
> cp data/<dataset_name>/raw/<dataset_name>/* data/<dataset_name>/raw/
> ```
>
