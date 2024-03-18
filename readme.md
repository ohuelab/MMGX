# MMGX: Multiple Molecular Graph eXplainable Discovery
Enhancing Model Learning and Interpretation Using Multiple Molecular Graph Representations for Compound Property and Activity Prediction 

![graphical abstract](https://github.com/ohuelab/mmgx/blob/main/blob/graphicalabstract.png?raw=true)

## Usage 💻

### 1. Install environment

This code was tested in Python 3.8 with PyTorch 1.13 and rdkit 2023.3.2
- Using [Conda](https://www.anaconda.com/):
`conda create -f mmgx.yaml`
- Then, activate the environment
`conda activate mmgx`

### 2. Prepare dataset

- Prepare dataset in `dataset/` folder. Dataset should be in `.csv` format with `smiles`, `label`, and `splitting` columns.
- Indicate the column name in `dataset/_dataset.csv` file.

### 3. Hyperparameter tuning

- [dataset] = name of dataset without `.csv` extension
- [model] = {GAT, GIN, GAT_edge, Benchmark_GCN, Benchmark_GIN, Benchmark_AttentiveFP}
- [schema] = {A (for atom graph only), AR_0 (for combination with pooling), R (for reduced graph)} 
- [reduced] = {functional, junctiontree, pharmacophore}

```bash
python3 hyperparameter.py \
-f [dataset] \
-m [model] \
--schema [schema] \
--reduced [reduced_(optional)] \
--mol_embedding 256 \
--batch_normalize \
--fold 5 \
--seed 42
```

- Examples

```bash
# Example, for Atom graph only model
python3 hyperparameter.py \
-f bbbp \
-m GIN \
--schema A \
--reduced \
--mol_embedding 256 \
--batch_normalize \
--fold 5 \
--seed 42

# Example, for Functional graph only model
python3 hyperparameter.py \
-f bbbp \
-m GIN \
--schema R \
--reduced functional \
--mol_embedding 256 \
--batch_normalize \
--fold 5 \
--seed 42

# Example, for 2-graph only model (Atom+Functional)
python3 hyperparameter.py \
-f bbbp \
-m GIN \
--schema AR_0 \
--reduced functional \
--mol_embedding 256 \
--batch_normalize \
--fold 5 \
--seed 42

# Example, for 3-graph model (Atom+Functional+Pharmacophore)
python3 hyperparameter.py \
-f bbbp \
-m GIN \
--schema AR_0 \
--reduced functional pharmacophore \
--mol_embedding 256 \
--batch_normalize \
--fold 5 \
--seed 42
```

### 4. Train and test the model

(All can be retrieved from hyperparameter tuning)
- [dataset] = name of dataset without `.csv` extension
- [model] = {GAT, GIN, GAT_edge, Benchmark_GCN, Benchmark_GIN, Benchmark_AttentiveFP}
- [schema] = {A (for atom graph only), AR_0 (for combination with pooling), R (for reduced graph)} 
- [reduced] = {functional, junctiontree, pharmacophore}
- [batch_size] = {batch size}
- [number_of_layer] = {number of node embedding layers for Atom graph}
- [number_of_layer_reduced] = {number of node embedding layers for reduced graph}
- [in_channels] = {number of input features}
- [hidden_channels] = {number of hidden features}
- [out_channels] = {number of output features}
- [number_of_layer_self] = {number of molecule embedding layers for Atom graph}
- [number_of_layer_self_reduced] = {number of molecule embedding layers for reduced graph}

```bash
python3 main.py \
-f [dataset] \
-m [model] \
--schema [schema] \
--reduced [reduced graph (optional)] \
--mol_embedding 256 \
--batch_normalize \
--fold 5 \
--seed 42 \
--batch_size [batch_size] \
--num_layers [number_of_layer] \
--num_layers_reduced [number_of_layer_reduced] \
--in_channels [in_channels] \
--hidden_channels [hidden_channels] \
--out_channels [out_channels] \
--num_layers_self [number_of_layer_self] \
--num_layers_self_reduced [number_of_layer_self_reduced] \
```

## Citation 📃
> - Kengkanna A, Ohue M. **Enhancing property and activity prediction and interpretation using multiple molecular graph representations with MMGX**. *Communications Chemistry*. (in press)
> - Kengkanna A, Ohue M. **Enhancing Model Learning and Interpretation Using Multiple Molecular Graph Representations for Compound Property and Activity Prediction**.
In *Proceedings of The 20th IEEE International Conference on Computational Intelligence in Bioinformatics and Computational Biology (CIBCB 2023)*, 8 pages, 2023. [doi: 10.1109/CIBCB56990.2023.10264879](https://doi.org/10.1109/CIBCB56990.2023.10264879)
