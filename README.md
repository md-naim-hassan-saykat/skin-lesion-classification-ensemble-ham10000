# skin-lesion-classification-ensemble-ham10000
Code and results for HAM10000 ensemble deep learning study.
## Data & Weights
- **HAM10000 dataset**: https://www.isic-archive.com/ (ISIC Archive)
- **Trained weights** (ViT and ensemble components):
  - GitHub Release v1.0: <link>
  - Zenodo DOI: <doi link>  (recommended for citation)
  - Hugging Face: <hf link> (optional)

### Reproduce
```bash
git clone https://github.com/<user>/skin-lesion-classification-ensemble-ham10000.git
cd skin-lesion-classification-ensemble-ham10000
# Download weights from the link above into output/<model_name>/
python infer.py --weights output/vit_ham10000_finetuned/model.safetensors
