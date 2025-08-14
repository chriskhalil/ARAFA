# Abstract
Automatic fact-checking poses a significant challenge in Arabic natural language processing due to the scarcity of datasets and resources. In this manuscript, we introduce ARAFA, a new large-scale dataset for fact-checking in Modern Standard Arabic, constructed through an automated framework leveraging large language models (LLMs). The dataset was constructed through a three-step pipeline: (1)claim generation from Arabic Wikipedia pages with supporting textual evidence,(2) claim mutation to generate challenging counterfactual claims with refuting ev-idence, and (3) an automatic validation step to validate that the generated claimsare either supported or refuted by their accompanying evidence, or if the evidencedoes not provide enough information to judge the validity of the claims. The resulting dataset comprises 181,976 claim-evidence pairs labeled as supported, refuted, or not enough information. Human evaluation carried out on a test sample from the dataset demonstrated strong inter-annotator agreement (κ = 0.89)using Cohen’s Kappa for supported claims and (κ = 0.94) for refuted claims. Automatic validation based on human-evaluated sample achieved 86% accuracy for supported claims and 88% for refuted ones. To showcase ARAFA’s value as a resource for automatic Arabic fact-checking, four open-source transformer-based models were fine-tuned using ARAFA, with the top-performing model achieving a Macro F1-score of 77% on the test data. In addition to ARAFA being the first large-scale dataset for Arabic fact-checking, our framework presents a scalable approach for developing similar resources for other low-resource languages.

## PrePrint
https://www.researchsquare.com/article/rs-7335564/v1


## Citation
```bibtex
@article{khalil2025arafa,
  title={ARAFA: An LLM Generated Arabic Fact-Checking Dataset},
  author={Khalil, Christophe and Elbassuoni, Shady and Assaf, Rida},
  year={2025},
  journal={Research Square},
  doi={10.21203/rs.3.rs-7335564/v1},
  url={https://doi.org/10.21203/rs.3.rs-7335564/v1}
}
```
## Original repository
Note: This Dataset is primeraly uploaded to zenodo as of official archive. https://doi.org/10.5281/zenodo.16762969
