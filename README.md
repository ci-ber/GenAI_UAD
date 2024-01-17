# gen_ai_uad
Evaluation of normative learning or generative AI methods for Unsupervised Anomaly Detection 
<p align="center">
<img src="[https://github.com/ci-ber/autoDDPM/assets/106509806/91715b7d-beb2-4ce1-ab8e-917145c940d5](https://github.com/ci-ber/GenAI_UAD/assets/106509806/a1b6785e-0844-422b-911e-69da4e7aced1)" width=200>
</p>

<h1 align="center">
  <br>
Evaluating Normative Learning in Generative AI for Robust Medical Anomaly Detection
  <br>
</h1>
</h1>
  <p align="center">
    <a href="https://ci.bercea.net">Cosmin Bercea</a> •
    <a href="https://www.neurokopfzentrum.med.tum.de/neuroradiologie/mitarbeiter-profil-wiestler.html">Benedikt Wiestler</a> •
    <a href="https://aim-lab.io/author/daniel-ruckert/">Daniel Rueckert </a> •
    <a href="https://compai-lab.github.io/author/julia-a.-schnabel/">Julia A. Schnabel </a>
  </p>
<h4 align="center">Official repository of the paper</h4>
<h4 align="center">Under Review</h4>
<h4 align="center"> <a href="https://openreview.net/pdf?id=kTpafpXrqa](https://www.researchsquare.com/article/rs-3749187/v1)">Paper</a> </h4>

<p align="center">
<img src="https://github.com/ci-ber/autoDDPM/assets/106509806/54bebddf-d074-4eb9-82f3-3115f8625fc7">
</p>

## Citation

If you find our work useful, please cite our paper:
```
@article{bercea2023evaluating,
  title={Evaluating Normative Learning in Generative AI for Robust Medical Anomaly Detection},
  author={Bercea, Cosmin and Wiestler, Benedikt and Rueckert, Daniel and Schnabel, Julia},
  year={2023}
}
```

> **Abstract:** *In Generative Artificial Intelligence (AI) for medical imaging, normative learning involves training AI models on large datasets of typical images from healthy volunteers, such as MRIs or CT scans. These models acquire the distribution of normal anatomical structures, allowing them to effectively detect and correct anomalies in new, unseen pathological data. This approach allows the detection of unknown pathologies without the need for expert labeling.
Traditional anomaly detection methods often evaluate the anomaly detection performance, overlooking the crucial role of normative learning. In our analysis, we introduce novel metrics, specifically designed to evaluate this facet in AI models. We apply these metrics across various generative AI frameworks, including advanced diffusion models, and rigorously test them against complex and diverse brain pathologies. Our analysis demonstrates that models proficient in normative learning exhibit exceptional versatility, adeptly detecting a wide range of unseen medical conditions.* 

## Setup and Run

The code is based on the deep learning framework from the Institute of Machine Learning in Biomedical Imaging: https://github.com/compai-lab/iml-dl

### Framework Overview 

<p align="center">
<img src="https://github.com/ci-ber/autoDDPM/assets/106509806/678c5d6c-efb0-4934-a635-284b06636a78">
</p>

#### 1). Set up wandb (https://docs.wandb.ai/quickstart)

Sign up for a free account and login to your wandb account.
```bash
wandb login
```
Paste the API key from https://wandb.ai/authorize when prompted.

#### 2). Clone repository

```bash
git clone https://github.com/ci-ber/autoDDPM.git
cd autoDDPM
```

#### 3). Install requirements
*Optional* create virtual env:
```bash
conda create --name autoddpm python=3.8.0
conda activate autoddpm
```

```bash
pip install -r pip_requirements.txt
```

#### 4). Install PyTorch 

> Example installation: 
* *with cuda*: 
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
* *w/o cuda*:
```
pip3 install torch==1.9.1 torchvision==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 5). Download datasets 

<h4 align="center"><a href="https://brain-development.org/ixi-dataset/">IXI</a> • <a href="https://fastmri.org">FastMRI</a> • <a href="https://github.com/microsoft/fastmri-plus"> Labels for FastMRI</a> • <a href="https://fcon_1000.projects.nitrc.org/indi/retro/atlas.html">Atlas (Stroke) </a> </h4>

> Move the datasets to the target locations. You can find detailed information about the expected files and locations in the corresponding *.csv files under data/$DATASET/splits.

> *Alternatively you can use your own mid-axial slices of T1w brain scans with our <a href="https://www.dropbox.com/s/ooq7vdp9fp4ufag/latest_model.pt.zip?dl=0"> pre-trained weights</a> or train from scratch on other anatomies and modalities.*

#### 6). !!! Set the right threshold

You have to choose a threshold for binarizing the probable anomaly masks of the first step. Be mindful of this step since it can dramatically influence the outcome. In the paper, we use a threshold that delivers at most 5% false positive for inference *masking_threshold_infer: 0.13* in the *autoddpm.yaml* config file (for the given dataset). This has to be set for each dataset individually since the network might produce different errors on healthy data due to domain shifts! You can use the *thresholding* function in the *DownStreamEvaluator.py* to compute these on a healthy subsample of that distribution or use *-1* otherwise (This will automatically filter the 95% percentile of each scan individually). 

#### 7). Run the pipeline

Run the main script with the corresponding config like this:

```bash
python core/Main.py --config_path ./projects/autoddpm/autoddpm.yaml
```

Refer to the autoddpm.yaml for the default configuration. Store the pre-trained model from <a href="https://www.dropbox.com/s/ooq7vdp9fp4ufag/latest_model.pt.zip?dl=0"> HERE</a> into the specified directory to skip the training part.

By default, reconstructed images (from the first masking part of the pipeline) and inpainted images (after the second stitching and resampling part of the pipeline) are stored so that one can work on the parts of the pipeline in a modular way.

# That's it, enjoy! :rocket:
